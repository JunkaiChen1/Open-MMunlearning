"""SUA universal visual perturbation attack.

This is an in-repository implementation of the attack released with
``MLLM-Unlearning-Attack``.  The original script hard-codes the MLLMU dataset,
the DnCNN path, CUDA device, and LLaVA prompt.  This module keeps the same
objective while making the model, processor, records, and output checkpoint
framework-owned and configurable.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn


logger = logging.getLogger("sua_attack")

_IGNORE_INDEX = -100
_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class DnCNN(nn.Module):
    """The DnCNN denoiser used by the official SUA implementation."""

    def __init__(
        self,
        depth: int = 17,
        n_channels: int = 64,
        image_channels: int = 3,
        use_bnorm: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()
        del use_bnorm  # Kept for compatibility with the released constructor.
        layers: list[nn.Module] = []
        padding = kernel_size // 2
        layers.extend(
            [
                nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            ]
        )
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(
            nn.Conv2d(
                n_channels,
                image_channels,
                kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images - self.dncnn(images)

    def _initialize_weights(self) -> None:
        last_conv = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if last_conv is not None:
            nn.init.constant_(last_conv.weight, 0)


@dataclass
class SUAConfig:
    """Serializable controls for the universal perturbation optimization."""

    iterations: int = 500
    alpha: float = 1.0 / 255.0
    epsilon: float = 12.0 / 255.0
    batch_size: int = 6
    denoise_weight: float = 0.7
    image_size: int = 336
    save_every: int = 20
    max_samples: int | None = None
    seed: int = 0
    prompt_style: str = "raw"
    denoiser_path: str | None = None
    checkpoint_path: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SUAConfig":
        if value is None:
            return cls()
        fields = set(cls.__dataclass_fields__)
        values = {key: value[key] for key in fields if key in value}
        return cls(**values)


class SUAUniversalPerturbationAttack:
    """Optimize and evaluate a universal image perturbation for a VLM.

    A record is a mapping containing ``image`` (PIL image or image bytes),
    ``question`` and ``answer``.  The attack is intentionally limited to
    image-conditioned records; pure-text questions cannot be affected by a
    visual perturbation and should be evaluated separately.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        tokenizer: Any | None = None,
        config: SUAConfig | Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self.config = (
            config if isinstance(config, SUAConfig) else SUAConfig.from_mapping(config)
        )
        self.device = torch.device(device or self._infer_device())
        self._mean = torch.tensor(_DEFAULT_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_DEFAULT_STD, device=self.device).view(1, 3, 1, 1)
        self._model_dtype = self._infer_model_dtype()
        self._original_requires_grad: dict[nn.Parameter, bool] = {}
        self.denoiser = self._load_denoiser()
        self.delta: torch.Tensor | None = None

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_model_dtype(self) -> torch.dtype:
        try:
            dtype = next(self.model.parameters()).dtype
            return (
                dtype
                if dtype in (torch.float16, torch.bfloat16, torch.float32)
                else torch.float32
            )
        except StopIteration:
            return torch.float32

    def _default_denoiser_path(self) -> Path:
        return (
            Path(__file__).resolve().parent / "assets" / "DnCNN" / "checkpoint.pth.tar"
        )

    def _load_denoiser(self) -> nn.Module:
        path = Path(
            self.config.denoiser_path or self._default_denoiser_path()
        ).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "SUA DnCNN checkpoint was not found at "
                f"{path}. Set eval.sua.sua.denoiser_path to a local checkpoint."
            )
        denoiser = DnCNN(image_channels=3, depth=17, n_channels=64)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        denoiser.load_state_dict(state)
        denoiser.to(self.device, dtype=torch.float32).eval()
        for parameter in denoiser.parameters():
            parameter.requires_grad_(False)
        return denoiser

    @staticmethod
    def _image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")
        if isinstance(value, (str, Path)):
            return Image.open(value).convert("RGB")
        raise TypeError(f"Unsupported SUA image type: {type(value)!r}")

    def _records(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        for record in records:
            if record.get("image") is None:
                continue
            question = str(record.get("question", "")).strip()
            answer = str(record.get("answer", "")).strip()
            if question and answer:
                prepared.append(
                    {
                        "id": str(record.get("id", len(prepared))),
                        "image": self._image(record["image"]),
                        "question": question,
                        "answer": answer,
                    }
                )
        if self.config.max_samples is not None:
            prepared = prepared[: max(0, int(self.config.max_samples))]
        if not prepared:
            raise ValueError("SUA requires at least one image-conditioned question.")
        return prepared

    def _raw_prompt(self, question: str, answer: str | None = None) -> str:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        return prompt if answer is None else f"{prompt} {answer}"

    def _prompt(self, question: str, answer: str | None = None) -> str:
        if self.config.prompt_style.lower() == "raw":
            return self._raw_prompt(question, answer)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
        if answer is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )

    def _processor_images(self, images: Sequence[Image.Image]) -> Sequence[Any]:
        if self.processor.__class__.__name__ == "Gemma3Processor":
            return [[image] for image in images]
        return images

    def _move_batch(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            return batch.to(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _prepare_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        delta: torch.Tensor | None,
        include_answers: bool,
    ) -> tuple[Any, torch.Tensor | None, torch.Tensor]:
        images = [record["image"] for record in records]
        questions = [self._prompt(record["question"]) for record in records]
        full_texts = [
            self._prompt(record["question"], record["answer"]) for record in records
        ]
        image_inputs = self._processor_images(images)
        processor_kwargs = {
            "text": full_texts if include_answers else questions,
            "images": image_inputs,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        batch = self._move_batch(self.processor(**processor_kwargs))
        prompt_batch = self._move_batch(
            self.processor(
                text=questions,
                images=image_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        )
        labels = None
        if include_answers:
            labels = batch["input_ids"].clone()
            prompt_attention = prompt_batch.get("attention_mask")
            if prompt_attention is None:
                prompt_attention = torch.ones_like(prompt_batch["input_ids"])
            full_attention = batch.get("attention_mask")
            if full_attention is None:
                full_attention = torch.ones_like(batch["input_ids"])
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            for row, prompt_len in enumerate(prompt_attention.sum(dim=1).tolist()):
                if padding_side == "left":
                    full_len = int(full_attention[row].sum().item())
                    start = max(0, labels.shape[1] - full_len)
                    labels[row, : min(labels.shape[1], start + int(prompt_len))] = (
                        _IGNORE_INDEX
                    )
                else:
                    labels[row, : min(labels.shape[1], int(prompt_len))] = _IGNORE_INDEX
            labels[full_attention == 0] = _IGNORE_INDEX
        if delta is not None:
            if "pixel_values" not in batch:
                raise ValueError("SUA processor output does not contain pixel_values.")
            clean_pixels = batch["pixel_values"].float()
            if clean_pixels.ndim == 5 and clean_pixels.shape[1] == 1:
                clean_pixels = clean_pixels[:, 0]
            clean_pixels = self._denormalize(clean_pixels)
            perturbed_pixels = (clean_pixels + delta).clamp(0.0, 1.0)
            batch["pixel_values"] = self._normalize(perturbed_pixels)
        return batch, labels, batch.get("pixel_values")

    def _denormalize(self, pixels: torch.Tensor) -> torch.Tensor:
        return pixels.float() * self._std + self._mean

    def _normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        return ((pixels.float() - self._mean) / self._std).to(self._model_dtype)

    def _vision_features(self, pixels: torch.Tensor) -> torch.Tensor:
        tower = getattr(self.model, "vision_tower", None)
        if tower is None and hasattr(self.model, "get_vision_tower"):
            tower = self.model.get_vision_tower()
        if tower is None:
            raise ValueError("SUA requires a model exposing vision_tower.")
        outputs = tower(pixels.to(self._model_dtype), output_hidden_states=True)
        layer_index = int(getattr(self.model.config, "vision_feature_layer", -2))
        features = outputs.hidden_states[layer_index][:, 1:]
        projector = getattr(self.model, "multi_modal_projector", None)
        if projector is not None:
            features = projector(features)
        return torch.nn.functional.normalize(features.mean(dim=1), dim=1)

    def _denoise_alignment_loss(
        self, perturbed_pixels: torch.Tensor, denoised_pixels: torch.Tensor
    ) -> torch.Tensor:
        perturbed_features = self._vision_features(perturbed_pixels)
        with torch.no_grad():
            denoised_features = self._vision_features(denoised_pixels)
        return -(perturbed_features * denoised_features).sum(dim=1).mean()

    def _target_loss(
        self, batch: Any, labels: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        inputs = dict(batch)
        inputs["pixel_values"] = pixel_values
        outputs = self.model(**inputs, labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is None or loss.numel() != 1:
            raise ValueError("SUA requires a scalar model loss from labels.")
        return loss

    def _set_frozen(self) -> None:
        self._original_requires_grad = {
            parameter: parameter.requires_grad for parameter in self.model.parameters()
        }
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _restore_grad_flags(self) -> None:
        for parameter, requires_grad in self._original_requires_grad.items():
            parameter.requires_grad_(requires_grad)

    def fit(self, records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """Optimize and return the universal perturbation in pixel space."""
        records = self._records(records)
        if self.config.batch_size < 1 or self.config.iterations < 1:
            raise ValueError("SUA batch_size and iterations must be positive.")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self._set_frozen()
        try:
            with torch.enable_grad():
                first_batch, _, first_pixels = self._prepare_batch(
                    records[:1],
                    delta=torch.zeros(1, 3, 1, 1, device=self.device),
                    include_answers=False,
                )
                del first_batch
                if first_pixels is None:
                    raise ValueError("SUA could not infer image tensor shape.")
                delta = torch.zeros_like(
                    first_pixels[0:1], dtype=torch.float32, device=self.device
                ).requires_grad_(True)
                best_loss = float("inf")
                for step in range(int(self.config.iterations)):
                    batch_size = min(int(self.config.batch_size), len(records))
                    indices = torch.randperm(len(records), generator=generator)[
                        :batch_size
                    ].tolist()
                    batch_records = [records[index] for index in indices]
                    batch, labels, perturbed_pixels = self._prepare_batch(
                        batch_records, delta=delta, include_answers=True
                    )
                    if labels is None or perturbed_pixels is None:
                        raise ValueError(
                            "SUA failed to prepare target labels or pixels."
                        )
                    clean_pixels = self._denormalize(perturbed_pixels)
                    # ``perturbed_pixels`` is normalized model input; recover the
                    # pixel-space batch before applying the denoiser.
                    denoised_pixels = self.denoiser(clean_pixels).clamp(0.0, 1.0)
                    normalized_denoised = self._normalize(denoised_pixels)
                    loss = self._target_loss(batch, labels, perturbed_pixels)
                    denoised_loss = self._target_loss(
                        batch, labels, normalized_denoised
                    )
                    alignment_loss = self._denoise_alignment_loss(
                        clean_pixels, denoised_pixels
                    )
                    total_loss = (
                        loss
                        + denoised_loss
                        + float(self.config.denoise_weight) * alignment_loss
                    )
                    if not torch.isfinite(total_loss):
                        logger.warning("Skipping non-finite SUA loss at step %s", step)
                        continue
                    gradient = torch.autograd.grad(
                        total_loss, delta, allow_unused=False
                    )[0]
                    if not torch.isfinite(gradient).all():
                        logger.warning(
                            "Skipping non-finite SUA gradient at step %s", step
                        )
                        continue
                    delta = (
                        delta.detach()
                        - float(self.config.alpha) * gradient.detach().sign()
                    ).clamp(-float(self.config.epsilon), float(self.config.epsilon))
                    delta.requires_grad_(True)
                    loss_value = float(loss.detach().cpu().item())
                    if loss_value < best_loss:
                        best_loss = loss_value
                    if self.config.checkpoint_path and (
                        step == 0
                        or (step + 1) % max(1, int(self.config.save_every)) == 0
                    ):
                        self.save(
                            delta,
                            self.config.checkpoint_path,
                            step=step + 1,
                            best_loss=best_loss,
                        )
                self.delta = delta.detach()
        finally:
            self._restore_grad_flags()
        if self.config.checkpoint_path:
            self.save(
                self.delta,
                self.config.checkpoint_path,
                step=int(self.config.iterations),
                best_loss=best_loss,
            )
        return self.delta

    def save(
        self,
        delta: torch.Tensor | None,
        path: str | Path,
        *,
        step: int | None = None,
        best_loss: float | None = None,
    ) -> None:
        if delta is None:
            raise ValueError("No SUA perturbation is available to save.")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "delta": delta.detach().float().cpu(),
            "config": asdict(self.config),
            "step": step,
            "best_loss": best_loss,
        }
        torch.save(payload, path)

    def load(self, path: str | Path) -> torch.Tensor:
        payload = torch.load(
            Path(path).expanduser(), map_location=self.device, weights_only=True
        )
        delta = payload["delta"] if isinstance(payload, dict) else payload
        self.delta = delta.to(self.device, dtype=torch.float32)
        return self.delta

    def _generate_one(
        self, record: Mapping[str, Any], delta: torch.Tensor | None
    ) -> str:
        batch, _, pixel_values = self._prepare_batch(
            [record], delta=delta, include_answers=False
        )
        if pixel_values is None:
            raise ValueError("SUA generation requires pixel_values.")
        with torch.no_grad():
            outputs = self.model.generate(**batch)
        input_length = batch["input_ids"].shape[-1]
        generated = (
            outputs[:, input_length:] if outputs.shape[-1] > input_length else outputs
        )
        decoder = getattr(self.processor, "batch_decode", None)
        if decoder is not None:
            return str(decoder(generated, skip_special_tokens=True)[0]).strip()
        return str(
            self.tokenizer.decode(generated[0], skip_special_tokens=True)
        ).strip()

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Measure target recovery on perturbed images only."""
        records = self._records(records)
        if self.delta is None:
            raise ValueError("Call fit() or load() before evaluate().")
        self.model.eval()
        attacked_hits = 0
        details = []
        for record in records:
            attacked_answer = self._generate_one(record, delta=self.delta)
            target = str(record["answer"]).lower()
            attacked_hit = target in attacked_answer.lower()
            attacked_hits += int(attacked_hit)
            details.append(
                {
                    "id": str(record.get("id", "")),
                    "question": record["question"],
                    "target": record["answer"],
                    "attacked_answer": attacked_answer,
                    "attacked_hit": attacked_hit,
                }
            )
        n = len(records)
        return {
            "sua_target_recovery_rate": 100.0 * attacked_hits / n,
            "num_samples": n,
            "details": details,
        }
