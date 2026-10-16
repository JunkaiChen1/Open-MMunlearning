"""Image rephrasing attacks for multimodal unlearning evaluation.

The default transformations are deterministic and model-free.  Regenerated
images are supported through a precomputed image directory or a callback so
that an external diffusion/inpainting model is never silently introduced into
the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zlib import crc32

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from transformers import AutoProcessor

from evals.base import Evaluator
from evals.metrics.target_recovery import (
    recovery_percentage,
    target_hit,
    variant_recovery_rates,
)


@dataclass
class ImageRephraseConfig:
    """Configuration for deterministic image rephrasing variants."""

    variants: tuple[str, ...] = (
        "noise",
        "brightness",
        "contrast",
        "resize",
        "crop",
        "blur",
        "occlusion",
        "remove_irrelevant",
    )
    seed: int = 42
    noise_sigma: float = 0.02
    brightness_factor: float = 0.85
    contrast_factor: float = 0.85
    resize_scale: float = 0.85
    crop_ratio: float = 0.90
    rephrase_blur_radius: float = 2.0
    occlusion_ratio: float = 0.18
    occlusion_fill: str = "mean"
    remove_region: tuple[float, float, float, float] = (0.0, 0.0, 0.32, 0.32)
    remove_fill: str = "blur"
    blur_radius: float = 14.0
    regeneration_dir: str | None = None
    save_images: bool = True
    artifacts_dir: str | None = None
    prompt_style: str = "raw"
    max_new_tokens: int = 80

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ImageRephraseConfig":
        if value is None:
            return cls()
        names = {field.name for field in fields(cls)}
        data = {key: value[key] for key in names if key in value}
        if "variants" in data:
            data["variants"] = tuple(str(item) for item in data["variants"])
        if "remove_region" in data:
            data["remove_region"] = tuple(float(item) for item in data["remove_region"])
        return cls(**data)


class ImageRephraseAttack:
    """Generate image rephrases and compare their black-box answer recovery."""

    SUPPORTED_VARIANTS = {
        "noise",
        "brightness",
        "contrast",
        "resize",
        "crop",
        "blur",
        "occlusion",
        "remove_irrelevant",
        "regenerate",
    }

    def __init__(
        self,
        processor: Any,
        tokenizer: Any | None = None,
        config: ImageRephraseConfig | Mapping[str, Any] | None = None,
        regeneration_fn: Callable[[Image.Image, Mapping[str, Any]], Image.Image]
        | None = None,
    ):
        self.processor = processor
        self.tokenizer = tokenizer or getattr(processor, "tokenizer", None)
        self.config = (
            config
            if isinstance(config, ImageRephraseConfig)
            else ImageRephraseConfig.from_mapping(config)
        )
        unknown = set(self.config.variants) - self.SUPPORTED_VARIANTS
        if unknown:
            raise ValueError(
                f"Unsupported image rephrase variants: {sorted(unknown)}. "
                f"Supported variants: {sorted(self.SUPPORTED_VARIANTS)}"
            )
        self.regeneration_fn = regeneration_fn

    @staticmethod
    def _seed(base_seed: int, record_id: Any, variant: str) -> int:
        key = f"{int(base_seed)}:{record_id}:{variant}".encode("utf-8")
        return crc32(key)

    @staticmethod
    def _mean_fill(image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        color = tuple(
            np.clip(array.reshape(-1, 3).mean(axis=0), 0, 255).astype(np.uint8)
        )
        return Image.new("RGB", image.size, color=color)

    def _fill_image(self, image: Image.Image, mode: str) -> Image.Image:
        mode = str(mode).lower()
        if mode == "mean":
            return self._mean_fill(image)
        if mode == "black":
            return Image.new("RGB", image.size, color=(0, 0, 0))
        if mode == "white":
            return Image.new("RGB", image.size, color=(255, 255, 255))
        if mode == "blur":
            return image.filter(ImageFilter.GaussianBlur(self.config.blur_radius))
        raise ValueError(f"Unknown image rephrase fill mode: {mode}")

    def _noise(self, image: Image.Image, record_id: Any) -> Image.Image:
        rng = np.random.default_rng(self._seed(self.config.seed, record_id, "noise"))
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        noise = rng.normal(0.0, float(self.config.noise_sigma), size=array.shape)
        output = np.clip((array + noise) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")

    def _occlusion(self, image: Image.Image, record_id: Any) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        ratio = min(max(float(self.config.occlusion_ratio), 0.0), 0.9)
        rng = np.random.default_rng(
            self._seed(self.config.seed, record_id, "occlusion")
        )
        box_width = max(1, int(round(width * np.sqrt(ratio))))
        box_height = max(1, int(round(height * np.sqrt(ratio))))
        left = int(rng.integers(0, max(1, width - box_width + 1)))
        top = int(rng.integers(0, max(1, height - box_height + 1)))
        mask = Image.new("L", (width, height), color=0)
        ImageDraw.Draw(mask).rectangle(
            (left, top, left + box_width - 1, top + box_height - 1), fill=255
        )
        return Image.composite(
            self._fill_image(image, self.config.occlusion_fill), image, mask
        )

    def _brightness(self, image: Image.Image) -> Image.Image:
        factor = max(0.0, float(self.config.brightness_factor))
        return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)

    def _contrast(self, image: Image.Image) -> Image.Image:
        factor = max(0.0, float(self.config.contrast_factor))
        return ImageEnhance.Contrast(image.convert("RGB")).enhance(factor)

    @staticmethod
    def _scaled_back(image: Image.Image, scale: float) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(max(float(scale), 0.1), 1.0)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return resized.resize((width, height), Image.Resampling.LANCZOS)

    def _resize(self, image: Image.Image) -> Image.Image:
        return self._scaled_back(image, self.config.resize_scale)

    def _crop(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        ratio = min(max(float(self.config.crop_ratio), 0.1), 1.0)
        crop_width = max(1, round(width * ratio))
        crop_height = max(1, round(height * ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    def _blur(self, image: Image.Image) -> Image.Image:
        radius = max(0.0, float(self.config.rephrase_blur_radius))
        return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def _mask_from_record(value: Any, size: tuple[int, int]) -> Image.Image | None:
        if value is None or (
            isinstance(value, (float, np.floating)) and bool(np.isnan(value))
        ):
            return None
        if isinstance(value, Image.Image):
            mask = value.convert("L")
        elif isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.dtype != np.uint8 or (array.size and int(array.max()) <= 1):
                array = (np.asarray(array) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(array, mode="L")
        elif isinstance(value, (bytes, bytearray)):
            from io import BytesIO

            mask = Image.open(BytesIO(value)).convert("L")
        elif isinstance(value, str):
            mask = Image.open(value).convert("L")
        elif isinstance(value, dict) and value.get("bytes") is not None:
            from io import BytesIO

            mask = Image.open(BytesIO(value["bytes"])).convert("L")
        elif isinstance(value, dict) and value.get("path") is not None:
            mask = Image.open(value["path"]).convert("L")
        else:
            raise TypeError(f"Unsupported remove_mask type: {type(value)!r}")
        return mask.resize(size, Image.Resampling.NEAREST)

    def _remove_irrelevant(
        self, image: Image.Image, record: Mapping[str, Any]
    ) -> Image.Image:
        image = image.convert("RGB")
        mask = self._mask_from_record(record.get("remove_mask"), image.size)
        if mask is None:
            if len(self.config.remove_region) != 4:
                raise ValueError(
                    "remove_region must contain four normalized coordinates"
                )
            x0, y0, x1, y1 = self.config.remove_region
            width, height = image.size
            mask = Image.new("L", image.size, color=0)
            ImageDraw.Draw(mask).rectangle(
                (
                    int(width * min(x0, x1)),
                    int(height * min(y0, y1)),
                    int(width * max(x0, x1)),
                    int(height * max(y0, y1)),
                ),
                fill=255,
            )
        return Image.composite(
            self._fill_image(image, self.config.remove_fill), image, mask
        )

    def _regenerate(self, image: Image.Image, record: Mapping[str, Any]) -> Image.Image:
        candidate = record.get("regenerated_image")
        if candidate is None and self.regeneration_fn is not None:
            candidate = self.regeneration_fn(image.convert("RGB"), record)
        if candidate is None and self.config.regeneration_dir:
            root = Path(self.config.regeneration_dir).expanduser()
            record_id = str(record.get("id", "0")).replace("/", "_").replace(":", "_")
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                path = root / f"{record_id}{suffix}"
                if path.is_file():
                    candidate = path
                    break
        if candidate is None:
            raise ValueError(
                "The 'regenerate' variant requires record['regenerated_image'], "
                "a regeneration_dir containing <record-id>.png, or regeneration_fn."
            )
        if isinstance(candidate, Image.Image):
            return candidate.convert("RGB")
        if isinstance(candidate, (str, Path)):
            return Image.open(candidate).convert("RGB")
        raise TypeError(f"Unsupported regenerated_image type: {type(candidate)!r}")

    def transform(
        self, image: Image.Image, record: Mapping[str, Any], variant: str
    ) -> Image.Image:
        """Apply one named rephrase variant to an image."""
        record_id = record.get("id", 0)
        if variant == "noise":
            return self._noise(image, record_id)
        if variant == "brightness":
            return self._brightness(image)
        if variant == "contrast":
            return self._contrast(image)
        if variant == "resize":
            return self._resize(image)
        if variant == "crop":
            return self._crop(image)
        if variant == "blur":
            return self._blur(image)
        if variant == "occlusion":
            return self._occlusion(image, record_id)
        if variant == "remove_irrelevant":
            return self._remove_irrelevant(image, record)
        if variant == "regenerate":
            return self._regenerate(image, record)
        raise ValueError(f"Unsupported image rephrase variant: {variant}")

    def _prompt(self, text: str) -> str:
        if self.config.prompt_style.lower() == "raw":
            return f"USER: <image>\n{text}\nASSISTANT:"
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, model: Any, image: Image.Image, text: str) -> str:
        inputs = self.processor(
            text=[self._prompt(text)], images=[image], return_tensors="pt"
        )
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(self.config.max_new_tokens),
                do_sample=False,
            )
        input_length = inputs["input_ids"].shape[-1]
        generated = (
            outputs[:, input_length:] if outputs.shape[-1] > input_length else outputs
        )
        if hasattr(self.processor, "batch_decode"):
            return str(
                self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            ).strip()
        if self.tokenizer is None:
            return str(generated[0]).strip()
        return str(
            self.tokenizer.decode(generated[0], skip_special_tokens=True)
        ).strip()

    @staticmethod
    def _hit(answer: str, target: str) -> bool:
        return target_hit(answer, target)

    @staticmethod
    def _safe_id(value: Any) -> str:
        return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")

    @staticmethod
    def _checkpoint_dir(output_dir: str | Path) -> Path:
        path = Path(output_dir)
        return path if path.name == "checkpoint" else path / "checkpoint"

    def evaluate(
        self,
        model: Any,
        records: Sequence[Mapping[str, Any]],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate only the configured image rephrase variants."""
        model.eval()
        artifacts_dir = Path(
            self.config.artifacts_dir
            or (
                self._checkpoint_dir(output_dir) / "image_rephrase"
                if output_dir
                else "image_rephrase"
            )
        )
        if self.config.save_images:
            for variant in self.config.variants:
                (artifacts_dir / variant).mkdir(parents=True, exist_ok=True)
        details = []
        variant_hits = {variant: 0 for variant in self.config.variants}
        any_attack_hits = 0
        for index, record in enumerate(records):
            source = record["image"].convert("RGB")
            question = str(record["question"])
            target = str(record["answer"])
            variant_data = {}
            attack_hit = False
            for variant in self.config.variants:
                transformed = self.transform(source, record, variant)
                answer = self._generate(
                    model,
                    transformed,
                    question + "\nPlease provide only the correct answer.",
                )
                hit = self._hit(answer, target)
                variant_hits[variant] += int(hit)
                attack_hit = attack_hit or hit
                artifact = None
                if self.config.save_images:
                    artifact = (
                        artifacts_dir
                        / variant
                        / f"{index:05d}_{self._safe_id(record.get('id', index))}.png"
                    )
                    transformed.save(artifact)
                variant_data[variant] = {
                    "answer": answer,
                    "hit": hit,
                    "artifact": str(artifact) if artifact else None,
                }
            any_attack_hits += int(attack_hit)
            details.append(
                {
                    "id": str(record.get("id", index)),
                    "question": question,
                    "target": target,
                    "variants": variant_data,
                    "attack_success_at_b": attack_hit,
                }
            )
        total = len(details)
        if not total:
            raise ValueError("Image rephrase evaluation received no records.")
        variant_summary = variant_recovery_rates(variant_hits, total)
        return {
            "attack_success_at_b": recovery_percentage(any_attack_hits, total),
            "variants": variant_summary,
            "num_samples": total,
            "details": details,
        }


class ImageRephraseAttackEvaluator(Evaluator):
    """Run image rephrasing on MLLMU Test_Set images for forget IDs."""

    name = "ImageRephrase"

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(self.name, eval_cfg, **kwargs)
        self.processor = None

    @staticmethod
    def _to_container(value, default=None):
        if value is None:
            return default
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
        raise TypeError(f"Unsupported image rephrase image type: {type(value)!r}")

    def _read_parquet(self, path):
        path = Path(path).expanduser()
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No parquet files found at {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    def _data_path(self):
        explicit = self.eval_cfg.get("data_path", None)
        if explicit:
            return explicit
        root = Path(str(self.eval_cfg.data_root)).expanduser()
        split = str(self.eval_cfg.get("split", "test"))
        if split == "test":
            return root / "Test_Set"
        if split == "retain":
            ratio = int(self.eval_cfg.get("retain_ratio", 90))
            return root / f"retain_{ratio}" / "train-00000-of-00001.parquet"
        ratio = (
            split.removeprefix("forget_")
            if split.startswith("forget_")
            else str(self.eval_cfg.get("ratio", 10))
        )
        return root / f"forget_{ratio}" / "train-00000-of-00001.parquet"

    def _reference_ids(self):
        path = self.eval_cfg.get("reference_ids_path", None)
        if not path:
            return None
        frame = self._read_parquet(path)
        if "ID" not in frame:
            raise ValueError(f"Reference parquet {path} does not contain an ID column")
        return {str(value) for value in frame["ID"].tolist()}

    def _records(self):
        split = str(self.eval_cfg.get("split", "test"))
        reference_ids = self._reference_ids()
        image_strategy = str(self.eval_cfg.get("image_strategy", "all"))
        max_records = self.eval_cfg.get("max_records", None)
        records = []
        for _, row in self._read_parquet(self._data_path()).iterrows():
            row_id = str(row.get("ID", len(records)))
            if reference_ids is not None and row_id not in reference_ids:
                continue
            images = self._as_list(
                row.get("images") if split == "test" else row.get("image")
            )
            if image_strategy == "first":
                images = images[:1]
            tasks = self._as_list(row.get("Mask_Task", []))
            for image_index, image_value in enumerate(images):
                for task_index, task in enumerate(tasks):
                    if (
                        not isinstance(task, dict)
                        or task.get("Type") != "Image_Textual"
                    ):
                        continue
                    question = str(task.get("Question", "")).replace("__", "[Blank]")
                    question += "\nPlease provide only the correct answer for [Blank]."
                    answer = str(task.get("Ground_Truth", ""))
                    if not question.strip() or not answer.strip():
                        continue
                    records.append(
                        {
                            "id": f"{row_id}:{image_index}:{task_index}",
                            "image": self._image(image_value),
                            "question": question,
                            "answer": answer,
                            "remove_mask": row.get("remove_mask", None),
                        }
                    )
                    if max_records and len(records) >= int(max_records):
                        return records
        return records

    def _load_processor(self):
        if self.processor is None:
            processor_path = self.eval_cfg.get("processor_path", None)
            if not processor_path:
                raise ValueError("eval.image_rephrase.processor_path must be set.")
            self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        output_dir = output_dir or self.eval_cfg.output_dir
        output_dir = Path(output_dir)
        attack_cfg = (
            self._to_container(self.eval_cfg.get("image_rephrase", {}), {}) or {}
        )
        attack_cfg = dict(attack_cfg)
        attack_cfg.setdefault(
            "artifacts_dir", str(output_dir / "checkpoint" / "image_rephrase")
        )
        attack = ImageRephraseAttack(
            self._load_processor(),
            kwargs.get("tokenizer", None),
            ImageRephraseConfig.from_mapping(attack_cfg),
        )
        result = attack.evaluate(model, self._records(), output_dir=output_dir)
        result.update(
            {
                "attack": self.name,
                "split": str(self.eval_cfg.get("split", "test")),
                "reference_ids_path": self.eval_cfg.get("reference_ids_path", None),
                "config": attack_cfg,
            }
        )
        self.save_result(
            output_dir,
            result,
            self.summarize(result),
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        return result
