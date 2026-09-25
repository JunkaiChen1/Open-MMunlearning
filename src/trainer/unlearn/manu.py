import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers.trainer_utils import TrainOutput

from trainer.unlearn.base import UnlearnTrainer


logger = logging.getLogger(__name__)


DEFAULT_METRIC_WEIGHTS = {
    "I_abs": 2.0,
    "I_freq": 0.0,
    "I_var": 2.0,
    "I_rms": 2.0,
}


def compute_combined_importance(
    forget_scores, retain_scores, weights=None, epsilon=1e-5
):
    """Combine MANU activation statistics into per-neuron importance scores."""
    weights = dict(weights or DEFAULT_METRIC_WEIGHTS)
    combined = {}
    for metric, metric_weight in weights.items():
        if metric_weight == 0:
            continue
        for layer_name, forget_value in forget_scores.get(metric, {}).items():
            retain_value = retain_scores.get(metric, {}).get(layer_name)
            if retain_value is None:
                raise KeyError(
                    f"Retain activation statistics missing {metric}/{layer_name}"
                )
            if forget_value.shape != retain_value.shape:
                raise ValueError(
                    f"Activation score shape mismatch for {layer_name}: "
                    f"{tuple(forget_value.shape)} != {tuple(retain_value.shape)}"
                )
            score = metric_weight * (
                forget_value / (retain_value + epsilon) - 1.0
            )
            combined[layer_name] = combined.get(layer_name, 0.0) + score
    return combined


def compute_topk_masks(combined_scores, prune_percent):
    """Return 1D masks where one denotes a neuron selected for pruning."""
    if not combined_scores:
        raise ValueError("Cannot compute MANU masks from empty importance scores")
    if not 0 <= prune_percent <= 100:
        raise ValueError("prune_percent must be between 0 and 100")

    all_scores = torch.cat([score.flatten() for score in combined_scores.values()])
    if prune_percent == 0:
        return {
            name: torch.zeros_like(score, dtype=torch.bool)
            for name, score in combined_scores.items()
        }
    k = max(1, int((prune_percent / 100.0) * all_scores.numel()))
    k = min(k, all_scores.numel())
    threshold = torch.topk(all_scores, k, largest=True).values[-1]
    return {
        name: score >= threshold for name, score in combined_scores.items()
    }


def apply_output_neuron_masks(modules, masks):
    """Zero selected output rows and return the number of rows masked."""
    pruned = 0
    with torch.no_grad():
        for name, mask in masks.items():
            module = modules.get(name)
            if module is None:
                raise KeyError(f"No module registered for MANU mask {name}")
            if mask.ndim != 1 or mask.shape[0] != module.weight.shape[0]:
                raise ValueError(
                    f"MANU mask shape {tuple(mask.shape)} does not match "
                    f"{name} output size {module.weight.shape[0]}"
                )
            mask = mask.to(device=module.weight.device, dtype=torch.bool)
            lora_b = getattr(module, "lora_B", None)
            if lora_b:
                matched = False
                for adapter in lora_b:
                    adapter_weight = lora_b[adapter].weight
                    if adapter_weight.shape[0] != mask.shape[0]:
                        continue
                    adapter_weight[mask.to(adapter_weight.device)] = 0
                    matched = True
                if not matched:
                    raise ValueError(f"No LoRA output row matched MANU mask for {name}")
            else:
                module.weight[mask] = 0
                if module.bias is not None:
                    module.bias[mask] = 0
            pruned += mask.count_nonzero().item()
    return pruned


class _ActivationAccumulator:
    def __init__(self, modules, activation_threshold=0.1):
        self.modules = modules
        self.activation_threshold = activation_threshold
        self.sums = defaultdict(dict)
        self.counts = defaultdict(int)
        self.handles = []

    def _hook(self, layer_name):
        def collect(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            values = output.detach().float()
            reduce_dims = tuple(range(values.ndim - 1))
            metrics = {
                "I_abs": values.abs().mean(dim=reduce_dims),
                "I_freq": (values.abs() > self.activation_threshold)
                .float()
                .mean(dim=reduce_dims),
                "I_var": values.std(dim=reduce_dims, unbiased=False),
                "I_rms": values.square().mean(dim=reduce_dims).sqrt(),
            }
            for metric, score in metrics.items():
                score = score.cpu()
                previous = self.sums[metric].get(layer_name)
                if previous is None:
                    self.sums[metric][layer_name] = score
                else:
                    previous.add_(score)
            self.counts[layer_name] += 1

        return collect

    def __enter__(self):
        self.handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in self.modules.items()
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def finalize(self):
        scores = defaultdict(dict)
        for metric, layer_scores in self.sums.items():
            for layer_name, score in layer_scores.items():
                count = self.counts[layer_name]
                if count == 0:
                    raise ValueError(f"No activations collected for {layer_name}")
                scores[metric][layer_name] = score / count
        return dict(scores)


class MANU(UnlearnTrainer):
    """Modality-aware neuron pruning for LLaVA-style and Qwen2.5-VL MLLMs."""

    def __init__(
        self,
        prune_percent=10.0,
        num_iterations=1,
        activation_num_batches=None,
        activation_threshold=0.1,
        metric_weights=None,
        epsilon=1e-5,
        all_layers=False,
        vision_layers=None,
        language_layers=None,
        validate_after_pruning=False,
        source_adapter_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 <= prune_percent <= 100:
            raise ValueError("prune_percent must be between 0 and 100")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if activation_num_batches is not None and activation_num_batches <= 0:
            raise ValueError("activation_num_batches must be positive or null")

        self.prune_percent = float(prune_percent)
        self.num_iterations = int(num_iterations)
        self.activation_num_batches = activation_num_batches
        self.activation_threshold = float(activation_threshold)
        self.metric_weights = dict(metric_weights or DEFAULT_METRIC_WEIGHTS)
        self.epsilon = float(epsilon)
        self.all_layers = all_layers
        self.vision_layers = tuple(vision_layers or (-3, -2, -1))
        self.language_layers = tuple(language_layers or (-3, -2, -1))
        self.validate_after_pruning = validate_after_pruning
        self.source_adapter_path = source_adapter_path
        if not self.source_adapter_path:
            raise ValueError("MANU requires source_adapter_path for reloadable output")
        self._applied_masks = {}

    def _analysis_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    @staticmethod
    def _normalize_layer_indices(indices, layer_count):
        normalized = []
        for index in indices:
            index = index + layer_count if index < 0 else index
            if not 0 <= index < layer_count:
                raise IndexError(
                    f"MANU layer index {index} is outside a {layer_count}-layer model"
                )
            if index not in normalized:
                normalized.append(index)
        return normalized

    def _resolve_llava_modules(self, model):
        """Resolve the CLIP-vision/LLaMA-language module layout."""
        if not hasattr(model, "vision_tower") or not hasattr(
            model, "language_model"
        ):
            return None

        vision_encoder = model.vision_tower.vision_model.encoder
        language_decoder = model.language_model.model
        vision_indices = (
            list(range(len(vision_encoder.layers)))
            if self.all_layers
            else self._normalize_layer_indices(
                self.vision_layers, len(vision_encoder.layers)
            )
        )
        language_indices = (
            list(range(len(language_decoder.layers)))
            if self.all_layers
            else self._normalize_layer_indices(
                self.language_layers, len(language_decoder.layers)
            )
        )

        vision_modules = {}
        for index in vision_indices:
            mlp = vision_encoder.layers[index].mlp
            prefix = f"vision_tower.vision_model.encoder.layers.{index}.mlp"
            vision_modules[f"{prefix}.fc1"] = mlp.fc1
            vision_modules[f"{prefix}.fc2"] = mlp.fc2

        language_modules = {}
        for index in language_indices:
            mlp = language_decoder.layers[index].mlp
            prefix = f"language_model.model.layers.{index}.mlp"
            language_modules[f"{prefix}.gate_proj"] = mlp.gate_proj
            language_modules[f"{prefix}.up_proj"] = mlp.up_proj
            language_modules[f"{prefix}.down_proj"] = mlp.down_proj

        return (
            "llava",
            model.vision_tower,
            language_decoder,
            vision_modules,
            language_modules,
        )

    def _resolve_qwen2_5_vl_modules(self, model):
        """Resolve Qwen2.5-VL's `visual` and `model` module layout."""
        visual_encoder = getattr(model, "visual", None)
        language_decoder = getattr(model, "model", None)
        vision_blocks = getattr(visual_encoder, "blocks", None)
        language_layers = getattr(language_decoder, "layers", None)
        if vision_blocks is None or language_layers is None:
            return None

        vision_indices = (
            list(range(len(vision_blocks)))
            if self.all_layers
            else self._normalize_layer_indices(
                self.vision_layers, len(vision_blocks)
            )
        )
        language_indices = (
            list(range(len(language_layers)))
            if self.all_layers
            else self._normalize_layer_indices(
                self.language_layers, len(language_layers)
            )
        )

        vision_modules = {}
        for index in vision_indices:
            mlp = vision_blocks[index].mlp
            prefix = f"visual.blocks.{index}.mlp"
            vision_modules[f"{prefix}.gate_proj"] = mlp.gate_proj
            vision_modules[f"{prefix}.up_proj"] = mlp.up_proj
            vision_modules[f"{prefix}.down_proj"] = mlp.down_proj

        language_modules = {}
        for index in language_indices:
            mlp = language_layers[index].mlp
            prefix = f"model.layers.{index}.mlp"
            language_modules[f"{prefix}.gate_proj"] = mlp.gate_proj
            language_modules[f"{prefix}.up_proj"] = mlp.up_proj
            language_modules[f"{prefix}.down_proj"] = mlp.down_proj

        return (
            "qwen2_5_vl",
            visual_encoder,
            language_decoder,
            vision_modules,
            language_modules,
        )

    def _resolve_model_modules(self, model):
        # The supported launcher merges the source adapter before MANU runs.
        # Retain PEFT resolution for direct API callers, while saved artifacts
        # always record masks against the underlying model's module names.
        model_for_modules = model.get_base_model() if isinstance(model, PeftModel) else model
        modules = self._resolve_llava_modules(model_for_modules)
        if modules is not None:
            return modules
        modules = self._resolve_qwen2_5_vl_modules(model_for_modules)
        if modules is not None:
            return modules
        raise TypeError(
            "MANU supports Llava-compatible models and Qwen2.5-VL module layouts"
        )

    def _select_view(self, batch, text_only):
        if text_only:
            if "text_only" not in batch:
                raise ValueError(
                    "MANU requires a text-only collator view. Set "
                    "collator.DataCollatorForMultimodalQADataset.args."
                    "include_text_only=true."
                )
            batch = batch["text_only"]
        return self._prepare_model_inputs(batch)

    def _collect_scores(
        self,
        loader,
        modules,
        forward,
        text_only,
        num_batches,
    ):
        accumulator = _ActivationAccumulator(
            modules, activation_threshold=self.activation_threshold
        )
        with accumulator, torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if batch_index >= num_batches:
                    break
                batch = self._prepare_inputs(batch)
                model_inputs = self._select_view(batch, text_only=text_only)
                forward(model_inputs)
                del batch, model_inputs
        scores = accumulator.finalize()
        if not scores:
            raise ValueError("MANU activation analysis collected no scores")
        return scores

    @staticmethod
    def _vision_forward(vision_tower, inputs, architecture):
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("MANU multimodal view does not contain pixel_values")
        if architecture == "qwen2_5_vl":
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError(
                    "MANU Qwen2.5-VL multimodal view needs image_grid_thw"
                )
            return vision_tower(pixel_values, grid_thw=image_grid_thw)

        # LLaVA-Next processors group image tiles as
        # (batch, patches, channels, height, width), while CLIP expects 4D input.
        if pixel_values.ndim == 5:
            pixel_values = pixel_values.flatten(0, 1)
        if pixel_values.ndim != 4:
            raise ValueError(
                "MANU vision input must be 4D or tiled 5D, got "
                f"{tuple(pixel_values.shape)}"
            )
        return vision_tower(pixel_values=pixel_values, return_dict=True)

    @staticmethod
    def _language_forward(language_decoder, inputs):
        return language_decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )

    def _run_pruning_iteration(
        self,
        base_model,
        forget_loader,
        retain_loader,
        num_batches,
    ):
        (
            architecture,
            vision_tower,
            language_decoder,
            vision_modules,
            language_modules,
        ) = self._resolve_model_modules(base_model)
        base_model.eval()

        forget_vision = self._collect_scores(
            forget_loader,
            vision_modules,
            lambda inputs: self._vision_forward(vision_tower, inputs, architecture),
            text_only=False,
            num_batches=num_batches,
        )
        retain_vision = self._collect_scores(
            retain_loader,
            vision_modules,
            lambda inputs: self._vision_forward(vision_tower, inputs, architecture),
            text_only=False,
            num_batches=num_batches,
        )
        forget_language = self._collect_scores(
            forget_loader,
            language_modules,
            lambda inputs: self._language_forward(language_decoder, inputs),
            text_only=True,
            num_batches=num_batches,
        )
        retain_language = self._collect_scores(
            retain_loader,
            language_modules,
            lambda inputs: self._language_forward(language_decoder, inputs),
            text_only=True,
            num_batches=num_batches,
        )

        vision_importance = compute_combined_importance(
            forget_vision,
            retain_vision,
            weights=self.metric_weights,
            epsilon=self.epsilon,
        )
        language_importance = compute_combined_importance(
            forget_language,
            retain_language,
            weights=self.metric_weights,
            epsilon=self.epsilon,
        )
        vision_masks = compute_topk_masks(vision_importance, self.prune_percent)
        language_masks = compute_topk_masks(
            language_importance, self.prune_percent
        )
        for name, mask in {**vision_masks, **language_masks}.items():
            previous = self._applied_masks.get(name)
            self._applied_masks[name] = (
                mask.cpu() if previous is None else previous | mask.cpu()
            )
        pruned_vision = apply_output_neuron_masks(vision_modules, vision_masks)
        pruned_language = apply_output_neuron_masks(
            language_modules, language_masks
        )
        return pruned_vision, pruned_language

    def _validate_pruned_model(self, model, loader):
        batch = self._prepare_inputs(next(iter(loader)))
        model_inputs = self._select_view(batch, text_only=False)
        model.eval()
        with torch.inference_mode():
            outputs = model(**model_inputs)
        if outputs.loss is not None and not torch.isfinite(outputs.loss):
            raise RuntimeError("MANU produced a non-finite post-pruning loss")
        return float(outputs.loss) if outputs.loss is not None else None

    def train(self, *args, **kwargs):
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "MANU pruning currently requires a single accelerator process"
            )
        train_dataset = self.train_dataset
        if not hasattr(train_dataset, "forget") or not hasattr(
            train_dataset, "retain"
        ):
            raise TypeError("MANU requires ForgetRetainDataset")
        if train_dataset.forget is None or train_dataset.retain is None:
            raise ValueError("MANU requires both forget and retain datasets")

        forget_loader = self._analysis_loader(train_dataset.forget)
        retain_loader = self._analysis_loader(train_dataset.retain)
        num_batches = min(len(forget_loader), len(retain_loader))
        if self.activation_num_batches is not None:
            num_batches = min(num_batches, self.activation_num_batches)
        if num_batches == 0:
            raise ValueError("MANU received an empty analysis dataset")

        base_model = self.accelerator.unwrap_model(self.model)
        metrics = {"train_loss": 0.0}
        for iteration in range(self.num_iterations):
            pruned_vision, pruned_language = self._run_pruning_iteration(
                base_model,
                forget_loader,
                retain_loader,
                num_batches,
            )
            step = iteration + 1
            metrics[f"manu_iteration_{step}_pruned_vision_neurons"] = float(
                pruned_vision
            )
            metrics[f"manu_iteration_{step}_pruned_language_neurons"] = float(
                pruned_language
            )
            logger.info(
                "MANU iteration %d pruned %d vision and %d language neurons",
                step,
                pruned_vision,
                pruned_language,
            )

        if self.validate_after_pruning:
            loss = self._validate_pruned_model(base_model, forget_loader)
            if loss is not None:
                metrics["manu_post_pruning_forget_loss"] = loss
            logger.info("MANU post-pruning multimodal forward completed")

        self.state.global_step += self.num_iterations
        self.log(metrics)
        return TrainOutput(
            global_step=self.state.global_step,
            training_loss=0.0,
            metrics=metrics,
        )

    def save_model(self, output_dir=None, _internal_call=False):
        if not self._applied_masks:
            raise RuntimeError("MANU cannot save before pruning masks are built")
        output_dir = Path(output_dir or self.args.output_dir)
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            masks_path = output_dir / "manu_masks.pt"
            torch.save(self._applied_masks, masks_path)
            artifact = {
                "version": 1,
                "type": "manu",
                "source_adapter_path": str(Path(self.source_adapter_path).resolve()),
                "masks_path": masks_path.name,
            }
            with (output_dir / "unlearning_artifact.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(artifact, handle, indent=2, sort_keys=True)
                handle.write("\n")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        self.accelerator.wait_for_everyone()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        model_inputs = self._select_view(inputs["forget"], text_only=False)
        outputs = model(**model_inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
