"""MIP-Editor path attribution and adaptive RMU within the Trainer interface.

The scoring, first-token pruning, selected-row updates and representation losses
follow MIP-Editor's fisher.py, ig.py, partial_linear.py and train_eval.py.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler

from model.mip_editor import PartialLinear, export_row_edits, install_path_hooks
from trainer.unlearn.base import UnlearnTrainer


logger = logging.getLogger(__name__)


def resolve_path_modules(model):
    language, vision = [], []
    for name, module in model.named_modules():
        if not name.endswith(("mlp.down_proj", "mlp.fc2")):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"MIP-Editor requires a merged, unquantized FFN: {name}")
        is_visual = "vision_tower" in name or "visual." in name
        (vision if is_visual else language).append(name)
    if not language or not vision:
        raise ValueError("MIP-Editor requires both language and vision FFN projections")
    return language, vision


def _scaled_path_inputs(activations, path, steps):
    scaled, increments = [], []
    for layer, activation in enumerate(activations):
        baseline = torch.zeros_like(activation)
        if layer < len(path):
            baseline = activation.clone()
            baseline[0, path[layer]] = 0
        increment = (activation - baseline) / steps
        scaled.append(torch.cat([baseline + increment * i for i in range(steps)]))
        increments.append(increment)
    return scaled, increments


def _accumulate_path_score(gradients, increments, path, steps, fisher):
    predictions = []
    for gradient in gradients:
        if fisher:
            gradient = (
                (gradient - gradient.mean(dim=0, keepdim=True))
                / (gradient.std(dim=0, keepdim=True) + 1e-12)
            ).square()
        predictions.append(gradient.reshape(steps, 1, -1))
    score, increment = predictions[-1], increments[-1]
    for layer, selected in enumerate(path):
        # Upstream assigns to the same slot for every selected neuron; the
        # last top-k entry consequently supplies the previous-layer term.
        score = score + predictions[layer][:, :, [selected[-1]]]
    score = score.sum(dim=0)
    for layer, selected in enumerate(path):
        increment = increment + increments[layer][:, [selected[-1]]]
    return score.cpu() * increment.cpu()


def compute_path_scores(model, inputs, projection_names, steps, topk, *, fisher):
    """Official sequential path attribution for one example (analysis batch=1)."""
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("MIP-Editor path discovery requires batch_size=1")
    if steps < 2 or topk < 1:
        raise ValueError("MIP-Editor needs at least two integration steps and topk > 0")
    modules = [model.get_submodule(name) for name in projection_names]
    activations = [None] * len(modules)
    handles = []
    try:
        for index, module in enumerate(modules):

            def capture(_module, args, _output, index=index):
                value = args[0]
                activations[index] = (
                    (value[:, 0] if value.ndim == 3 else value[0].unsqueeze(0))
                    .detach()
                    .clone()
                )

            handles.append(module.register_forward_hook(capture))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in activations):
        raise ValueError("MIP-Editor path analysis did not visit every projection")
    if any(topk > value.shape[-1] for value in activations):
        raise ValueError("MIP-Editor path_topk exceeds an FFN intermediate size")
    if fisher:
        repeated = {
            key: value.detach().repeat_interleave(steps, dim=0)
            for key, value in inputs.items()
        }
    else:
        # The official text IG forward reuses the original embedding output.
        repeated = {
            key: value.detach().repeat(steps, *([1] * (value.ndim - 1)))
            for key, value in inputs.items()
            if key != "input_ids"
        }
        repeated["inputs_embeds"] = (
            outputs.hidden_states[0].detach().repeat(steps, 1, 1)
        )
    del outputs

    path, scores = [], []
    for layer in range(len(modules)):
        scaled, increments = _scaled_path_inputs(activations[: layer + 1], path, steps)
        scaled = [value.detach().requires_grad_() for value in scaled]
        captured = [None] * len(modules)
        handles = []
        try:
            for index, module in enumerate(modules):
                if index <= layer:

                    def replace(_module, args, index=index):
                        value = args[0].clone()
                        if value.ndim == 3:
                            value[:, 0] = scaled[index]
                        else:
                            # Preserve upstream's packed-vision interpolation.
                            positions = torch.arange(
                                0, len(value), len(scaled[index]), device=value.device
                            )[: len(scaled[index])]
                            value[positions] = scaled[index][: len(positions)]
                        return (value, *args[1:])

                    handles.append(module.register_forward_pre_hook(replace))

                def capture(_module, args, _output, index=index):
                    captured[index] = args[0]

                handles.append(module.register_forward_hook(capture))
            loss = model(**repeated).loss
            gradients = list(
                torch.autograd.grad(loss, captured[: layer + 1], allow_unused=True)
            )
        finally:
            for handle in handles:
                handle.remove()
        if fisher and captured[0].ndim == 3 and layer == len(modules) - 1:
            if layer == 0:
                raise ValueError("Official vision Fisher needs at least two layers")
            # CLIP's final layer is not the selected image representation.
            gradients[-1] = gradients[-2]
        if any(gradient is None for gradient in gradients):
            raise ValueError("MIP-Editor attribution found a disconnected FFN")
        gradients = [
            gradient.reshape(steps, -1, gradient.shape[-1])[:, 0].detach()
            for gradient in gradients
        ]
        score = _accumulate_path_score(gradients, increments, path, steps, fisher)
        scores.append(score.squeeze(0))
        path.append(torch.topk(score, topk, dim=-1).indices.flatten().tolist())
    return torch.stack(scores)


class MIPEditor(UnlearnTrainer):
    """Discover multimodal paths, train selected rows, and save reloadable edits."""

    def __init__(
        self,
        path_topk=5,
        path_num_batches=None,
        fisher_steps=4,
        ig_steps=4,
        rmu_layer_id=7,
        steering_coeff=1.0,
        retain_alpha=0.5,
        forget_beta=0.5,
        control_coeff=10.0,
        source_adapter_path=None,
        *args,
        **kwargs,
    ):
        model = kwargs.get("model")
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            kwargs["model"] = model
        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        ):
            raise ValueError(
                "MIPEditor requires an unquantized model for selected-row training"
            )
        super().__init__(*args, **kwargs)
        if path_topk <= 0 or min(fisher_steps, ig_steps) < 2:
            raise ValueError("MIP-Editor requires topk > 0 and integration steps >= 2")
        if path_num_batches is not None and path_num_batches <= 0:
            raise ValueError("path_num_batches must be positive or null (all examples)")
        if steering_coeff <= 0:
            raise ValueError("steering_coeff must be positive")
        self.path_topk = int(path_topk)
        self.path_num_batches = path_num_batches
        self.fisher_steps = int(fisher_steps)
        self.ig_steps = int(ig_steps)
        self.rmu_layer_id = int(rmu_layer_id)
        self.steering_coeff = float(steering_coeff)
        self.retain_alpha = float(retain_alpha)
        self.forget_beta = float(forget_beta)
        self.control_coeff = float(control_coeff)
        self.source_adapter_path = source_adapter_path
        self._path_ready = False
        self._paths = {}
        self._path_hooks = []
        self._control = None
        self.ref_model = None

    def _analysis_loader(self):
        return DataLoader(
            self.train_dataset.forget,
            batch_size=1,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _build_path_edit(self):
        model = self.model
        language, vision = resolve_path_modules(model)
        if not 0 <= self.rmu_layer_id < len(language):
            raise ValueError("MIPEditor rmu_layer_id is outside the decoder layers")
        self._target_name = language[self.rmu_layer_id].rsplit(".mlp.", 1)[0]
        model.eval().requires_grad_(True)
        was_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        if was_checkpointing:
            model.gradient_checkpointing_disable()
        paths = {name: set() for name in language + vision}
        counts = {"text": 0, "vision": 0}
        try:
            for index, batch in enumerate(self._analysis_loader()):
                if self.path_num_batches is not None and index >= self.path_num_batches:
                    break
                batch = self._prepare_inputs(batch)
                if "text_only" not in batch:
                    raise ValueError(
                        "MIPEditor requires collator include_text_only=true"
                    )
                text = self._prepare_model_inputs(batch["text_only"])
                multi = self._prepare_model_inputs(batch)
                for names, inputs, steps, fisher, key in (
                    (vision, multi, self.fisher_steps, True, "vision"),
                    (language, text, self.ig_steps, False, "text"),
                ):
                    if fisher and inputs.get("pixel_values") is None:
                        continue
                    with self.compute_loss_context_manager():
                        scores = compute_path_scores(
                            model, inputs, names, steps, self.path_topk, fisher=fisher
                        )
                    # Official saved scores are bfloat16 before the final
                    # per-example top-k selection and union across samples.
                    indices = (
                        scores.to(torch.bfloat16).topk(self.path_topk, dim=-1).indices
                    )
                    for name, selected in zip(names, indices):
                        paths[name].update(selected.tolist())
                    counts[key] += 1
        finally:
            model.zero_grad(set_to_none=True)
            if was_checkpointing:
                model.gradient_checkpointing_enable()
        if not all(counts.values()):
            raise ValueError("MIPEditor needs forget examples with text and images")
        self._paths = {
            name: torch.tensor(sorted(selected), dtype=torch.long)
            for name, selected in paths.items()
        }
        self.ref_model = copy.deepcopy(model).eval().requires_grad_(False)
        model.requires_grad_(False)
        self._path_hooks = install_path_hooks(model, self._paths)
        # Upstream creates all language row parameters before the visual ones.
        for name in language + vision:
            ffn = model.get_submodule(name.rsplit(".", 1)[0])
            projections = (
                ("up_proj", "gate_proj") if hasattr(ffn, "up_proj") else ("fc1",)
            )
            for projection in projections:
                setattr(
                    ffn,
                    projection,
                    PartialLinear(getattr(ffn, projection), self._paths[name].tolist()),
                )
        config = getattr(model.config, "text_config", model.config)
        weight = next(model.parameters())
        control = torch.rand(
            1, 1, config.hidden_size, device=weight.device, dtype=weight.dtype
        )
        self._control = control / control.norm() * self.steering_coeff
        self._path_ready = True
        logger.info(
            "MIPEditor discovered paths from %s examples; target=%s",
            counts,
            self._target_name,
        )

    def _get_train_sampler(self, train_dataset=None):
        # Traverse retain once, pairing it with resampled forget examples via
        # ForgetRetainDataset(anchor='retain').
        return SequentialSampler(
            train_dataset if train_dataset is not None else self.train_dataset
        )

    def train(self, *args, **kwargs):
        if self.accelerator.num_processes != 1:
            raise ValueError(
                "MIPEditor currently requires a single accelerator process"
            )
        if self.args.gradient_checkpointing:
            raise ValueError(
                "MIPEditor requires gradient_checkpointing=false for representation hooks"
            )
        if not hasattr(self.train_dataset, "forget") or not hasattr(
            self.train_dataset, "retain"
        ):
            raise TypeError("MIPEditor requires ForgetRetainDataset")
        if self.train_dataset.forget is None or self.train_dataset.retain is None:
            raise ValueError("MIPEditor requires both forget and retain examples")
        self.train_dataset.anchor = "retain"
        if not self._path_ready:
            self._build_path_edit()
        # Build row parameters before Trainer creates the optimizer/wrapped model.
        return super().train(*args, **kwargs)

    def _forward_with_target(self, model, inputs):
        captured = []

        def capture(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)

        handle = model.get_submodule(self._target_name).register_forward_hook(capture)
        try:
            outputs = model(**inputs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("MIPEditor did not capture the decoder representation")
        return outputs, captured[0]

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget = self._prepare_model_inputs(inputs["forget"])
        retain = self._prepare_model_inputs(inputs["retain"])
        outputs, forget_activation = self._forward_with_target(model, forget)
        _, retain_activation = self._forward_with_target(model, retain)
        with torch.no_grad():
            _, frozen_activation = self._forward_with_target(self.ref_model, retain)
        target = (self._control * self.control_coeff).expand_as(forget_activation)
        forget_loss = F.mse_loss(forget_activation, target)
        retain_loss = F.mse_loss(retain_activation, frozen_activation)
        loss = self.retain_alpha * retain_loss + self.forget_beta * forget_loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        if not self._path_ready:
            raise RuntimeError("MIPEditor has no path edits to save; run train() first")
        output_dir = Path(output_dir or self.args.output_dir)
        if self.accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            torch.save(
                {
                    "paths": self._paths,
                    "rows": export_row_edits(model),
                },
                output_dir / "mip_editor_edits.pt",
            )
            config = {
                "type": "mip_editor",
                "source_adapter_path": (
                    str(Path(self.source_adapter_path).resolve())
                    if self.source_adapter_path is not None
                    else None
                ),
                "edits_path": "mip_editor_edits.pt",
            }
            with (output_dir / "unlearning_artifact.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(config, handle, indent=2)
            torch.save(self.args, output_dir / "training_args.bin")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        self.accelerator.wait_for_everyone()
