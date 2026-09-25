"""Native MIP-Editor style path editing for multimodal unlearning.

The implementation keeps the algorithm inside the OpenUnlearning Trainer
contract: influential FFN paths are measured on forget batches, selected output
input channels are edited once, and subsequent updates steer the selected language
representation while preserving retain loss.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from trainer.unlearn.base import UnlearnTrainer


logger = logging.getLogger(__name__)


def _tensor_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def compute_masked_representation_loss(activations, target, labels):
    if activations.ndim != 3:
        raise ValueError(
            "MIP-Editor language activations must have shape [batch, sequence, hidden]"
        )
    if tuple(activations.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "MIP-Editor labels must match activation batch and sequence dimensions, got "
            f"activations {tuple(activations.shape)} and labels {tuple(labels.shape)}"
        )
    if target.shape[-1] != activations.shape[-1]:
        raise ValueError(
            "MIP-Editor control vector and activations must have the same hidden size"
        )
    token_mask = labels.ne(-100).to(activations.device)
    token_counts = token_mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise ValueError("MIP-Editor requires at least one supervised token per example")
    target = target.to(activations.device, activations.dtype).expand_as(activations)
    per_token = torch.nn.functional.mse_loss(
        activations.float(), target.float(), reduction="none"
    ).mean(dim=-1)
    return ((per_token * token_mask).sum(dim=-1) / token_counts).mean()


def apply_input_path_mask(module, linear, mask):
    """Zero selected FFN input channels in either LoRA or a plain linear layer."""
    mask = mask.to(dtype=torch.bool)
    with torch.no_grad():
        lora_a = getattr(module, "lora_A", None)
        if lora_a:
            matched = False
            for adapter in lora_a:
                adapter_weight = lora_a[adapter].weight
                if adapter_weight.shape[1] != mask.shape[0]:
                    continue
                adapter_weight[:, mask.to(adapter_weight.device)] = 0
                matched = True
            if not matched:
                raise ValueError("No LoRA input column matched MIP path mask")
            return
        if linear.weight.shape[1] != mask.shape[0]:
            raise ValueError(
                f"MIP path mask {tuple(mask.shape)} does not match linear input "
                f"size {linear.weight.shape[1]}"
            )
        linear.weight[:, mask.to(linear.weight.device)] = 0


class MIPEditor(UnlearnTrainer):
    """Influential-path editing followed by representation steering.

    This is a framework-native port of MIP-Editor's two essential operations:
    gradient-based path scoring and adaptive representation steering. It does
    not import or execute the external MIP-Editor repository.
    """

    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        retain_loss_type="NLL",
        path_topk=5,
        path_num_batches=4,
        path_layers=None,
        steering_coeff=1.0,
        retain_alpha=0.5,
        forget_beta=0.5,
        control_coeff=10.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if path_topk <= 0:
            raise ValueError("path_topk must be positive")
        if path_num_batches <= 0:
            raise ValueError("path_num_batches must be positive")
        if steering_coeff <= 0:
            raise ValueError("steering_coeff must be positive")
        self.path_topk = int(path_topk)
        self.path_num_batches = int(path_num_batches)
        self.path_layers = tuple(path_layers or ())
        self.steering_coeff = float(steering_coeff)
        self.retain_alpha = float(retain_alpha)
        self.forget_beta = float(forget_beta)
        self.control_coeff = float(control_coeff)
        self._path_ready = False
        self._path_masks = {}
        self._path_hooks = []
        self._target_name = None
        self._control = None
        self.gamma = float(gamma)
        self.retain_loss_type = retain_loss_type

    @staticmethod
    def _base_linear(module):
        base = getattr(module, "base_layer", module)
        if not isinstance(base, nn.Linear):
            return None
        return base

    def _resolve_path_modules(self, model):
        modules = []
        for name, module in model.named_modules():
            linear = self._base_linear(module)
            if linear is None or linear.weight.ndim != 2:
                continue
            is_ffn = name.endswith("mlp.down_proj") or name.endswith("mlp.fc2")
            if not is_ffn:
                continue
            if self.path_layers:
                layer_tokens = [int(token) for token in name.split(".") if token.isdigit()]
                if not layer_tokens or layer_tokens[-1] not in self.path_layers:
                    continue
            modules.append((name, module, linear))
        if not modules:
            raise ValueError(
                "MIPEditor found no FFN down_proj/fc2 modules. "
                "Check the selected multimodal model architecture."
            )
        language = [
            item
            for item in modules
            if "language_model" in item[0] or ".model.layers." in item[0]
        ]
        return modules, (language or modules)

    def _analysis_loader(self):
        dataset = getattr(self.train_dataset, "forget", None)
        if dataset is None:
            raise TypeError("MIPEditor requires a ForgetRetainDataset with a forget split")
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _build_path_edit(self, model):
        modules, language_modules = self._resolve_path_modules(model)
        importance = defaultdict(lambda: None)
        activations = {}
        handles = []

        def make_hook(name):
            def hook(_module, module_inputs):
                if not module_inputs:
                    raise RuntimeError(f"MIPEditor received no input for {name}")
                value = module_inputs[0]
                if not value.requires_grad:
                    value = value.requires_grad_()
                value.retain_grad()
                activations[name] = value

            return hook

        for name, module, _linear in modules:
            handles.append(module.register_forward_pre_hook(make_hook(name)))

        loader = self._analysis_loader()
        language_names = {name for name, _module, _linear in language_modules}
        model.train()
        batches = 0
        try:
            for batch in loader:
                if batches >= self.path_num_batches:
                    break
                batch = self._prepare_inputs(batch)
                model.zero_grad(set_to_none=True)
                inputs = self._prepare_model_inputs(batch)
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                    loss = outputs.loss
                self.accelerator.backward(loss)
                with torch.no_grad():
                    for name, value in activations.items():
                        grad = value.grad
                        if grad is None:
                            continue
                        score = (value.detach().float() * grad.detach().float()).abs()
                        if name in language_names and score.ndim == 3:
                            labels = inputs["labels"]
                            if tuple(score.shape[:2]) != tuple(labels.shape):
                                raise ValueError(
                                    "MIPEditor language path activations and labels have "
                                    "different batch/sequence dimensions"
                                )
                            valid_tokens = labels.ne(-100)
                            if not valid_tokens.any():
                                raise ValueError(
                                    "MIPEditor path analysis requires supervised tokens"
                                )
                            score = score[valid_tokens].mean(dim=0).cpu()
                        else:
                            score = score.reshape(-1, score.shape[-1]).mean(dim=0).cpu()
                        importance[name] = (
                            score
                            if importance[name] is None
                            else importance[name] + score
                        )
                batches += 1
                activations.clear()
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)

        if batches == 0:
            raise ValueError("MIPEditor path analysis received no forget batches")

        for name, module, linear in modules:
            score = importance.get(name)
            if score is None:
                continue
            score = score / batches
            count = min(self.path_topk, score.numel())
            selected = torch.topk(score, count, largest=True).indices
            mask = torch.zeros(score.numel(), dtype=torch.bool)
            mask[selected] = True
            self._path_masks[name] = mask
            try:
                apply_input_path_mask(module, linear, mask)
            except ValueError as error:
                raise ValueError(f"Could not apply MIP mask for {name}: {error}") from error

            # Keep edited FFN input paths fixed during the subsequent LoRA update.
            for parameter_name, parameter in module.named_parameters():
                if not parameter_name.endswith("lora_A.default.weight"):
                    continue
                device_mask = mask.to(parameter.device)
                self._path_hooks.append(
                    parameter.register_hook(
                        lambda grad, fixed=device_mask: grad.masked_fill(
                            fixed.view(1, -1), 0
                        )
                    )
                )

        if not self._path_masks:
            raise ValueError("MIPEditor could not score any FFN path")

        self._target_name = language_modules[-1][0]
        target_linear = language_modules[-1][2]
        hidden_size = target_linear.out_features
        control = torch.randn(1, 1, hidden_size, device=target_linear.weight.device)
        self._control = control / control.norm().clamp_min(1e-12) * self.steering_coeff
        self._path_ready = True
        logger.info(
            "MIPEditor edited %d FFN paths across %d analysis batches; target=%s",
            len(self._path_masks),
            batches,
            self._target_name,
        )

    def _forward_with_target(self, model, inputs):
        captured = []
        target = dict(model.named_modules()).get(self._target_name)
        if target is None:
            raise RuntimeError(f"MIPEditor target module disappeared: {self._target_name}")

        def hook(_module, _inputs, output):
            captured.append(_tensor_output(output))
            return output

        handle = target.register_forward_hook(hook)
        try:
            outputs = model(**inputs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("MIPEditor target activation was not captured")
        return outputs, captured[-1]

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self._path_ready:
            self._build_path_edit(model)
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._prepare_model_inputs(inputs["forget"])
        retain_inputs = self._prepare_model_inputs(inputs["retain"])
        forget_outputs, forget_activation = self._forward_with_target(model, forget_inputs)
        retain_outputs = model(**retain_inputs)
        target = self._control.to(
            forget_activation.device, dtype=forget_activation.dtype
        )
        forget_representation_loss = compute_masked_representation_loss(
            forget_activation,
            target,
            forget_inputs["labels"],
        )
        loss = (
            self.gamma * self.retain_alpha * retain_outputs.loss
            + self.forget_beta * self.control_coeff * forget_representation_loss
        )
        return (loss, forget_outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the edited LoRA adapter without materializing a full model."""
        output_dir = output_dir or self.args.output_dir
        model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            from pathlib import Path

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                output_dir,
                safe_serialization=getattr(self.args, "save_safetensors", True),
            )
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        self.accelerator.wait_for_everyone()
