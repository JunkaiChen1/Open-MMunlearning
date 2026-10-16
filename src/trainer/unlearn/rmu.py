"""Borrowed implementation from https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py"""

import re
import torch
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


class RMU(GradDiff):
    def __init__(
        self,
        module_regex=(
            r"base_model\.model\.language_model\.model\.layers\.7",
            r"base_model\.model\.model\.layers\.7",
            r"language_model\.model\.layers\.7",
            r"model\.layers\.7",
        ),
        trainable_params_regex=None,
        steering_coeff=20,
        restore_all_grads_after_optimizer=False,
        *args,
        **kwargs,
    ):
        """
        RMU Trainer that fine-tunes only specific layers and parameters using regex-based filtering.

        Args:
            module_path (str): Regex pattern to match module names.
            trainable_param_paths (list of str): List of regex patterns for trainable parameters.
        """
        super().__init__(*args, **kwargs)

        # Create reference model if not already set
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        # Regex for selecting params. None preserves the model's existing trainable
        # set, which keeps LoRA-only training from unfreezing base parameters.
        self.trainable_params_regex = trainable_params_regex

        # Get actual module references
        self.module_regex = module_regex  # Regex for selecting modules
        self.model_module = self._get_matching_module(self.model, self.module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.restore_all_grads_after_optimizer = restore_all_grads_after_optimizer
        self.control_vec = None

    def create_optimizer(self):
        if self.trainable_params_regex is not None:
            self._freeze_all_params(self.model, False)
            matched_params = self._set_trainable_params(
                self.model, self.trainable_params_regex, True
            )
            if matched_params == 0:
                raise ValueError(
                    "No parameters matched trainable_params_regex: "
                    f"{self.trainable_params_regex}"
                )
        super().create_optimizer()
        if (
            self.trainable_params_regex is not None
            and self.restore_all_grads_after_optimizer
        ):
            self._freeze_all_params(self.model, True)

    def _get_matching_module(self, model, module_regex):
        """Returns a single module matching the given regex from a DeepSpeed/DDP-wrapped model."""
        # Handle DeepSpeed and DDP-wrapped models by accessing the underlying module
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module  # Extract the actual PyTorch model inside

        if isinstance(module_regex, str):
            module_regexes = [module_regex]
        else:
            module_regexes = list(module_regex)

        matched_modules = {
            name: module
            for name, module in model.named_modules()
            if any(re.fullmatch(pattern, name) for pattern in module_regexes)
        }

        if len(matched_modules) > 1:
            raise ValueError(
                f"More than one module matched with {module_regexes}: {list(matched_modules.keys())}"
            )
        elif not matched_modules:
            raise ValueError(f"No module matched with {module_regexes}")

        return next(iter(matched_modules.values()))  # Return the single matched module

    def _freeze_all_params(self, model, requires_grad=True):
        """Freeze all parameters in the model initially."""
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        """Unfreeze specific parameters that match the regex patterns."""
        matched_params = 0
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in trainable_params_regex):
                param.requires_grad = requires_grad
                matched_params += 1
        return matched_params

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        """Performs a forward pass while caching the output of a specified module."""
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None

        hook_handle = module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(not no_grad):
                outputs = model(**inputs)
        finally:
            hook_handle.remove()

        if not cache:
            raise RuntimeError("The selected RMU module did not run during model forward")
        return cache[0], outputs

    def get_control_vector(self, dim):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = (
                random_vector / torch.norm(random_vector) * self.steering_coeff
            )
        return self.control_vec

    def compute_activation_loss(self, activation1, activation2, mask):
        if activation1.shape != activation2.shape:
            raise ValueError(
                "RMU activations must have identical shapes, got "
                f"{tuple(activation1.shape)} and {tuple(activation2.shape)}"
            )
        if activation1.ndim != 3:
            raise ValueError(
                "RMU activations must have shape [batch, sequence, hidden], got "
                f"{tuple(activation1.shape)}"
            )
        if tuple(mask.shape) != tuple(activation1.shape[:2]):
            raise ValueError(
                "RMU mask must match the activation batch and sequence dimensions, got "
                f"mask {tuple(mask.shape)} and activations {tuple(activation1.shape)}"
            )

        token_mask = mask.to(device=activation1.device, dtype=torch.bool)
        num_tokens = token_mask.sum(dim=-1)
        if torch.any(num_tokens == 0):
            raise ValueError("RMU requires at least one non-masked token per example")

        per_token_loss = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        ).mean(dim=-1)
        per_example_loss = (per_token_loss * token_mask).sum(dim=-1) / num_tokens
        return per_example_loss.mean()

    def compute_retain_loss(self, model, retain_inputs):
        retain_loss = 0.0

        if self.retain_loss_type == "EMBED_DIFF":
            model_retain_activations, _ = self.forward_with_cache(
                model, retain_inputs, module=self.model_module, no_grad=False
            )
            ref_retain_activations, _ = self.forward_with_cache(
                self.ref_model, retain_inputs, module=self.ref_module, no_grad=True
            )
            mask = retain_inputs["labels"] != -100  # Shape: [b, s]
            retain_loss = self.compute_activation_loss(
                model_retain_activations,
                ref_retain_activations.to(model_retain_activations.device),
                mask,
            )
        else:
            retain_loss = super().compute_retain_loss(model, retain_inputs)
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._prepare_model_inputs(inputs["forget"])

        model_forget_activations, forget_outputs = self.forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False
        )
        # If multiple datasets or concepts need unlearning, pass the control vector during processing; otherwise, default to a random vector during training.
        control_vec = forget_inputs.get(
            "control_vec", self.get_control_vector(model_forget_activations.shape[-1])
        )
        control_vec = control_vec.to(
            dtype=model_forget_activations.dtype, device=model_forget_activations.device
        )
        control_vec = control_vec.expand_as(model_forget_activations)
        mask = forget_inputs["labels"] != -100  # Shape: [b, s]
        forget_loss = self.compute_activation_loss(
            model_forget_activations, control_vec, mask
        )

        retain_inputs = self._prepare_model_inputs(inputs["retain"])
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
