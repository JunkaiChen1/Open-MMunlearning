"""MIP-Editor row updates and token-path hooks shared by training and loading."""

from __future__ import annotations

import torch
from torch import nn


class PartialLinear(nn.Module):
    """Train selected output rows, matching the official PartialLinear.

    Upstream omits bias during training and restores it for inference.
    Keep that behavior, including for vision FFNs.
    """

    def __init__(self, original_linear, trainable_cols):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("MIP-Editor row editing requires an unquantized nn.Linear")
        self.original_linear = original_linear
        self.original_linear.requires_grad_(False)
        self.num_rows = original_linear.in_features
        self.num_columns = original_linear.out_features
        self.trainable_cols = sorted(set(int(i) for i in trainable_cols))
        if any(i < 0 or i >= self.num_columns for i in self.trainable_cols):
            raise ValueError("MIP-Editor selected row is outside the projection")
        self.frozen_cols = [
            i for i in range(self.num_columns) if i not in self.trainable_cols
        ]
        self.trainable_linear = nn.Linear(
            self.num_rows, len(self.trainable_cols), bias=False
        )
        self.trainable_linear.weight = nn.Parameter(
            original_linear.weight[self.trainable_cols].detach().clone()
        )
        self.trainable_linear.to(original_linear.weight.device)
        self.frozen_linear = nn.Linear(self.num_rows, len(self.frozen_cols), bias=False)
        self.frozen_linear.weight = nn.Parameter(
            original_linear.weight[self.frozen_cols].detach().clone(),
            requires_grad=False,
        )
        self.frozen_linear.to(original_linear.weight.device)
        self._merged_versions = None

    def forward(self, inputs):
        if not self.training:
            versions = (
                self.trainable_linear.weight._version,
                self.frozen_linear.weight._version,
            )
            if self._merged_versions != versions:
                self.merge_to_linear()
            return self.original_linear(inputs)
        self._merged_versions = None
        selected = self.trainable_linear(inputs)
        frozen = self.frozen_linear(inputs)
        # Autocast can change the projection dtype while residual inputs stay fp32.
        output = selected.new_zeros(*inputs.shape[:-1], self.num_columns)
        output[..., self.trainable_cols] = selected
        output[..., self.frozen_cols] = frozen
        return output

    def merge_to_linear(self):
        with torch.no_grad():
            self.original_linear.weight[self.trainable_cols] = (
                self.trainable_linear.weight
            )
            self.original_linear.weight[self.frozen_cols] = self.frozen_linear.weight
        self._merged_versions = (
            self.trainable_linear.weight._version,
            self.frozen_linear.weight._version,
        )
        return self.original_linear


def install_path_hooks(model, paths):
    """Mask the first token/flattened visual patch, never weight columns."""
    modules = dict(model.named_modules())
    handles = []
    try:
        for name, indices in paths.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"MIP-Editor path projection not found: {name}")
            indices = torch.as_tensor(indices, dtype=torch.long).flatten().tolist()
            if any(i < 0 or i >= module.in_features for i in indices):
                raise ValueError(f"MIP-Editor path index outside projection: {name}")

            def mask_first_token(_module, inputs, selected=indices):
                value = inputs[0]
                if value.ndim not in (2, 3):
                    raise ValueError("MIP-Editor expects 2D or 3D FFN activations")
                # Avoid changing a tensor another autograd operation saved.
                value = value.clone()
                if value.ndim == 3:
                    value[:, 0, selected] = 0
                else:
                    value[0, selected] = 0
                return (value, *inputs[1:])

            handles.append(module.register_forward_pre_hook(mask_first_token))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def export_row_edits(model):
    return {
        name: {
            "indices": torch.tensor(module.trainable_cols, dtype=torch.long),
            "weight": module.trainable_linear.weight.detach().cpu().clone(),
        }
        for name, module in model.named_modules()
        if isinstance(module, PartialLinear)
    }


def apply_row_edits(model, edits):
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, edit in edits.items():
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"MIP-Editor row projection not found: {name}")
            indices = edit["indices"].to(module.weight.device, dtype=torch.long)
            weight = edit["weight"]
            if indices.ndim != 1 or weight.shape != (
                indices.numel(),
                module.in_features,
            ):
                raise ValueError(f"MIP-Editor row shape does not match {name}")
            if ((indices < 0) | (indices >= module.out_features)).any():
                raise ValueError(f"MIP-Editor row index outside projection: {name}")
            module.weight[indices] = weight.to(
                module.weight.device, module.weight.dtype
            )
