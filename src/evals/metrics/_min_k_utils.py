"""Shared input validation, token statistics and Min-K weighting."""

import math
from collections.abc import Iterable
from typing import Any

from .base import BaseScorer, ScoreResult


def standard_min_k_value(values, ratio, *, python_sort=False):
    """Preserve the caller's dtype and the legacy Min-K++ sorting path."""
    import numpy as np

    if len(values) == 0:
        return 0
    selected_count = max(1, int(len(values) * ratio))
    ordered = sorted(values) if python_sort else np.sort(values)
    return -np.mean(ordered[:selected_count])


def _as_vector(value: Any, *, name: str, allow_empty: bool = False):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Min-K scorers require numpy.") from exc

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}.")
    if vector.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def token_log_probability_statistics(
    logits: Any,
    labels: Any,
    *,
    ignore_index: int = -100,
    exclude_last_label: bool = True,
) -> dict[str, Any]:
    """Extract target log-probabilities and Min-K++ moments for one sample.

    ``logits`` and ``labels`` must already be aligned by the evaluator. The
    final labelled token is excluded by default because the repository's
    standard MIA utilities treat it as EOS and omit its prediction.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Token probability statistics require torch.") from exc

    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch tensors.")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("Token statistics require a single-sample logits batch.")
        logits = logits[0]
    if labels.ndim == 2:
        if labels.shape[0] != 1:
            raise ValueError("Token statistics require a single-sample labels batch.")
        labels = labels[0]
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            "logits and labels must have shapes [sequence, vocabulary] and "
            "[sequence]."
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Token statistics require aligned sequence lengths, got "
            f"{logits.shape[0]} logits and {labels.shape[0]} labels."
        )

    shifted_logits = logits[:-1]
    shifted_labels = labels[1:]
    valid = shifted_labels != int(ignore_index)
    if exclude_last_label:
        valid_indices = valid.nonzero(as_tuple=True)[0]
        if valid_indices.numel():
            valid[valid_indices[-1]] = False

    if not valid.any():
        empty = torch.empty(0, dtype=torch.float32)
        return {"token_log_probs": empty, "mu": empty.clone(), "sigma": empty.clone()}

    selected_logits = shifted_logits[valid]
    selected_labels = shifted_labels[valid]
    log_probs = F.log_softmax(selected_logits, dim=-1)
    probabilities = torch.exp(log_probs)
    token_log_probs = log_probs.gather(
        dim=-1, index=selected_labels.unsqueeze(-1)
    ).squeeze(-1)
    mu = (probabilities * log_probs).sum(-1)
    sigma = (probabilities * torch.square(log_probs)).sum(-1) - torch.square(mu)
    return {
        "token_log_probs": token_log_probs.detach().float().cpu(),
        "mu": mu.detach().float().cpu(),
        "sigma": sigma.detach().float().cpu(),
    }


class _StandardMinKScorer(BaseScorer):
    """Shared aggregation for the conventional fixed-ratio Min-K attacks."""

    def __init__(self, *, ratio: float = 0.2):
        try:
            ratio = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError("Standard Min-K ratio must be numeric.") from exc
        if not math.isfinite(ratio) or ratio <= 0 or ratio > 1:
            raise ValueError("Standard Min-K ratio must be in the interval (0, 1].")
        self.ratio = ratio

    def _result(self, values, *, extra_details=None):
        token_count = int(len(values))
        if token_count == 0:
            selected_count = 0
            value = 0.0
        else:
            selected_count = max(1, int(token_count * self.ratio))
            value = float(standard_min_k_value(values, self.ratio))
        details = {
            "token_count": token_count,
            "ratio": self.ratio,
            "selected_count": selected_count,
            "orientation": "negative_mean",
        }
        if extra_details:
            details.update(extra_details)
        return ScoreResult(self.name, value, details)


class _WeightedMinKScorer(BaseScorer):
    def __init__(
        self,
        ratios: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
        *,
        weights: Iterable[float] = (0.3, 0.3, 0.2, 0.1, 0.1),
        exponentiate: bool = True,
        renormalize_valid_weights: bool = False,
    ):
        self.ratios = tuple(float(value) for value in ratios)
        self.weights = tuple(float(value) for value in weights)
        if not self.ratios:
            raise ValueError("At least one Min-K ratio is required.")
        if len(self.ratios) != len(self.weights):
            raise ValueError("Min-K ratios and weights must have the same length.")
        if any(value <= 0 or value > 1 for value in self.ratios):
            raise ValueError("Min-K ratios must be in the interval (0, 1].")
        if any(value < 0 or not math.isfinite(value) for value in self.weights):
            raise ValueError("Min-K weights must be finite and non-negative.")
        if sum(self.weights) <= 0:
            raise ValueError("At least one Min-K weight must be positive.")
        self.exponentiate = bool(exponentiate)
        self.renormalize_valid_weights = bool(renormalize_valid_weights)

    def _aggregate(self, values):
        import numpy as np

        sorted_values = np.sort(values)
        components: list[float | None] = []
        weighted_sum = 0.0
        valid_weight = 0.0
        for ratio, weight in zip(self.ratios, self.weights):
            k_length = int(len(sorted_values) * ratio)
            if k_length <= 0:
                components.append(None)
                continue
            component = float(np.mean(sorted_values[:k_length]))
            if self.exponentiate:
                component = float(math.exp(component))
            components.append(component)
            weighted_sum += component * weight
            valid_weight += weight

        if not components or valid_weight == 0:
            value = 0.0
        elif self.renormalize_valid_weights:
            value = weighted_sum / valid_weight
        else:
            # FIUBench intentionally does not renormalize weights when short
            # answers make one or more ratios empty.
            value = weighted_sum
        return float(value), components, valid_weight

    def _result(self, values, *, extra_details=None):
        value, components, valid_weight = self._aggregate(values)
        details = {
            "token_count": int(len(values)),
            "ratios": list(self.ratios),
            "weights": list(self.weights),
            "scores_by_ratio": {
                str(ratio): component
                for ratio, component in zip(self.ratios, components)
            },
            "valid_weight": float(valid_weight),
            "exponentiate": self.exponentiate,
            "renormalize_valid_weights": self.renormalize_valid_weights,
        }
        if extra_details:
            details.update(extra_details)
        return ScoreResult(self.name, value, details)
