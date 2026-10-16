"""Model-agnostic scoring primitives for membership inference."""

import math
import zlib
from collections.abc import Iterable
from typing import Any

from .base import BaseScorer, ScoreResult


class _SignedLossScorer(BaseScorer):
    """Common validation and orientation handling for loss-based MIA scores.

    The canonical LOSS and ZLIB attacks return a positive average loss. Some
    benchmark evaluators historically negate that value before aggregation,
    so the shared scorers expose the orientation explicitly through ``sign``.
    """

    def __init__(self, *, sign: float = 1.0):
        try:
            sign = float(sign)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MIA score sign must be a finite non-zero number."
            ) from exc
        if not math.isfinite(sign) or sign == 0.0:
            raise ValueError("MIA score sign must be a finite non-zero number.")
        self.sign = sign

    def _signed_loss(self, average_loss: Any) -> tuple[float, float]:
        try:
            average_loss = float(average_loss)
        except (TypeError, ValueError) as exc:
            raise ValueError("Average loss must be numeric.") from exc
        if math.isnan(average_loss):
            raise ValueError("Average loss must not be NaN.")
        return average_loss, float(self.sign * average_loss)

    def _loss_details(self, average_loss: float) -> dict[str, float]:
        return {
            "average_loss": average_loss,
            "sign": self.sign,
        }


class LossMIAScorer(_SignedLossScorer):
    """Average token negative log-likelihood used by the LOSS attack.

    ``sign=1`` reproduces the original single-modal MIA implementation. Set
    ``sign=-1`` for evaluators, such as FIUBench, whose historical log values
    use the opposite orientation.
    """

    name = "loss_mia"

    def score(self, average_loss: Any, reference: Any = None, **_: Any) -> ScoreResult:
        average_loss, value = self._signed_loss(average_loss)
        return ScoreResult(self.name, value, self._loss_details(average_loss))


class ZlibMIAScorer(_SignedLossScorer):
    """ZLIB-normalized average loss used by the ZLIB MIA attack.

    The normalization denominator is the byte length of the default zlib
    compression of the target text. Strings are encoded as UTF-8, matching the
    upstream attack and FIUBench protocol; byte strings are accepted directly.
    """

    name = "zlib_mia"

    def __init__(self, *, sign: float = 1.0, encoding: str = "utf-8"):
        super().__init__(sign=sign)
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("ZLIB text encoding must be a non-empty string.")
        self.encoding = encoding

    def score(
        self,
        average_loss: Any,
        text: Any = None,
        *,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        if text is None:
            text = reference
        if text is None:
            raise ValueError("ZlibMIAScorer requires target text.")

        average_loss, signed_loss = self._signed_loss(average_loss)
        if isinstance(text, bytes):
            encoded_text = text
        else:
            encoded_text = str(text).encode(self.encoding)
        compressed_length = len(zlib.compress(encoded_text))
        # zlib always returns bytes, but keep the denominator contract explicit.
        if compressed_length <= 0:
            raise ValueError("Compressed target text must have a positive length.")

        return ScoreResult(
            self.name,
            float(signed_loss / compressed_length),
            {
                **self._loss_details(average_loss),
                "compressed_length": int(compressed_length),
                "text_length": int(len(encoded_text)),
                "encoding": self.encoding,
            },
        )


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
        import numpy as np

        token_count = int(len(values))
        if token_count == 0:
            selected_count = 0
            value = 0.0
        else:
            selected_count = max(1, int(token_count * self.ratio))
            value = float(-np.mean(np.sort(values)[:selected_count]))
        details = {
            "token_count": token_count,
            "ratio": self.ratio,
            "selected_count": selected_count,
            "orientation": "negative_mean",
        }
        if extra_details:
            details.update(extra_details)
        return ScoreResult(self.name, value, details)


class StandardMinKScorer(_StandardMinKScorer):
    """Conventional Min-K score, using the lowest 20% of token log-probs."""

    name = "standard_min_k"

    def score(
        self,
        token_log_probs: Any,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        values = _as_vector(
            token_log_probs,
            name="token_log_probs",
            allow_empty=True,
        )
        return self._result(values)


class StandardMinKPlusPlusScorer(_StandardMinKScorer):
    """Conventional Min-K++ 20% score with vocabulary normalization."""

    name = "standard_min_k_plus_plus"

    def __init__(self, *, ratio: float = 0.2, variance_floor: float = 1e-6):
        super().__init__(ratio=ratio)
        try:
            variance_floor = float(variance_floor)
        except (TypeError, ValueError) as exc:
            raise ValueError("Min-K++ variance floor must be numeric.") from exc
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("Min-K++ variance floor must be positive.")
        self.variance_floor = variance_floor

    def score(
        self,
        token_log_probs: Any,
        mu: Any,
        sigma: Any,
        **_: Any,
    ) -> ScoreResult:
        import numpy as np

        token_values = _as_vector(
            token_log_probs,
            name="token_log_probs",
            allow_empty=True,
        )
        means = _as_vector(mu, name="mu", allow_empty=True)
        variances = _as_vector(sigma, name="sigma", allow_empty=True)
        if token_values.shape != means.shape or token_values.shape != variances.shape:
            raise ValueError(
                "Min-K++ token_log_probs, mu, and sigma must have the same "
                f"shape, got {token_values.shape}, {means.shape}, and "
                f"{variances.shape}."
            )
        standardized = (token_values - means) / np.sqrt(
            np.maximum(variances, self.variance_floor)
        )
        return self._result(
            standardized,
            extra_details={"variance_floor": self.variance_floor},
        )


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


class MinKScorer(_WeightedMinKScorer):
    """Weighted lowest-token log-probability score used by FIUBench."""

    name = "min_k"

    def score(
        self,
        token_log_probs: Any,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        values = _as_vector(token_log_probs, name="token_log_probs")
        return self._result(values)


class MinKPlusPlusScorer(_WeightedMinKScorer):
    """Min-K++ after per-token vocabulary-distribution normalization.

    ``mu`` is the expected log-probability and ``sigma`` is its variance, using
    the field names emitted by FIUBench's token-statistics implementation.
    """

    name = "min_k_plus_plus"

    def __init__(self, *args: Any, variance_floor: float = 1e-12, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if float(variance_floor) <= 0:
            raise ValueError("Min-K++ variance floor must be positive.")
        self.variance_floor = float(variance_floor)

    def score(
        self,
        token_log_probs: Any,
        mu: Any,
        sigma: Any,
        **_: Any,
    ) -> ScoreResult:
        import numpy as np

        token_values = _as_vector(token_log_probs, name="token_log_probs")
        means = _as_vector(mu, name="mu")
        variances = _as_vector(sigma, name="sigma")
        if token_values.shape != means.shape or token_values.shape != variances.shape:
            raise ValueError(
                "Min-K++ token_log_probs, mu, and sigma must have the same "
                f"shape, got {token_values.shape}, {means.shape}, and "
                f"{variances.shape}."
            )
        standardized = (token_values - means) / np.sqrt(
            np.maximum(variances, self.variance_floor)
        )
        return self._result(
            standardized,
            extra_details={"variance_floor": self.variance_floor},
        )
