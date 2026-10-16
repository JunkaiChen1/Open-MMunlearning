"""Scorers comparing empirical score distributions."""

from typing import Any

from .base import BaseScorer, ScoreResult


def _as_vector(value: Any, *, name: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Distribution scorers require numpy.") from exc

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


class TwoSampleKSScorer(BaseScorer):
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""

    name = "two_sample_ks"
    _VALID_PRIMARY = {"pvalue", "statistic"}

    def __init__(
        self,
        *,
        primary: str = "pvalue",
        alternative: str = "two-sided",
        method: str = "auto",
    ):
        if primary not in self._VALID_PRIMARY:
            raise ValueError(
                f"Unknown KS primary value {primary!r}; expected 'pvalue' or 'statistic'."
            )
        self.primary = primary
        self.alternative = alternative
        self.method = method

    def score(self, first: Any, second: Any = None, **_: Any) -> ScoreResult:
        if second is None:
            raise ValueError("TwoSampleKSScorer requires two distributions.")
        first_vector = _as_vector(first, name="first distribution")
        second_vector = _as_vector(second, name="second distribution")
        try:
            from scipy.stats import ks_2samp
        except ImportError as exc:
            raise ImportError("TwoSampleKSScorer requires scipy.") from exc

        result = ks_2samp(
            first_vector,
            second_vector,
            alternative=self.alternative,
            method=self.method,
        )
        details = {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "first_size": int(first_vector.size),
            "second_size": int(second_vector.size),
            "alternative": self.alternative,
            "method": self.method,
        }
        return ScoreResult(self.name, details[self.primary], details)


class JensenShannonDistanceScorer(BaseScorer):
    """Jensen-Shannon distance with CLEAR-compatible input alignment.

    With ``alignment='auto'``, equal-length inputs are treated as non-negative
    weight vectors, matching CLEAR's direct SciPy call. Unequal inputs are
    interpreted as samples and converted to smoothed histograms over shared
    bins, matching the evaluator's compatibility fallback.
    """

    name = "jensen_shannon_distance"
    _VALID_ALIGNMENTS = {"auto", "weights", "histogram"}

    def __init__(
        self,
        *,
        alignment: str = "auto",
        bins: int = 50,
        epsilon: float = 1e-12,
        base: float | None = None,
    ):
        if alignment not in self._VALID_ALIGNMENTS:
            raise ValueError(
                f"Unknown JS alignment {alignment!r}; expected one of "
                f"{sorted(self._VALID_ALIGNMENTS)}."
            )
        if int(bins) <= 0:
            raise ValueError("Jensen-Shannon histogram bins must be positive.")
        if float(epsilon) < 0:
            raise ValueError("Jensen-Shannon epsilon must be non-negative.")
        self.alignment = alignment
        self.bins = int(bins)
        self.epsilon = float(epsilon)
        self.base = base

    def _histograms(self, first, second):
        import numpy as np

        low = float(min(np.min(first), np.min(second)))
        high = float(max(np.max(first), np.max(second)))
        if low == high:
            high = low + 1e-8
        edges = np.linspace(low, high, self.bins + 1)
        first_hist, _ = np.histogram(first, bins=edges, density=False)
        second_hist, _ = np.histogram(second, bins=edges, density=False)
        first_hist = first_hist.astype(float) + self.epsilon
        second_hist = second_hist.astype(float) + self.epsilon
        return first_hist / first_hist.sum(), second_hist / second_hist.sum()

    def score(self, first: Any, second: Any = None, **_: Any) -> ScoreResult:
        if second is None:
            raise ValueError("JensenShannonDistanceScorer requires two distributions.")
        first_vector = _as_vector(first, name="first distribution")
        second_vector = _as_vector(second, name="second distribution")
        try:
            from scipy.spatial.distance import jensenshannon
        except ImportError as exc:
            raise ImportError("JensenShannonDistanceScorer requires scipy.") from exc

        alignment = self.alignment
        if alignment == "auto":
            alignment = (
                "weights" if first_vector.shape == second_vector.shape else "histogram"
            )
        if alignment == "weights":
            if first_vector.shape != second_vector.shape:
                raise ValueError(
                    "Weight-vector JS inputs must have the same shape, got "
                    f"{first_vector.shape} and {second_vector.shape}."
                )
            if (first_vector < 0).any() or (second_vector < 0).any():
                raise ValueError("Jensen-Shannon weight vectors must be non-negative.")
            if first_vector.sum() <= 0 or second_vector.sum() <= 0:
                raise ValueError(
                    "Jensen-Shannon weight vectors must have positive sums."
                )
            first_aligned, second_aligned = first_vector, second_vector
        else:
            first_aligned, second_aligned = self._histograms(
                first_vector, second_vector
            )

        value = float(jensenshannon(first_aligned, second_aligned, base=self.base))
        return ScoreResult(
            self.name,
            value,
            {
                "alignment": alignment,
                "first_size": int(first_vector.size),
                "second_size": int(second_vector.size),
                "bins": self.bins if alignment == "histogram" else None,
                "base": self.base,
            },
        )
