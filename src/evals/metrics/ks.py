"""KS distribution metric."""

from typing import Any

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import _as_vector, distribution_values


@register_metric
@unlearning_metric("ks")
def ks(
    model,
    *,
    pre_compute,
    primary="pvalue",
    alternative="two-sided",
    method="auto",
    current_key="current",
    reference_key="reference",
    **kwargs,
):
    result = TwoSampleKSScorer(
        primary=primary, alternative=alternative, method=method
    ).score(
        distribution_values(pre_compute[current_key]),
        distribution_values(pre_compute[reference_key]),
    )
    return {"agg_value": result.value, "value_by_index": result.details}


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
