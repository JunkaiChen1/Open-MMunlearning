"""Contains calculation and configured metric."""

from typing import Any

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import normalize_text, score_predictions


class ContainsScorer(BaseScorer):
    """Check whether a reference phrase occurs in a prediction.

    ``normalization="canonical"`` is the default reusable behavior.  The
    legacy MLLMU protocol used Python's ``str.lower()`` directly, so callers
    that reproduce that protocol can request ``normalization="lower"``.
    """

    name = "contains"

    def __init__(self, normalization: str = "canonical"):
        if normalization not in {"canonical", "lower"}:
            raise ValueError(
                "ContainsScorer normalization must be 'canonical' or 'lower'."
            )
        self.normalization = normalization

    def _normalize(self, value: Any) -> str:
        if self.normalization == "lower":
            return str(value).lower()
        return normalize_text(value)

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        prediction_text = self._normalize(prediction)
        reference_text = self._normalize(reference)
        value = float(reference_text in prediction_text)
        return ScoreResult(self.name, value)


@register_metric
@unlearning_metric("contains")
def contains(
    model, *, pre_compute, normalization="canonical", access_key="generation", **kwargs
):
    return score_predictions(
        pre_compute, ContainsScorer(normalization=normalization), access_key=access_key
    )
