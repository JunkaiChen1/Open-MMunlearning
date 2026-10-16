"""Exact match calculation and configured metric."""

from typing import Any

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import normalize_text, score_predictions


class ExactMatchScorer(BaseScorer):
    """Case-insensitive, whitespace-normalized string equality."""

    name = "exact_match"

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        value = float(normalize_text(prediction) == normalize_text(reference))
        return ScoreResult(self.name, value)


@register_metric
@unlearning_metric("exact_match")
def exact_match(model, *, pre_compute, access_key="generation", **kwargs):
    return score_predictions(pre_compute, ExactMatchScorer(), access_key=access_key)
