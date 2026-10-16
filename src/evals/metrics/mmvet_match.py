"""MM-Vet AND/OR fragment matching from the original evaluator."""

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import normalize_benchmark_text, score_predictions


def mmvet_references(answer):
    """Expand MM-Vet's ``<OR>``/``<AND>`` answer convention."""
    return [part.strip() for part in str(answer).split("<OR>") if part.strip()]


def mmvet_match(prediction, answer):
    prediction_text = normalize_benchmark_text(prediction)
    for alternative in mmvet_references(answer):
        required = [
            normalize_benchmark_text(part)
            for part in alternative.split("<AND>")
            if normalize_benchmark_text(part)
        ]
        if required and all(part in prediction_text for part in required):
            return 1.0
    return 0.0


class MMVetMatchScorer(BaseScorer):
    name = "mmvet_match"

    def score(self, prediction, reference=None, **kwargs):
        return ScoreResult(self.name, mmvet_match(prediction, reference))


@register_metric
@unlearning_metric("mmvet_match")
def mmvet(model, *, pre_compute, access_key="generation", **kwargs):
    return score_predictions(pre_compute, MMVetMatchScorer(), access_key=access_key)
