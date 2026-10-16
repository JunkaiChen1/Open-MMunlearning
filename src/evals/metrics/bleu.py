"""Bleu calculation and configured metric."""

from typing import Any

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import score_predictions


class BleuScorer(BaseScorer):
    """Sentence BLEU with the smoothing rule used by MLLMU-Bench."""

    name = "bleu"

    def __init__(self, smoothing_method: str = "method1"):
        self.smoothing_method = smoothing_method

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        except ImportError as exc:
            raise ImportError(
                "BleuScorer requires the optional 'nltk' package."
            ) from exc
        smoothing = getattr(SmoothingFunction(), self.smoothing_method)
        value = sentence_bleu(
            [str(reference).split()],
            str(prediction).split(),
            smoothing_function=smoothing,
        )
        return ScoreResult(self.name, float(value))


@register_metric
@unlearning_metric("bleu")
def bleu(
    model, *, pre_compute, smoothing_method="method1", access_key="generation", **kwargs
):
    return score_predictions(
        pre_compute, BleuScorer(smoothing_method), access_key=access_key
    )


def mllmu_bleu_scores(totals):
    return {
        f"Average BLEU ({label})": totals[bucket]["bleu"] / totals[bucket]["n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["n"]
    }
