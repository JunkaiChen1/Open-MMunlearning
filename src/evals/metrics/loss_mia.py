"""Signed loss membership inference metric."""

import math
from typing import Any

from evals.metrics import register_metric
from evals.utils.inference import evaluate_probability

from .base import Attack, BaseScorer, ScoreResult, unlearning_metric
from .utils import sample_loss, score_inputs


def loss_score(average_loss, *, sign=1.0):
    """Keep the input precision used by each attack or benchmark."""
    return average_loss if sign == 1.0 else sign * average_loss


def mllmu_loss_scores(totals):
    return {
        f"Average Loss MIA ({label})": totals[bucket]["loss_mia"]
        / totals[bucket]["mia_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["mia_n"]
    }


@register_metric
@unlearning_metric("loss_mia", required_inputs=("avg_loss",))
def loss_mia(model, *, pre_compute, sign=1.0, access_key="generation", **kwargs):
    scorer = LossMIAScorer(sign=sign)
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(sample_loss(item)).value
        if sample_loss(item) is not None
        else None,
        access_key=access_key,
    )


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
        return average_loss, float(loss_score(average_loss, sign=self.sign))

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


class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        """Return the average loss for the sample."""
        return loss_score(sample_stats["avg_loss"])
