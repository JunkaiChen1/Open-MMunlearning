"""Reference-based membership inference from target and reference losses."""

from evals.metrics import register_metric
from evals.utils.inference import evaluate_probability

from .base import Attack, unlearning_metric
from .utils import score_inputs


def reference_loss_score(target_loss, reference_loss):
    return target_loss - reference_loss


@register_metric
@unlearning_metric("reference_loss", required_inputs=("target_loss", "ref_loss"))
def reference_loss(model, *, pre_compute, access_key="generation", **kwargs):
    return score_inputs(
        pre_compute,
        lambda item: reference_loss_score(item["target_loss"], item["ref_loss"]),
        access_key=access_key,
        required_inputs=("target_loss", "ref_loss"),
    )


class ReferenceAttack(Attack):
    def setup(self, reference_model, **kwargs):
        """Setup reference model."""
        self.reference_model = reference_model

    def compute_batch_values(self, batch):
        """Compute loss scores for both target and reference models."""
        ref_results = evaluate_probability(self.reference_model, batch)
        target_results = evaluate_probability(self.model, batch)
        return [
            {"target_loss": t["avg_loss"], "ref_loss": r["avg_loss"]}
            for t, r in zip(target_results, ref_results)
        ]

    def compute_score(self, sample_stats):
        """Score using difference between target and reference model losses."""
        return reference_loss_score(
            sample_stats["target_loss"], sample_stats["ref_loss"]
        )
