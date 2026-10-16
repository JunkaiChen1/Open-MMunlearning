"""Answer recovery scores used by visual and prompt attacks."""

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import score_predictions


def target_hit(answer, target, *, normalization="whitespace"):
    if normalization == "sua":
        return str(target).lower() in answer.lower()
    if normalization != "whitespace":
        raise ValueError("Target normalization must be 'whitespace' or 'sua'.")
    answer = " ".join(str(answer).lower().split())
    target = " ".join(str(target).lower().split())
    return bool(target and target in answer)


def recovery_percentage(hits, total):
    return 100.0 * hits / total


def variant_recovery_rates(variant_hits, total):
    return {
        variant: {"target_recovery_rate": recovery_percentage(hits, total)}
        for variant, hits in variant_hits.items()
    }


def task_recovery_rates(task_hits, task_counts, variants):
    return {
        task: {
            variant: {
                "target_recovery_rate": recovery_percentage(
                    task_hits[task][variant], task_counts[task]
                )
            }
            for variant in variants
        }
        for task in task_counts
    }


class TargetRecoveryScorer(BaseScorer):
    name = "target_recovery"

    def __init__(self, normalization="whitespace"):
        self.normalization = normalization

    def score(self, prediction, reference=None, **kwargs):
        return ScoreResult(
            self.name,
            float(target_hit(prediction, reference, normalization=self.normalization)),
        )


@register_metric
@unlearning_metric("target_recovery")
def target_recovery(
    model, *, pre_compute, normalization="whitespace", access_key="generation", **kwargs
):
    result = score_predictions(
        pre_compute, TargetRecoveryScorer(normalization), access_key=access_key
    )
    if result["agg_value"] is not None:
        values = result["value_by_index"].values()
        result["agg_value"] = recovery_percentage(
            sum(item["score"] for item in values), len(result["value_by_index"])
        )
    return result
