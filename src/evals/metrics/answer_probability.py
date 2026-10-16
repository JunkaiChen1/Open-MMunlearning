"""Answer probability with original benchmark conventions."""

import math
from typing import Any, Iterable

import numpy as np

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import sample_loss, score_inputs


def normalize_ground_truth_loss(logs, scorer):
    if "avg_gt_loss" not in logs or "average_perturb_loss" not in logs:
        return
    normalized = {}
    for index, avg_gt_loss in logs["avg_gt_loss"].items():
        if index not in logs["average_perturb_loss"]:
            continue
        probability = scorer.score(
            avg_gt_loss, logs["average_perturb_loss"][index]
        ).value
        normalized[index] = (
            float("inf") if probability == 0 else float(-math.log(probability))
        )
    logs["normalized_gt_loss"] = normalized


def mllmu_probability_scores(totals):
    return {
        f"Average Answer Probability ({label})": totals[bucket]["answer_probability"]
        / totals[bucket]["probability_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["n"] and totals[bucket]["probability_n"]
    }


@register_metric
@unlearning_metric("answer_probability", required_inputs=("avg_loss",))
def answer_probability(model, *, pre_compute, access_key="generation", **kwargs):
    scorer = AnswerProbabilityScorer()
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(sample_loss(item)).value
        if sample_loss(item) is not None
        else None,
        access_key=access_key,
    )


@register_metric
@unlearning_metric(
    "contrastive_probability", required_inputs=("avg_loss", "perturbation_losses")
)
def contrastive_probability(model, *, pre_compute, access_key="generation", **kwargs):
    scorer = ContrastiveProbabilityScorer()
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(sample_loss(item), item["perturbation_losses"]).value,
        access_key=access_key,
        required_inputs=("perturbation_losses",),
    )


class AnswerProbabilityScorer(BaseScorer):
    """Convert an average answer-token negative log-likelihood to probability."""

    name = "answer_probability"

    def score(self, average_loss: Any, reference: Any = None, **_: Any) -> ScoreResult:
        average_loss = float(average_loss)
        if math.isnan(average_loss):
            raise ValueError("Average loss must not be NaN.")
        try:
            value = float(math.exp(-average_loss))
        except OverflowError:
            value = float("inf")
        return ScoreResult(
            self.name,
            value,
            {"average_loss": average_loss},
        )


class ContrastiveProbabilityScorer(BaseScorer):
    """Probability of a target answer among target and competing answers.

    Inputs are average negative log-likelihoods. The implementation uses a
    shifted exponential calculation so it remains defined when every direct
    ``exp(-loss)`` would underflow.
    """

    name = "contrastive_probability"

    def score(
        self,
        target_loss: Any,
        competing_losses: Iterable[Any],
        **_: Any,
    ) -> ScoreResult:
        target_loss = float(target_loss)
        losses = [float(value) for value in competing_losses]
        if not losses:
            raise ValueError("At least one competing-answer loss is required.")
        if math.isnan(target_loss) or any(math.isnan(value) for value in losses):
            raise ValueError("Target and competing losses must not contain NaN.")

        log_probabilities = [-target_loss, *(-value for value in losses)]
        shift = max(log_probabilities)
        if shift == float("inf"):
            winners = sum(value == shift for value in log_probabilities)
            value = 1.0 / winners if log_probabilities[0] == shift else 0.0
        elif shift == float("-inf"):
            value = 1.0 / len(log_probabilities)
        else:
            shifted = [math.exp(value - shift) for value in log_probabilities]
            value = shifted[0] / sum(shifted)

        return ScoreResult(
            self.name,
            float(value),
            {
                "target_loss": target_loss,
                "competing_losses": losses,
                "num_competing_answers": len(losses),
            },
        )


def clear_probability(self, task_file, task_result):
    avg_gt_loss = self._dict_values_to_array(task_result["avg_gt_loss"])
    if avg_gt_loss.size == 0:
        return None
    true_probs = np.asarray(
        [self._answer_probability_scorer.score(loss).value for loss in avg_gt_loss],
        dtype=float,
    )
    if "eval_log" in task_file or "average_perturb_loss" not in task_result:
        return float(true_probs.mean())
    false_values = task_result["average_perturb_loss"]
    if isinstance(false_values, dict):
        false_values = list(false_values.values())
    if len(false_values) != true_probs.shape[0]:
        return float(true_probs.mean())
    contrastive_probabilities = []
    for target_loss, value in zip(avg_gt_loss, false_values):
        losses = self._flatten_numeric(value)
        if not losses:
            return float(true_probs.mean())
        contrastive_probabilities.append(
            self._contrastive_probability_scorer.score(target_loss, losses).value
        )
    return float(np.mean(contrastive_probabilities))
