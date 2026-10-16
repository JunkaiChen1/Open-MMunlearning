"""Truth ratio with original benchmark conventions."""

import math
from typing import Any, Iterable

import numpy as np

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import score_inputs


def mllmu_positive_loss(answer_loss, paraphrase_loss):
    return (answer_loss + paraphrase_loss) / 2.0


def mllmu_truth_ratio_scores(totals):
    return {
        f"Average Truth Ratio ({label})": totals[bucket]["truth_ratio"]
        / totals[bucket]["truth_ratio_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["n"] and totals[bucket]["truth_ratio_n"]
    }


@register_metric
@unlearning_metric("truth_ratio", required_inputs=("base_loss", "perturbation_losses"))
def truth_ratio(model, *, pre_compute, access_key="generation", **kwargs):
    scorer = PerturbationTruthRatioScorer()
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(item["base_loss"], item["perturbation_losses"]).value,
        access_key=access_key,
        required_inputs=("base_loss", "perturbation_losses"),
    )


class PerturbationTruthRatioScorer(BaseScorer):
    """Compare a base-answer loss with the mean perturbed-answer loss.

    This is the loss-space form used by the FIU/CLEAR implementations:
    ``exp(base_loss - mean(perturbation_losses))``.
    """

    name = "truth_ratio"

    def score(
        self,
        base_loss: Any,
        perturbation_losses: Iterable[Any],
        **_: Any,
    ) -> ScoreResult:
        losses = [float(value) for value in perturbation_losses]
        if not losses:
            raise ValueError("At least one perturbation loss is required.")
        base_loss = float(base_loss)
        mean_perturbation_loss = sum(losses) / len(losses)
        try:
            value = float(math.exp(base_loss - mean_perturbation_loss))
        except OverflowError:
            value = float("inf")
        return ScoreResult(
            self.name,
            value,
            {
                "base_loss": base_loss,
                "mean_perturbation_loss": mean_perturbation_loss,
                "num_perturbations": len(losses),
            },
        )


class TruthRatioTransformScorer(BaseScorer):
    """Apply the forget/retain transformations used by FIU and CLEAR.

    Their aggregate code defines the protocol ratio as
    ``exp(mean(perturbation_loss) - base_loss)``. The existing
    :class:`PerturbationTruthRatioScorer` emits its reciprocal, so the input
    convention is explicit rather than silently assuming one orientation.
    """

    name = "truth_ratio_transform"
    _VALID_MODES = {"forget", "retain"}
    _VALID_CONVENTIONS = {"perturb_over_base", "base_over_perturb"}

    def __init__(
        self,
        mode: str | None = None,
        *,
        ratio_convention: str = "perturb_over_base",
    ):
        if mode is not None and mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown truth-ratio mode {mode!r}; expected 'forget' or 'retain'."
            )
        if ratio_convention not in self._VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown ratio convention {ratio_convention!r}; expected one of "
                f"{sorted(self._VALID_CONVENTIONS)}."
            )
        self.mode = mode
        self.ratio_convention = ratio_convention

    @staticmethod
    def _reciprocal(value: float) -> float:
        if value == 0.0:
            return float("inf")
        if value == float("inf"):
            return 0.0
        return 1.0 / value

    def score(
        self,
        truth_ratio: Any,
        reference: Any = None,
        *,
        mode: str | None = None,
        ratio_convention: str | None = None,
        **_: Any,
    ) -> ScoreResult:
        selected_mode = mode or self.mode
        if selected_mode not in self._VALID_MODES:
            raise ValueError("Truth-ratio mode must be 'forget' or 'retain'.")
        convention = ratio_convention or self.ratio_convention
        if convention not in self._VALID_CONVENTIONS:
            raise ValueError(
                f"Unknown ratio convention {convention!r}; expected one of "
                f"{sorted(self._VALID_CONVENTIONS)}."
            )

        raw_ratio = float(truth_ratio)
        if math.isnan(raw_ratio) or raw_ratio < 0:
            raise ValueError("Truth ratio must be non-negative and not NaN.")
        protocol_ratio = (
            raw_ratio
            if convention == "perturb_over_base"
            else self._reciprocal(raw_ratio)
        )
        reciprocal = self._reciprocal(protocol_ratio)
        if selected_mode == "forget":
            value = min(protocol_ratio, reciprocal)
        else:
            value = max(0.0, 1.0 - reciprocal)

        return ScoreResult(
            self.name,
            float(value),
            {
                "mode": selected_mode,
                "raw_truth_ratio": raw_ratio,
                "protocol_truth_ratio": protocol_ratio,
                "ratio_convention": convention,
            },
        )


def clear_truth_distribution(self, task_result):
    """Match CLEAR/mm/calculate_metrics.py's global perturbation-loss mean."""
    paraphrase = self._dict_values_to_array(task_result.get("avg_paraphrased_loss", {}))
    perturbed = self._dict_values_to_array(task_result.get("average_perturb_loss", {}))
    if paraphrase.size == 0 or perturbed.size == 0:
        return np.array([])
    # The official summary uses one scalar mean across all samples and
    # perturbations for both Model Utility and the KS/JS comparison.
    return np.exp(perturbed.mean() - paraphrase)
