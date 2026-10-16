"""Loss-derived scorers shared by multimodal benchmarks."""

import math
from typing import Any, Iterable

from .base import BaseScorer, ScoreResult


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
