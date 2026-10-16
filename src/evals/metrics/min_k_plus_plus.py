"""Min-K calculation with separate standard and FIUBench conventions."""

import math
from typing import Any

from evals.metrics import register_metric
from evals.utils.inference import tokenwise_logprobs, tokenwise_vocab_logprobs

from ._min_k_utils import (
    _as_vector,
    _StandardMinKScorer,
    _WeightedMinKScorer,
    standard_min_k_value,
)
from .base import ScoreResult, unlearning_metric
from .min_k import MinKProbAttack
from .utils import score_inputs


def attack_min_k_plus_plus_score(vocab_log_probs, token_log_probs, *, ratio=0.2):
    """Original attack arithmetic, including tensor dtype and float32 casts."""
    import torch

    if len(token_log_probs) == 0:
        return 0
    mu = (torch.exp(vocab_log_probs) * vocab_log_probs).sum(-1)
    sigma = (torch.exp(vocab_log_probs) * torch.square(vocab_log_probs)).sum(
        -1
    ) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-6)
    scores = (
        token_log_probs.float().cpu().numpy() - mu.float().cpu().numpy()
    ) / torch.sqrt(sigma).cpu().numpy()
    return standard_min_k_value(scores, ratio, python_sort=True)


def mllmu_min_k_plus_plus_scores(totals):
    return {
        f"Average Min-K++ 20% MIA ({label})": totals[bucket]["min_k_plus_plus_20_mia"]
        / totals[bucket]["min_k_mia_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["min_k_mia_n"]
    }


@register_metric
@unlearning_metric(
    "min_k_plus_plus", required_inputs=("token_log_probs", "mu", "sigma")
)
def min_k_plus_plus(
    model,
    *,
    pre_compute,
    convention="standard",
    ratio=0.2,
    variance_floor=None,
    access_key="generation",
    **kwargs,
):
    if convention not in {"standard", "fiubench"}:
        raise ValueError("Min-K++ convention must be 'standard' or 'fiubench'.")
    options = {} if variance_floor is None else {"variance_floor": variance_floor}
    scorer = (
        StandardMinKPlusPlusScorer(ratio=ratio, **options)
        if convention == "standard"
        else MinKPlusPlusScorer(**options)
    )
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(
            item["token_log_probs"], item["mu"], item["sigma"]
        ).value,
        access_key=access_key,
        required_inputs=("token_log_probs", "mu", "sigma"),
    )


class StandardMinKPlusPlusScorer(_StandardMinKScorer):
    """Conventional Min-K++ 20% score with vocabulary normalization."""

    name = "standard_min_k_plus_plus"

    def __init__(self, *, ratio: float = 0.2, variance_floor: float = 1e-6):
        super().__init__(ratio=ratio)
        try:
            variance_floor = float(variance_floor)
        except (TypeError, ValueError) as exc:
            raise ValueError("Min-K++ variance floor must be numeric.") from exc
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("Min-K++ variance floor must be positive.")
        self.variance_floor = variance_floor

    def score(
        self,
        token_log_probs: Any,
        mu: Any,
        sigma: Any,
        **_: Any,
    ) -> ScoreResult:
        import numpy as np

        token_values = _as_vector(
            token_log_probs,
            name="token_log_probs",
            allow_empty=True,
        )
        means = _as_vector(mu, name="mu", allow_empty=True)
        variances = _as_vector(sigma, name="sigma", allow_empty=True)
        if token_values.shape != means.shape or token_values.shape != variances.shape:
            raise ValueError(
                "Min-K++ token_log_probs, mu, and sigma must have the same "
                f"shape, got {token_values.shape}, {means.shape}, and "
                f"{variances.shape}."
            )
        standardized = (token_values - means) / np.sqrt(
            np.maximum(variances, self.variance_floor)
        )
        return self._result(
            standardized,
            extra_details={"variance_floor": self.variance_floor},
        )


class MinKPlusPlusAttack(MinKProbAttack):
    def compute_batch_values(self, batch):
        """Get both token-wise and vocab-wise log probabilities for the batch."""
        vocab_log_probs = tokenwise_vocab_logprobs(self.model, batch, grad=False)
        token_log_probs = tokenwise_logprobs(self.model, batch, grad=False)
        return [
            {"vocab_log_probs": vlp, "token_log_probs": tlp}
            for vlp, tlp in zip(vocab_log_probs, token_log_probs)
        ]

    def compute_score(self, sample_stats):
        """Score using min-k negative log probs scores with vocab-wise normalization."""
        return attack_min_k_plus_plus_score(
            sample_stats["vocab_log_probs"],
            sample_stats["token_log_probs"],
            ratio=self.k,
        )


class MinKPlusPlusScorer(_WeightedMinKScorer):
    """Min-K++ after per-token vocabulary-distribution normalization.

    ``mu`` is the expected log-probability and ``sigma`` is its variance, using
    the field names emitted by FIUBench's token-statistics implementation.
    """

    name = "min_k_plus_plus"

    def __init__(self, *args: Any, variance_floor: float = 1e-12, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if float(variance_floor) <= 0:
            raise ValueError("Min-K++ variance floor must be positive.")
        self.variance_floor = float(variance_floor)

    def score(
        self,
        token_log_probs: Any,
        mu: Any,
        sigma: Any,
        **_: Any,
    ) -> ScoreResult:
        import numpy as np

        token_values = _as_vector(token_log_probs, name="token_log_probs")
        means = _as_vector(mu, name="mu")
        variances = _as_vector(sigma, name="sigma")
        if token_values.shape != means.shape or token_values.shape != variances.shape:
            raise ValueError(
                "Min-K++ token_log_probs, mu, and sigma must have the same "
                f"shape, got {token_values.shape}, {means.shape}, and "
                f"{variances.shape}."
            )
        standardized = (token_values - means) / np.sqrt(
            np.maximum(variances, self.variance_floor)
        )
        return self._result(
            standardized,
            extra_details={"variance_floor": self.variance_floor},
        )
