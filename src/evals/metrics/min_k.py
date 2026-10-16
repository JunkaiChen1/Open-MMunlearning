"""Min-K calculation with separate standard and FIUBench conventions."""

from typing import Any

from evals.metrics import register_metric
from evals.utils.inference import tokenwise_logprobs

from ._min_k_utils import (
    _as_vector,
    _StandardMinKScorer,
    _WeightedMinKScorer,
    standard_min_k_value,
)
from .base import Attack, ScoreResult, unlearning_metric
from .utils import score_inputs


def attack_min_k_score(token_log_probs, *, ratio=0.2):
    values = token_log_probs.float().cpu().numpy()
    return standard_min_k_value(values, ratio)


def mllmu_min_k_scores(totals):
    return {
        f"Average Min-K 20% MIA ({label})": totals[bucket]["min_k_20_mia"]
        / totals[bucket]["min_k_mia_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["min_k_mia_n"]
    }


@register_metric
@unlearning_metric("min_k", required_inputs=("token_log_probs",))
def min_k(
    model,
    *,
    pre_compute,
    convention="standard",
    ratio=0.2,
    access_key="generation",
    **kwargs,
):
    if convention not in {"standard", "fiubench"}:
        raise ValueError("Min-K convention must be 'standard' or 'fiubench'.")
    scorer = (
        StandardMinKScorer(ratio=ratio) if convention == "standard" else MinKScorer()
    )
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(item["token_log_probs"]).value,
        access_key=access_key,
        required_inputs=("token_log_probs",),
    )


class StandardMinKScorer(_StandardMinKScorer):
    """Conventional Min-K score, using the lowest 20% of token log-probs."""

    name = "standard_min_k"

    def score(
        self,
        token_log_probs: Any,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        values = _as_vector(
            token_log_probs,
            name="token_log_probs",
            allow_empty=True,
        )
        return self._result(values)


class MinKScorer(_WeightedMinKScorer):
    """Weighted lowest-token log-probability score used by FIUBench."""

    name = "min_k"

    def score(
        self,
        token_log_probs: Any,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        values = _as_vector(token_log_probs, name="token_log_probs")
        return self._result(values)


class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get token-wise log probabilities for the batch."""
        return tokenwise_logprobs(self.model, batch, grad=False)

    def compute_score(self, sample_stats):
        """Score single sample using min-k negative log probs scores attack."""
        return attack_min_k_score(sample_stats, ratio=self.k)
