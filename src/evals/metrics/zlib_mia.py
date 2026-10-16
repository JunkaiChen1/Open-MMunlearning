"""ZLIB normalized membership inference metric."""

import zlib
from typing import Any

from evals.metrics import register_metric
from evals.utils.inference import (
    evaluate_probability,
    extract_target_texts_from_processed_data,
)

from .base import Attack, ScoreResult, unlearning_metric
from .loss_mia import _SignedLossScorer
from .utils import sample_loss, score_inputs


def compressed_text_length(text, *, encoding="utf-8"):
    encoded = text if isinstance(text, bytes) else str(text).encode(encoding)
    return len(zlib.compress(encoded))


def zlib_score(average_loss, text, *, encoding="utf-8"):
    return average_loss / compressed_text_length(text.encode(encoding))


def mllmu_zlib_scores(totals):
    return {
        f"Average ZLIB MIA ({label})": totals[bucket]["zlib_mia"]
        / totals[bucket]["mia_n"]
        for bucket, label in (
            ("image_textual", "Image_Textual"),
            ("pure_text", "Pure_Text"),
        )
        if totals[bucket]["mia_n"]
    }


@register_metric
@unlearning_metric("zlib_mia", required_inputs=("avg_loss",))
def zlib_mia(
    model, *, pre_compute, sign=1.0, encoding="utf-8", access_key="generation", **kwargs
):
    scorer = ZlibMIAScorer(sign=sign, encoding=encoding)
    return score_inputs(
        pre_compute,
        lambda item: scorer.score(
            sample_loss(item), item.get("reference", item.get("ground_truth"))
        ).value
        if sample_loss(item) is not None
        else None,
        access_key=access_key,
    )


class ZlibMIAScorer(_SignedLossScorer):
    """ZLIB-normalized average loss used by the ZLIB MIA attack.

    The normalization denominator is the byte length of the default zlib
    compression of the target text. Strings are encoded as UTF-8, matching the
    upstream attack and FIUBench protocol; byte strings are accepted directly.
    """

    name = "zlib_mia"

    def __init__(self, *, sign: float = 1.0, encoding: str = "utf-8"):
        super().__init__(sign=sign)
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("ZLIB text encoding must be a non-empty string.")
        self.encoding = encoding

    def score(
        self,
        average_loss: Any,
        text: Any = None,
        *,
        reference: Any = None,
        **_: Any,
    ) -> ScoreResult:
        if text is None:
            text = reference
        if text is None:
            raise ValueError("ZlibMIAScorer requires target text.")

        average_loss, signed_loss = self._signed_loss(average_loss)
        if isinstance(text, bytes):
            encoded_text = text
        else:
            encoded_text = str(text).encode(self.encoding)
        compressed_length = compressed_text_length(encoded_text)
        # zlib always returns bytes, but keep the denominator contract explicit.
        if compressed_length <= 0:
            raise ValueError("Compressed target text must have a positive length.")

        return ScoreResult(
            self.name,
            float(signed_loss / compressed_length),
            {
                **self._loss_details(average_loss),
                "compressed_length": int(compressed_length),
                "text_length": int(len(encoded_text)),
                "encoding": self.encoding,
            },
        )


class ZLIBAttack(Attack):
    def setup(self, tokenizer=None, **kwargs):
        """Setup tokenizer."""
        self.tokenizer = tokenizer or self.model.tokenizer

    def compute_batch_values(self, batch):
        """Get loss and text for batch."""
        eval_results = evaluate_probability(self.model, batch)
        texts = extract_target_texts_from_processed_data(self.tokenizer, batch)
        return [{"loss": r["avg_loss"], "text": t} for r, t in zip(eval_results, texts)]

    def compute_score(self, sample_stats):
        """Score using loss normalized by compressed text length."""
        return zlib_score(sample_stats["loss"], sample_stats["text"])
