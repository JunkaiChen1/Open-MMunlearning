"""Text-level scorers used by multimodal benchmark evaluators."""

import unicodedata
from typing import Any, Iterable

from .base import BaseScorer, ScoreResult


def normalize_text(value: Any) -> str:
    """Normalize text for exact or substring comparisons."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


class ExactMatchScorer(BaseScorer):
    """Case-insensitive, whitespace-normalized string equality."""

    name = "exact_match"

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        value = float(normalize_text(prediction) == normalize_text(reference))
        return ScoreResult(self.name, value)


class ContainsScorer(BaseScorer):
    """Check whether a reference phrase occurs in a prediction.

    ``normalization="canonical"`` is the default reusable behavior.  The
    legacy MLLMU protocol used Python's ``str.lower()`` directly, so callers
    that reproduce that protocol can request ``normalization="lower"``.
    """

    name = "contains"

    def __init__(self, normalization: str = "canonical"):
        if normalization not in {"canonical", "lower"}:
            raise ValueError(
                "ContainsScorer normalization must be 'canonical' or 'lower'."
            )
        self.normalization = normalization

    def _normalize(self, value: Any) -> str:
        if self.normalization == "lower":
            return str(value).lower()
        return normalize_text(value)

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        prediction_text = self._normalize(prediction)
        reference_text = self._normalize(reference)
        value = float(reference_text in prediction_text)
        return ScoreResult(self.name, value)


class RougeScorer(BaseScorer):
    """Compute one or more ROUGE variants with a selected scalar value.

    The optional ``rouge_score`` dependency is imported only when this scorer
    is used, keeping the scorer registry importable for lightweight tests.
    ``prediction`` comes first in this API; the underlying package receives
    ``reference`` first as required by its API.
    """

    name = "rouge"
    _VALID_AGGREGATIONS = {"precision", "recall", "fmeasure"}

    def __init__(
        self,
        rouge_types: Iterable[str] = ("rougeL",),
        *,
        use_stemmer: bool = True,
        aggregation: str = "recall",
        primary: str | None = None,
    ):
        self.rouge_types = tuple(rouge_types)
        if not self.rouge_types:
            raise ValueError("RougeScorer requires at least one rouge type.")
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown ROUGE aggregation {aggregation!r}; expected one of "
                f"{sorted(self._VALID_AGGREGATIONS)}."
            )
        if primary is not None and primary not in self.rouge_types:
            raise ValueError(f"Primary ROUGE type {primary!r} is not configured.")
        self.use_stemmer = bool(use_stemmer)
        self.aggregation = aggregation
        self.primary = primary or self.rouge_types[0]
        self._backend_instance = None

    def _backend(self):
        if self._backend_instance is not None:
            return self._backend_instance
        try:
            from rouge_score import rouge_scorer as rouge_backend
        except ImportError as exc:
            raise ImportError(
                "RougeScorer requires the optional 'rouge-score' package."
            ) from exc
        self._backend_instance = rouge_backend.RougeScorer(
            list(self.rouge_types), use_stemmer=self.use_stemmer
        )
        return self._backend_instance

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        scores = self._backend().score(str(reference), str(prediction))
        details = {
            rouge_type: float(getattr(scores[rouge_type], self.aggregation))
            for rouge_type in self.rouge_types
        }
        return ScoreResult(self.name, details[self.primary], details)


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
            raise ImportError("BleuScorer requires the optional 'nltk' package.") from exc
        smoothing = getattr(SmoothingFunction(), self.smoothing_method)
        value = sentence_bleu(
            [str(reference).split()],
            str(prediction).split(),
            smoothing_function=smoothing,
        )
        return ScoreResult(self.name, float(value))
