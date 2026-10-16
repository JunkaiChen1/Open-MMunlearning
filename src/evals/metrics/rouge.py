"""Rouge calculation and configured metric."""

from typing import Any, Iterable

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import score_predictions


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


@register_metric
@unlearning_metric("rouge")
def rouge(
    model,
    *,
    pre_compute,
    rouge_types=("rougeL",),
    aggregation="recall",
    use_stemmer=True,
    primary=None,
    access_key="generation",
    **kwargs,
):
    scorer = RougeScorer(
        rouge_types, aggregation=aggregation, use_stemmer=use_stemmer, primary=primary
    )
    return score_predictions(pre_compute, scorer, access_key=access_key)


def clear_rouge_recall(self, generated_outputs, ground_truths, indices):
    rouge1_recall = {}
    rouge_l_recall = {}
    for generated, ground_truth, index in zip(
        generated_outputs, ground_truths, indices
    ):
        scores = self._rouge_scorer.score(str(generated), str(ground_truth)).details
        rouge1_recall[str(index)] = float(scores["rouge1"])
        rouge_l_recall[str(index)] = float(scores["rougeL"])
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}


def fiu_rouge_recall(self, generated_outputs, ground_truths, indices):
    rouge1_recall = {}
    rouge_l_recall = {}
    for generated, ground_truth, index in zip(
        generated_outputs, ground_truths, indices
    ):
        rouge_text = str(generated)
        if "." in rouge_text:
            rouge_text = rouge_text[: rouge_text.find(".")]
        scores = self._rouge_scorer.score(rouge_text, str(ground_truth)).details
        rouge1_recall[str(index)] = scores["rouge1"]
        rouge_l_recall[str(index)] = scores["rougeL"]
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}


def safeeraser_rouge_summary(values, answer_order):
    return {
        "average_rougeL": sum(values) / len(values) if values else None,
        "rougeL_count": len(values),
        "answer_order": answer_order,
    }


def mllmu_rouge_scores(totals):
    result = {}
    for bucket, label in (
        ("image_textual", "Image_Textual"),
        ("pure_text", "Pure_Text"),
    ):
        n = totals[bucket]["n"]
        if n:
            for field, metric in (
                ("rouge1", "ROUGE-1"),
                ("rouge2", "ROUGE-2"),
                ("rougeL", "ROUGE-L"),
            ):
                result[f"Average {metric} ({label})"] = totals[bucket][field] / n
    return result


def clear_saved_rouge_recall(self, generated_text):
    rouge1_recall = {}
    rouge_l_recall = {}
    for index, pair in generated_text.items():
        if len(pair) < 3:
            continue
        _, generated, ground_truth, *_ = pair
        scores = self._rouge_scorer.score(str(generated), str(ground_truth)).details
        rouge1_recall[str(index)] = float(scores["rouge1"])
        rouge_l_recall[str(index)] = float(scores["rougeL"])
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}
