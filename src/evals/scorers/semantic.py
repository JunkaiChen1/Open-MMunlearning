"""Embedding-based scorers with no dependency on a particular encoder."""

from typing import Any

from .base import BaseScorer, ScoreResult


def _as_numpy(value: Any):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("SemanticDissimilarityScorer requires numpy.") from exc
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype="float32")


class SemanticDissimilarityScorer(BaseScorer):
    """Pairwise cosine dissimilarity on CoVUBench's 0-100 scale."""

    name = "semantic_dissimilarity"

    def _pairwise(self, prediction_embeddings: Any, answer_embeddings: Any):
        import numpy as np

        predictions = _as_numpy(prediction_embeddings)
        answers = _as_numpy(answer_embeddings)
        if predictions.ndim == 1:
            predictions = predictions[None, :]
        if answers.ndim == 1:
            answers = answers[None, :]
        if predictions.shape != answers.shape:
            raise ValueError(
                "Prediction and answer embeddings must have the same shape, got "
                f"{tuple(predictions.shape)} and {tuple(answers.shape)}."
            )
        if predictions.ndim != 2:
            raise ValueError("Embeddings must be one- or two-dimensional.")
        numerator = np.sum(predictions * answers, axis=-1)
        denominator = np.linalg.norm(predictions, axis=-1) * np.linalg.norm(
            answers, axis=-1
        )
        similarities = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype="float32"),
            where=denominator > 0,
        )
        similarities = np.clip(similarities, -1.0, 1.0)
        dissimilarities = np.clip((1.0 - similarities) * 100.0, 0.0, 100.0)
        return similarities.tolist(), dissimilarities.tolist()

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        if reference is None:
            return ScoreResult(self.name, None)
        similarities, dissimilarities = self._pairwise(prediction, reference)
        if len(dissimilarities) != 1:
            raise ValueError("score() accepts one embedding pair; use score_many().")
        return ScoreResult(
            self.name,
            float(dissimilarities[0]),
            {"cosine_similarity": float(similarities[0])},
        )

    def score_many(self, predictions: Any, references: Any = None, **_: Any):
        if references is None:
            raise ValueError("References are required for semantic dissimilarity.")
        similarities, dissimilarities = self._pairwise(predictions, references)
        return [
            ScoreResult(
                self.name,
                float(dissimilarity),
                {"cosine_similarity": float(similarity)},
            )
            for similarity, dissimilarity in zip(similarities, dissimilarities)
        ]

    def values(self, predictions: Any, references: Any):
        """Return scalar values for compatibility with legacy helper APIs."""
        return [result.value for result in self.score_many(predictions, references)]
