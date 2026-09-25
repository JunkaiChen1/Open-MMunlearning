"""Small, model-agnostic building blocks for benchmark scoring.

Evaluators own data loading, prompting, generation, and benchmark-specific
aggregation. Scorers only transform already available predictions, references,
losses, or embeddings into a named result.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ScoreResult:
    """A scalar score plus optional diagnostic values.

    ``value`` is the number that an evaluator normally aggregates. ``details``
    keeps auxiliary values such as individual ROUGE variants or matched
    keywords without forcing every benchmark to invent another return type.
    """

    name: str
    value: float | None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for evaluator logs."""
        return {"name": self.name, "value": self.value, **self.details}


class Scorer(Protocol):
    """Structural interface implemented by every scorer."""

    name: str

    def score(self, *args: Any, **kwargs: Any) -> ScoreResult:
        ...


class BaseScorer:
    """Convenience base class with callable and batch behavior."""

    name = "scorer"

    def __call__(self, *args: Any, **kwargs: Any) -> ScoreResult:
        return self.score(*args, **kwargs)

    def score_many(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoreResult]:
        if references is None:
            references = [None] * len(predictions)
        if len(predictions) != len(references):
            raise ValueError(
                "Predictions and references must have the same length, got "
                f"{len(predictions)} and {len(references)}."
            )
        return [
            self.score(prediction, reference, **kwargs)
            for prediction, reference in zip(predictions, references)
        ]
