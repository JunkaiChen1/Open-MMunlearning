"""Metric execution, pre-computation, datasets, and reference logs."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class UnlearningMetric:
    def __init__(self, name, metric_fn, pre_compute_metrics=None, required_inputs=()):
        self.name = name
        self._metric_fn = metric_fn
        self.pre_compute_metrics = pre_compute_metrics or {}
        self.required_inputs = required_inputs

    def input_requirements(self, config):
        required = self.required_inputs
        required = required(config) if callable(required) else required
        return set(required) | set(config.get("required_inputs", ()))

    def evaluate(self, model, metric_name, cache=None, **kwargs):
        cache = {} if cache is None else cache
        if metric_name in cache:
            return cache[metric_name]
        arguments = dict(kwargs)
        datasets = arguments.pop("datasets", None)
        collators = arguments.pop("collators", None)
        if datasets is not None:
            from data import get_datasets

            arguments["data"] = get_datasets(
                datasets,
                tokenizer=arguments.get("tokenizer"),
                template_args=arguments.get("template_args"),
            )
        if collators is not None:
            from data import get_collators

            arguments["collators"] = get_collators(
                collators, tokenizer=arguments.get("tokenizer")
            )
        pre_compute = {}
        for name, config in arguments.pop("pre_compute", {}).items():
            dependency_args = {**arguments, **dict(config)}
            pre_compute[config.get("access_key", name)] = self.pre_compute_metrics[
                name
            ](model, metric_name=name, cache=cache, **dependency_args)
        if pre_compute:
            arguments["pre_compute"] = pre_compute
        reference_logs = {}
        for name, config in arguments.pop("reference_logs", {}).items():
            path = config.get("path")
            if path in (None, "", "null", "None"):
                continue
            with Path(path).open(encoding="utf-8") as handle:
                logs = json.load(handle)
            include = config.get("include")
            reference_logs[name] = (
                {
                    item.get("access_key", key): logs.get(key)
                    for key, item in include.items()
                }
                if include is not None
                else logs
            )
        arguments["reference_logs"] = reference_logs
        for key in (
            "handler",
            "access_key",
            "phase",
            "task",
            "output",
            "required_inputs",
        ):
            arguments.pop(key, None)
        result = self._metric_fn(model, **arguments)
        cache[metric_name] = result
        return result

    def __call__(self, model, **kwargs):
        return self.evaluate(model, **kwargs)


class unlearning_metric:
    def __init__(self, name, *, required_inputs=()):
        self.name = name
        self.required_inputs = required_inputs

    def __call__(self, function):
        return UnlearningMetric(
            self.name, function, required_inputs=self.required_inputs
        )


def metric_result(value):
    """Wrap source-compatible benchmark results for dependency execution."""
    return {"agg_value": None, "value_by_index": value}


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

    def score(self, *args: Any, **kwargs: Any) -> ScoreResult: ...


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


class AllAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    MIN_K_PLUS_PLUS = "min_k++"
    GRADNORM = "gradnorm"
    RECALL = "recall"


class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        """Initialize attack with model and create dataloader."""
        from torch.utils.data import DataLoader

        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Setup attack-specific parameters."""
        pass

    def compute_batch_values(self, batch):
        """Process a batch through model to get needed statistics."""
        raise NotImplementedError

    def compute_score(self, sample_stats):
        """Compute MIA score for a single sample."""
        raise NotImplementedError

    def attack(self):
        """Run full MIA attack."""
        from tqdm import tqdm

        from .mia_auc import summarize_attack_scores

        all_scores = []
        all_indices = []

        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            indices = batch.pop("index").cpu().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]

            all_scores.extend(scores)
            all_indices.extend(indices)

        return summarize_attack_scores(all_scores, all_indices)
