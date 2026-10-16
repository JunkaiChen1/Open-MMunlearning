"""Shared result handling for metrics built from pre-computed predictions."""

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import numpy as np


def generation_samples(result):
    """Read the existing benchmark record layouts without changing their values."""
    payload = result.get("value_by_index", {})
    samples = result.get("samples")
    if samples is None and isinstance(payload, Mapping):
        samples = payload.get("samples", payload.get("details"))
        if samples is None and "generated_text" in payload:
            samples = [
                {"id": str(index), "prediction": pair[1], "reference": pair[2]}
                for index, pair in payload["generated_text"].items()
            ]
    if samples is None:
        raise ValueError("The generation dependency has no saved prediction records.")
    if isinstance(samples, Mapping):
        samples = [{"id": str(index), **sample} for index, sample in samples.items()]
    normalized = {}
    for index, sample in enumerate(samples):
        prediction = sample.get("generated_answer", sample.get("prediction"))
        reference = sample.get(
            "reference",
            sample.get("ground_truth", sample.get("target", sample.get("answer"))),
        )
        if prediction is None or reference is None:
            raise ValueError(
                f"Prediction or reference missing in generation record {index}."
            )
        # MLLMU can have several questions for the same image, so its image id
        # alone cannot be used as a unique sample key.
        key = str(sample.get("id", index))
        if key in normalized:
            raise ValueError(f"Duplicate generation record id: {key}")
        normalized[key] = {**sample, "prediction": prediction, "reference": reference}
    return normalized


def score_predictions(
    pre_compute, scorer, *, access_key="generation", reference_key="reference"
):
    samples = generation_samples(pre_compute[access_key])
    results = {
        index: {
            "score": scorer.score(item["prediction"], item[reference_key]).value,
            "prediction": item["prediction"],
            "reference": item[reference_key],
        }
        for index, item in samples.items()
    }
    values = [item["score"] for item in results.values() if item["score"] is not None]
    return {
        "agg_value": sum(values) / len(values) if values else None,
        "value_by_index": results,
    }


def score_inputs(
    pre_compute, calculate, *, access_key="generation", required_inputs=()
):
    """Apply a formula to saved sample inputs without repeating inference."""
    source = pre_compute[access_key]
    samples = source.get("samples")
    if samples is None:
        samples = source.get("value_by_index", {})
        samples = samples.get("samples", samples.get("details", samples))
    if isinstance(samples, Mapping):
        samples = [{"id": str(index), **sample} for index, sample in samples.items()]
    results = {}
    for index, sample in enumerate(samples):
        key = str(sample.get("id", index))
        if key in results:
            raise ValueError(f"Duplicate metric input id: {key}")
        missing = set(required_inputs) - sample.keys()
        if missing:
            raise ValueError(
                f"Metric input {key} is missing {sorted(missing)}; "
                "rerun its inference dependency with these required inputs."
            )
        results[key] = {
            "score": None
            if any(sample[field] is None for field in required_inputs)
            else calculate(sample)
        }
    values = [item["score"] for item in results.values() if item["score"] is not None]
    return {
        "agg_value": sum(values) / len(values) if values else None,
        "value_by_index": results,
    }


def sample_loss(sample):
    for key in ("avg_loss", "avg_gt_loss"):
        if key in sample:
            return sample[key]
    raise ValueError("The generation dependency has no average answer loss.")


def distribution_values(result):
    values = result.get("value_by_index", result)
    if isinstance(values, Mapping):
        values = [
            item.get("score", item.get("value")) if isinstance(item, Mapping) else item
            for item in values.values()
        ]
    return values


def normalize_text(value: Any) -> str:
    """Normalize text for exact or substring comparisons."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _as_vector(value: Any, *, name: str):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Distribution scorers require numpy.") from exc

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}.")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def normalize_benchmark_text(value):
    text = str(value or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\n\t")


def mean_non_nan(values):
    values = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def clear_flatten_numeric(value):
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    flattened = []
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray)):
            flattened.extend(
                float(x)
                for x in item
                if isinstance(x, (int, float, np.integer, np.floating))
                and not math.isnan(float(x))
            )
        elif isinstance(item, (int, float, np.integer, np.floating)):
            item = float(item)
            if not math.isnan(item):
                flattened.append(item)
    return flattened


def fiu_flatten_numeric(value):
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    flattened = []
    for item in values:
        if isinstance(item, (list, tuple)):
            flattened.extend(float(x) for x in item if isinstance(x, (int, float)))
        elif isinstance(item, (int, float)):
            flattened.append(float(item))
    return [x for x in flattened if not math.isnan(x)]


def clear_values_array(self, value):
    if isinstance(value, dict):
        value = list(value.values())
    try:
        return np.array(value, dtype=float)
    except (TypeError, ValueError):
        return np.array(self._flatten_numeric(value), dtype=float)


def fiu_values_array(task_logs, key):
    value = task_logs.get(key, {})
    if isinstance(value, dict):
        value = list(value.values())
    return np.asarray(value, dtype=float)
