"""Multiple-choice answer extraction and accuracy."""

import re

from evals.metrics import register_metric

from .base import unlearning_metric
from .utils import generation_samples


def extract_choice(answer, options):
    text = str(answer).strip().upper()
    valid = set(options)
    match = re.search(r"\b([A-F])\b", text)
    if match and match.group(1) in valid:
        return match.group(1)
    return text[:1] if text[:1] in valid else None


def choice_score(prediction, reference, options):
    predicted = extract_choice(prediction, options)
    target = extract_choice(reference, options) or reference.upper()
    return float(predicted == target), predicted, target


def extract_mllmu_choice(answer, options, strategy="first_char"):
    answer = str(answer).strip()
    if not answer:
        return None
    option_keys = {str(key).upper() for key in options.keys()}
    if strategy == "first_valid":
        for char in answer.upper():
            if char in option_keys:
                return char
        return None
    first = answer[0].upper()
    return first if first in option_keys else None


@register_metric
@unlearning_metric("choice_accuracy")
def choice_accuracy(model, *, pre_compute, access_key="generation", **kwargs):
    samples = generation_samples(pre_compute[access_key])
    results = {
        index: {
            "score": choice_score(
                item["prediction"], item["reference"], item.get("options", {})
            )[0]
        }
        for index, item in samples.items()
    }
    values = [item["score"] for item in results.values()]
    return {
        "agg_value": sum(values) / len(values) if values else None,
        "value_by_index": results,
    }
