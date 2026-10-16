"""Benchmark accuracy, category aggregation and POPE confusion statistics."""

import re

import numpy as np

from evals.metrics import register_metric

from .base import unlearning_metric
from .choice_accuracy import extract_choice
from .exact_match import ExactMatchScorer
from .mmvet_match import mmvet_match
from .utils import generation_samples, normalize_benchmark_text
from .vqa import vqa_score


def percentage(numerator, denominator):
    return (numerator / denominator) * 100 if denominator else 0.0


def mllmu_accuracy(counts, *, fill_in_the_blank=False):
    image_key = (
        "image_textual_accuracy"
        if fill_in_the_blank
        else "Image-Textual Question Accuracy"
    )
    text_key = (
        "pure_text_accuracy" if fill_in_the_blank else "Pure Text Question Accuracy"
    )
    return {
        image_key: percentage(
            counts["image_textual_correct"], counts["image_textual_total"]
        ),
        text_key: percentage(counts["pure_text_correct"], counts["pure_text_total"]),
        "counts": counts,
    }


@register_metric
@unlearning_metric("accuracy")
def accuracy(model, *, pre_compute, benchmark="gqa", access_key="generation", **kwargs):
    samples = generation_samples(pre_compute[access_key])
    results = {}
    for index, item in samples.items():
        record = {
            "answer": item["reference"],
            "options": item.get("options", {}),
            "references": item.get("references", []),
        }
        value, prediction, target = score_multimodal_record(
            benchmark, record, item["prediction"]
        )
        results[index] = {"score": value, "prediction": prediction, "target": target}
    values = [item["score"] for item in results.values()]
    return {
        "agg_value": float(np.mean(values)) if values else 0.0,
        "value_by_index": results,
    }


def score_multimodal_record(benchmark, record, generated):
    if benchmark == "pope":
        prediction = normalize_benchmark_text(generated)
        target = normalize_benchmark_text(record["answer"])
        prediction_match = re.search(r"\b(yes|no)\b", prediction)
        target_match = re.search(r"\b(yes|no)\b", target)
        prediction = prediction_match.group(1) if prediction_match else prediction
        target = target_match.group(1) if target_match else target
        return float(prediction == target), prediction, target
    if record["options"] or benchmark == "mmbench":
        prediction = extract_choice(generated, record["options"])
        target = (
            extract_choice(record["answer"], record["options"])
            or record["answer"].upper()
        )
        return float(prediction == target), prediction, target
    if benchmark == "mmvet":
        return mmvet_match(generated, record["answer"]), generated, record["answer"]
    if benchmark == "vizwiz":
        return (
            vqa_score(generated, record["references"]),
            generated,
            record["answer"],
        )
    if benchmark == "vqav2":
        return (
            vqa_score(generated, record["references"]),
            generated,
            record["answer"],
        )
    if benchmark == "gqa":
        prediction = normalize_benchmark_text(generated)
        target = normalize_benchmark_text(record["answer"])
        return float(prediction == target), prediction, target
    exact = ExactMatchScorer().score(generated, record["answer"]).value
    return float(exact), generated, record["answer"]


def summarize_accuracy(benchmark, details):
    scores = [item["score"] for item in details]
    result = {
        "count": len(details),
        "accuracy": float(np.mean(scores)) if scores else 0.0,
    }
    categories = sorted({item["category"] for item in details})
    result["category_accuracy"] = {
        category: float(
            np.mean([item["score"] for item in details if item["category"] == category])
        )
        for category in categories
    }
    if benchmark == "pope":
        tp = sum(
            item["prediction"] == "yes" and item["target"] == "yes" for item in details
        )
        fp = sum(
            item["prediction"] == "yes" and item["target"] == "no" for item in details
        )
        fn = sum(
            item["prediction"] == "no" and item["target"] == "yes" for item in details
        )
        tn = sum(
            item["prediction"] == "no" and item["target"] == "no" for item in details
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
                "yes_rate": (tp + fp) / len(details) if details else 0.0,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
        )
    if benchmark == "mmvet":
        result["rougeL"] = (
            float(np.mean([item["rougeL"] for item in details])) if details else 0.0
        )
    if benchmark in {"vizwiz", "vqav2"}:
        result["vqa_accuracy"] = result["accuracy"]
    return result
