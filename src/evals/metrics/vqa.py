"""VQAv2/VizWiz calculations from open-mmunlearning, including normalization."""

import re

import numpy as np

from evals.metrics import register_metric

from .base import BaseScorer, ScoreResult, unlearning_metric
from .utils import score_predictions


def normalize_vqa_answer(value):
    text = str(value or "").casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    digit_map = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    tokens = [
        digit_map.get(token, token)
        for token in text.split()
        if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def vqa_score(prediction, references):
    normalized_prediction = normalize_vqa_answer(prediction)
    normalized_references = [normalize_vqa_answer(item) for item in references]
    if not normalized_references:
        return 0.0
    annotator_scores = []
    for index in range(len(normalized_references)):
        matches = sum(
            normalized_prediction == reference
            for other_index, reference in enumerate(normalized_references)
            if other_index != index
        )
        annotator_scores.append(min(1.0, matches / 3.0))
    return float(np.mean(annotator_scores))


class VQAScorer(BaseScorer):
    name = "vqa"

    def score(self, prediction, reference=None, **kwargs):
        return ScoreResult(self.name, vqa_score(prediction, reference or []))


@register_metric
@unlearning_metric("vqa")
def vqa(model, *, pre_compute, access_key="generation", **kwargs):
    return score_predictions(
        pre_compute, VQAScorer(), access_key=access_key, reference_key="references"
    )
