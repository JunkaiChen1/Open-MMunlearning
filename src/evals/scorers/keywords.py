"""Keyword and phrase matching used by FIU and CoVUBench."""

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from .base import BaseScorer, ScoreResult
from .text import normalize_text


def _is_nan(value: Any) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def parse_keywords(value: Any) -> list[str]:
    """Flatten strings, JSON values, mappings, and list-like keyword fields."""
    if value is None or _is_nan(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return parse_keywords(json.loads(value))
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Mapping):
        keywords: list[str] = []
        for nested_value in value.values():
            keywords.extend(parse_keywords(nested_value))
        return keywords
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        # OmegaConf ListConfig is iterable but is not necessarily a list.
        if not isinstance(value, (str, Mapping)):
            keywords = []
            for nested_value in value:
                keywords.extend(parse_keywords(nested_value))
            return keywords
    text = str(value).strip()
    return [text] if text else []


class KeywordRecallScorer(BaseScorer):
    """Fraction of reference keyword phrases found in a prediction."""

    name = "keyword_recall"

    def __init__(self, normalization: str = "canonical"):
        if normalization not in {"canonical", "lower"}:
            raise ValueError(
                "KeywordRecallScorer normalization must be 'canonical' or 'lower'."
            )
        self.normalization = normalization

    def _normalize(self, value: Any) -> str:
        if self.normalization == "lower":
            return str(value).lower()
        return normalize_text(value)

    def match_details(self, prediction: Any, keywords: Any) -> dict[str, Any]:
        parsed_keywords = parse_keywords(keywords)
        normalized_prediction = self._normalize(prediction)
        matches = [
            self._normalize(keyword) in normalized_prediction
            for keyword in parsed_keywords
        ]
        recall = None
        if parsed_keywords:
            recall = float(sum(matches) / len(parsed_keywords))
        return {
            "keywords": parsed_keywords,
            "matches": matches,
            "recall": recall,
            "matched_count": sum(matches),
            "keyword_count": len(parsed_keywords),
        }

    def score(self, prediction: Any, reference: Any = None, **_: Any) -> ScoreResult:
        details = self.match_details(prediction, reference)
        return ScoreResult(self.name, details["recall"], details)
