"""Unified scorer layer for multimodal benchmark evaluations."""

from .base import BaseScorer, ScoreResult, Scorer
from .distribution import JensenShannonDistanceScorer, TwoSampleKSScorer
from .keywords import KeywordRecallScorer, parse_keywords
from .likelihood import (
    AnswerProbabilityScorer,
    ContrastiveProbabilityScorer,
    PerturbationTruthRatioScorer,
    TruthRatioTransformScorer,
)
from .mia import (
    LossMIAScorer,
    MinKPlusPlusScorer,
    MinKScorer,
    StandardMinKPlusPlusScorer,
    StandardMinKScorer,
    ZlibMIAScorer,
    token_log_probability_statistics,
)
from .registry import SCORER_REGISTRY, ScorerRegistry, get_scorer, register_scorer
from .semantic import SemanticDissimilarityScorer
from .text import (
    BleuScorer,
    ContainsScorer,
    ExactMatchScorer,
    RougeScorer,
    normalize_text,
)


for _name, _factory in {
    "answer_probability": AnswerProbabilityScorer,
    "bleu": BleuScorer,
    "contains": ContainsScorer,
    "contrastive_probability": ContrastiveProbabilityScorer,
    "exact_match": ExactMatchScorer,
    "jensen_shannon_distance": JensenShannonDistanceScorer,
    "keyword_recall": KeywordRecallScorer,
    "loss": LossMIAScorer,
    "loss_mia": LossMIAScorer,
    "min_k": MinKScorer,
    "min_k_plus_plus": MinKPlusPlusScorer,
    "rouge": RougeScorer,
    "semantic_dissimilarity": SemanticDissimilarityScorer,
    "standard_min_k": StandardMinKScorer,
    "standard_min_k_plus_plus": StandardMinKPlusPlusScorer,
    "truth_ratio_transform": TruthRatioTransformScorer,
    "truth_ratio": PerturbationTruthRatioScorer,
    "two_sample_ks": TwoSampleKSScorer,
    "zlib": ZlibMIAScorer,
    "zlib_mia": ZlibMIAScorer,
}.items():
    SCORER_REGISTRY.register(_name, _factory)


__all__ = [
    "AnswerProbabilityScorer",
    "BaseScorer",
    "BleuScorer",
    "ContainsScorer",
    "ContrastiveProbabilityScorer",
    "ExactMatchScorer",
    "JensenShannonDistanceScorer",
    "KeywordRecallScorer",
    "LossMIAScorer",
    "MinKPlusPlusScorer",
    "MinKScorer",
    "PerturbationTruthRatioScorer",
    "RougeScorer",
    "SCORER_REGISTRY",
    "ScoreResult",
    "Scorer",
    "ScorerRegistry",
    "SemanticDissimilarityScorer",
    "StandardMinKPlusPlusScorer",
    "StandardMinKScorer",
    "TruthRatioTransformScorer",
    "TwoSampleKSScorer",
    "ZlibMIAScorer",
    "get_scorer",
    "normalize_text",
    "parse_keywords",
    "register_scorer",
    "token_log_probability_statistics",
]
