"""Metric registration and YAML composition, following OpenUnlearning."""

from .base import UnlearningMetric

METRICS_REGISTRY = {}


def register_metric(metric):
    if metric.name in METRICS_REGISTRY:
        raise ValueError(f"Metric {metric.name!r} is already registered.")
    METRICS_REGISTRY[metric.name] = metric
    return metric


def get_metrics(configs):
    metrics = {}
    for name, config in configs.items():
        handler = config.get("handler")
        if handler not in METRICS_REGISTRY:
            raise ValueError(f"Unknown metric handler: {handler!r}")
        prototype = METRICS_REGISTRY[handler]
        metrics[name] = UnlearningMetric(
            prototype.name,
            prototype._metric_fn,
            get_metrics(config.get("pre_compute", {})),
            required_inputs=prototype.required_inputs,
        )
    return metrics


def get_input_requirements(configs):
    required = set()
    for config in configs.values():
        prototype = METRICS_REGISTRY[config["handler"]]
        required.update(prototype.input_requirements(config))
        required.update(get_input_requirements(config.get("pre_compute", {})))
    return required


# Import each metric to register its YAML handler.
from . import (  # noqa: E402,F401
    accuracy,
    choice_accuracy,
    forget_quality,
    gpt_score,
    gradnorm,
    mia_auc,
    mmvet_match,
    model_utility,
    reference_loss,
    refusal_rate,
    statistics,
    target_recovery,
    unsafe_rate,
    vqa,
)  # noqa: E402,F401
from ._min_k_utils import token_log_probability_statistics  # noqa: E402,F401
from .answer_probability import (  # noqa: E402
    AnswerProbabilityScorer,  # noqa: E402,F401
    ContrastiveProbabilityScorer,  # noqa: E402,F401
)

# Public MIA execution entry points share the metric implementations.
from .base import (  # noqa: E402
    AllAttacks,  # noqa: F401
    Attack,  # noqa: F401
    BaseScorer,  # noqa: E402,F401
    Scorer,  # noqa: E402,F401
    ScoreResult,  # noqa: E402,F401
)
from .bleu import BleuScorer  # noqa: E402,F401
from .contains import ContainsScorer  # noqa: E402,F401
from .exact_match import ExactMatchScorer  # noqa: E402,F401
from .gradnorm import GradNormAttack  # noqa: E402,F401
from .js import JensenShannonDistanceScorer  # noqa: E402,F401
from .keyword_recall import (  # noqa: E402
    KeywordRecallScorer,  # noqa: E402,F401
    parse_keywords,  # noqa: E402,F401
)
from .ks import TwoSampleKSScorer  # noqa: E402,F401
from .loss_mia import (  # noqa: E402
    LOSSAttack,  # noqa: E402,F401
    LossMIAScorer,  # noqa: E402,F401
)
from .mia_auc import get_attacker, run_mia_auc  # noqa: E402,F401
from .min_k import (  # noqa: E402
    MinKProbAttack,  # noqa: E402,F401
    MinKScorer,  # noqa: E402,F401
    StandardMinKScorer,  # noqa: E402,F401
)
from .min_k_plus_plus import (  # noqa: E402
    MinKPlusPlusAttack,  # noqa: E402,F401
    MinKPlusPlusScorer,  # noqa: E402,F401
    StandardMinKPlusPlusScorer,  # noqa: E402,F401
)
from .reference_loss import ReferenceAttack  # noqa: E402,F401
from .rouge import RougeScorer  # noqa: E402,F401
from .semantic import SemanticDissimilarityScorer  # noqa: E402,F401
from .truth_ratio import (  # noqa: E402
    PerturbationTruthRatioScorer,  # noqa: E402,F401
    TruthRatioTransformScorer,  # noqa: E402,F401
)
from .utils import normalize_text  # noqa: E402,F401
from .zlib_mia import (  # noqa: E402
    ZLIBAttack,  # noqa: E402,F401
    ZlibMIAScorer,  # noqa: E402,F401
)
