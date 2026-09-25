from typing import Dict, Any
import importlib
import logging

from omegaconf import DictConfig

EVALUATOR_REGISTRY: Dict[str, Any] = {}
EVALUATOR_IMPORT_ERRORS: Dict[str, Exception] = {}
logger = logging.getLogger("evaluator")


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        if evaluator_handler_name in EVALUATOR_IMPORT_ERRORS:
            raise RuntimeError(
                f"{evaluator_handler_name} could not be imported. Original error: "
                f"{EVALUATOR_IMPORT_ERRORS[evaluator_handler_name]}"
            ) from EVALUATOR_IMPORT_ERRORS[evaluator_handler_name]
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


def _register_evaluator_from_module(module_name, evaluator_class_name):
    try:
        module = importlib.import_module(module_name)
        evaluator_class = getattr(module, evaluator_class_name)
    except Exception as exc:
        EVALUATOR_IMPORT_ERRORS[evaluator_class_name] = exc
        logger.warning(
            "Skipping evaluator registration for %s: %s",
            evaluator_class_name,
            exc,
        )
        return
    _register_evaluator(evaluator_class)


# Register benchmark evaluators. Import failures are deferred so one benchmark's
# optional dependencies do not prevent running another benchmark.
for _module_name, _evaluator_class_name in (
    ("evals.lm_eval", "LMEvalEvaluator"),
    ("evals.mllmu", "MLLMUBenchEvaluator"),
    ("evals.fiubench", "FIUBenchEvaluator"),
    ("evals.clear", "CLEAREvaluator"),
    ("evals.covubench", "CoVUBenchEvaluator"),
    ("evals.sua", "SUAAttackEvaluator"),
    ("evals.figstep", "FigStepAttackEvaluator"),
    ("evals.image_rephrase", "ImageRephraseAttackEvaluator"),
    ("evals.jailbreak", "JailbreakAttackEvaluator"),
    ("evals.multimodal_benchmarks", "MultimodalBenchmarkEvaluator"),
):
    _register_evaluator_from_module(_module_name, _evaluator_class_name)
