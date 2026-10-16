"""Reference-model comparison with original CLEAR/FIUBench formulas."""

import logging

import numpy as np

from evals.metrics import register_metric

from .base import unlearning_metric

logger = logging.getLogger("evaluator")


def clear_quality(self, unlearned_data, retained_data):
    task_key = "eval_log_forget.json"
    if task_key not in unlearned_data or task_key not in retained_data:
        return {}
    first = self._task_truth_distribution(unlearned_data[task_key])
    second = self._task_truth_distribution(retained_data[task_key])
    if first.size == 0 or second.size == 0:
        return {}
    try:
        ks_result = self._ks_scorer.score(first, second)
        js_result = self._js_scorer.score(first, second)
    except ImportError:
        logger.warning("scipy is not installed; skipping CLEAR forget-quality metrics.")
        return {}
    return {
        "KS test p-value": float(ks_result.details["pvalue"]),
        "JS metric": float(js_result.value),
    }


def fiu_quality(self, logs, retained_logs):
    """Reproduce FIUBench's aggregate_eval_stat.get_forget_quality."""
    task_key = "eval_forget_log.json"
    if task_key not in logs:
        return {}
    unlearned = logs[task_key]
    retained = retained_logs.get(task_key, retained_logs)
    required = {"avg_paraphrased_loss", "average_perturb_loss"}
    if not required.issubset(unlearned) or not required.issubset(retained):
        return {}

    output = {}
    for source_key, output_key in (
        ("mink", "Mink"),
        ("mink++", "Mink++"),
        ("exact_match", "Exact Match"),
    ):
        values = unlearned.get(source_key)
        if values:
            output[output_key] = float(np.mean(values))

    unlearned_paraphrase = self._upstream_values(unlearned, "avg_paraphrased_loss")
    unlearned_perturbed = self._upstream_values(unlearned, "average_perturb_loss")
    retained_paraphrase = self._upstream_values(retained, "avg_paraphrased_loss")
    retained_perturbed = self._upstream_values(retained, "average_perturb_loss")
    if (
        unlearned_perturbed.ndim != 2
        or retained_perturbed.ndim != 2
        or not unlearned_paraphrase.size
        or not retained_paraphrase.size
    ):
        return output
    unlearned_truth = np.exp(unlearned_perturbed.mean(axis=-1) - unlearned_paraphrase)
    retained_truth = np.exp(retained_perturbed.mean(axis=-1) - retained_paraphrase)
    try:
        ks_result = self._ks_scorer.score(unlearned_truth, retained_truth)
    except ImportError:
        logger.warning("scipy is not installed; skipping FIUBench KS metrics.")
        return output
    output.update(
        {
            "Forget Quality": float(ks_result.details["pvalue"]),
            "KS Test PVal Forget": float(ks_result.details["pvalue"]),
            "KS Test Forget": float(ks_result.details["statistic"]),
        }
    )
    return output


@register_metric
@unlearning_metric("clear_forget_quality")
def clear_metric_forget_quality(model, *, protocol, logs, reference_logs, **kwargs):
    reference = reference_logs.get("retain_model_logs")
    value = (
        protocol._evaluate_forget_quality(logs, reference)
        if reference is not None
        else {}
    )
    return {"agg_value": None, "summary": value}


@register_metric
@unlearning_metric("fiubench_forget_quality")
def fiubench_metric_forget_quality(model, *, protocol, logs, reference_logs, **kwargs):
    reference = reference_logs.get("retain_model_logs")
    value = (
        protocol._upstream_forget_quality(logs, reference)
        if reference is not None
        else {}
    )
    return {"agg_value": None, "summary": value}
