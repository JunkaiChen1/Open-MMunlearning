"""Original benchmark utility aggregation; context supplies evaluator settings."""

import math

import numpy as np

from evals.metrics import register_metric

from .base import unlearning_metric
from .utils import mean_non_nan as _mean


def clear_harmonic_mean(values):
    values = [
        float(value)
        for value in values
        if isinstance(value, (int, float, np.integer, np.floating))
        and not math.isnan(float(value))
        and float(value) >= 0
    ]
    if not values:
        return None
    if any(value == 0 for value in values):
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def clear_utility(self, eval_results):
    aggregated_results = {}
    for task_file, task_result in eval_results.items():
        if not isinstance(task_result, dict) or "avg_gt_loss" not in task_result:
            continue
        task_name = self._task_name(task_file)
        probability = self._probability_metric(task_file, task_result)
        if probability is not None:
            aggregated_results[f"Prob. {task_name}"] = probability

        if not task_result.get("rougeL_recall") and task_result.get("generated_text"):
            task_result.update(
                self._generated_rouge_recall(task_result["generated_text"])
            )
        rouge_values = self._flatten_numeric(task_result.get("rougeL_recall", {}))
        if rouge_values:
            aggregated_results[f"ROUGE {task_name}"] = float(np.mean(rouge_values))

        if (
            "avg_paraphrased_loss" in task_result
            and "average_perturb_loss" in task_result
            and task_result["avg_paraphrased_loss"]
            and task_result["average_perturb_loss"]
        ):
            truth_ratio = self._task_truth_distribution(task_result)
            mode = "forget" if "forget" in task_file else "retain"
            transformed = [
                self._truth_ratio_transform_scorer.score(value, mode=mode).value
                for value in truth_ratio
            ]
            aggregated_results[f"Truth Ratio {task_name}"] = float(np.mean(transformed))

    # Keep the historical image-task Model Utility comparable.  The
    # newly attached QA-only retain scores are reported separately.
    utility_values = [
        value
        for key, value in aggregated_results.items()
        if "Forget" not in key and "Pure Text" not in key
    ]
    model_utility = self._harmonic_mean(utility_values)
    if model_utility is not None:
        aggregated_results["Model Utility"] = model_utility

    text_utility_values = [
        value for key, value in aggregated_results.items() if "Pure Text Retain" in key
    ]
    pure_text_utility = self._harmonic_mean(text_utility_values)
    if pure_text_utility is not None:
        aggregated_results["Pure Text Utility"] = pure_text_utility

    if self.eval_cfg.get("include_extended_group_metrics", False):
        self._add_group_metrics(aggregated_results)
    return aggregated_results


def clear_group_utility(self, aggregated_results):
    groups = {
        "Real metric": ("Real Faces", "Real World"),
        "Retain metric": ("Retain", "Retain FaceRec"),
        "Forget metric": ("Forget", "Forget FaceRec"),
    }
    for group_name, task_names in groups.items():
        values = [
            value
            for key, value in aggregated_results.items()
            if any(task_name in key for task_name in task_names)
        ]
        group_value = self._harmonic_mean(values)
        if group_value is not None:
            aggregated_results[group_name] = group_value


def fiu_utility(self, logs):
    """Reproduce FIUBench's aggregate_eval_stat.get_model_utility."""
    output = {}
    names = {
        "eval_forget_log.json": "Forget",
        "eval_retain_log.json": "Retain",
    }
    for task_file, task_logs in logs.items():
        if task_file not in names or not isinstance(task_logs, dict):
            continue
        required = {
            "avg_gt_loss",
            "average_perturb_loss",
            "avg_paraphrased_loss",
            "rougeL_recall",
        }
        if not required.issubset(task_logs):
            continue
        true_losses = self._upstream_values(task_logs, "avg_gt_loss")
        if true_losses.size == 0:
            continue
        true_probs = np.asarray(
            [self._answer_probability_scorer.score(loss).value for loss in true_losses],
            dtype=float,
        )
        if "eval_forget_log" in task_file:
            probability = float(np.mean(true_probs))
        else:
            false_losses = self._upstream_values(task_logs, "average_perturb_loss")
            if false_losses.ndim != 2 or false_losses.shape[0] != true_losses.size:
                continue
            probability = float(
                np.mean(
                    [
                        self._contrastive_probability_scorer.score(
                            target_loss, competing_losses
                        ).value
                        for target_loss, competing_losses in zip(
                            true_losses, false_losses
                        )
                    ]
                )
            )
        name = names[task_file]
        output[f"Prob. {name}"] = probability

        rouge_values = self._upstream_values(task_logs, "rougeL_recall")
        if rouge_values.size:
            output[f"ROUGE {name}"] = float(np.mean(rouge_values))

        paraphrase = self._upstream_values(task_logs, "avg_paraphrased_loss")
        perturbed = self._upstream_values(task_logs, "average_perturb_loss")
        if (
            paraphrase.size
            and perturbed.ndim == 2
            and perturbed.shape[0] == paraphrase.size
        ):
            ratio = np.exp(perturbed.mean(axis=-1) - paraphrase)
            mode = "forget" if "forget" in task_file else "retain"
            transformed = [
                self._truth_ratio_transform_scorer.score(value, mode=mode).value
                for value in ratio
            ]
            output[f"Truth Ratio {name}"] = float(np.mean(transformed))

        if "retain" in task_file and task_logs.get("gpt"):
            output[f"GPT {name}"] = float(np.mean(task_logs["gpt"]))
        if "retain" in task_file and task_logs.get("exact_match"):
            output[f"EM {name}"] = float(np.mean(task_logs["exact_match"]))

    utility_values = [value for key, value in output.items() if "Forget" not in key]
    if utility_values:
        if any(value == 0 for value in utility_values):
            output["Model Utility"] = 0.0
        else:
            output["Model Utility"] = float(
                len(utility_values) / sum(1.0 / value for value in utility_values)
            )
    return output


def covu_scaled_metrics(forget_samples, test_samples, retain_samples):
    forget_em = _mean(sample.get("keyword_recall") for sample in forget_samples)
    test_em = _mean(sample.get("keyword_recall") for sample in test_samples)
    retain_em = _mean(sample.get("keyword_recall") for sample in retain_samples)
    rouge_l = _mean(sample.get("rougeL_recall") for sample in retain_samples)
    divergence = _mean(
        sample.get("semantic_dissimilarity") for sample in forget_samples
    )

    metrics = {}
    if forget_em is not None:
        metrics["Efficacy"] = float((1.0 - forget_em) * 100.0)
    if test_em is not None:
        metrics["Generality"] = float((1.0 - test_em) * 100.0)
    if divergence is not None:
        metrics["Divergence"] = divergence
    if rouge_l is not None:
        metrics["Fluency"] = float(rouge_l * 100.0)
    if retain_em is not None:
        metrics["Specificity"] = float(retain_em * 100.0)
    return metrics, {
        "forget_keyword_em": forget_em,
        "test_keyword_em": test_em,
        "retain_keyword_em": retain_em,
        "retain_rougeL_recall": rouge_l,
        "forget_semantic_dissimilarity": divergence,
    }


def covu_utility(self, logs):
    forget_samples = self._samples(logs, "forget")
    test_samples = self._samples(logs, "test")
    retain_samples = self._samples(logs, "retain")
    forget_log = logs.get("splits", {}).get("forget", {})
    if not forget_log.get("divergence_completed", False):
        forget_samples_for_metrics = [
            {
                key: value
                for key, value in sample.items()
                if key != "semantic_dissimilarity"
            }
            for sample in forget_samples
        ]
    else:
        forget_samples_for_metrics = forget_samples

    metrics, raw_metrics = self._scaled_metrics(
        forget_samples_for_metrics, test_samples, retain_samples
    )
    summary = dict(metrics)
    summary["metric_scale"] = "0-100"
    summary["raw_metrics"] = raw_metrics
    summary["sample_counts"] = {
        split: len(self._samples(logs, split, completed_only=False))
        for split in ("forget", "test", "retain")
    }
    summary["keyword_scored_counts"] = {
        split: sum(
            sample.get("keyword_recall") is not None
            for sample in self._samples(logs, split, completed_only=False)
        )
        for split in ("forget", "test", "retain")
    }
    summary["by_question_type"] = self._breakdown(logs, "question_type")
    summary["by_domain"] = self._breakdown(logs, "type")
    return summary


@register_metric
@unlearning_metric("clear_model_utility")
def clear_metric_model_utility(model, *, protocol, logs, **kwargs):
    return {"agg_value": None, "summary": protocol._compute_model_utility(logs)}


@register_metric
@unlearning_metric("fiubench_model_utility")
def fiubench_metric_model_utility(model, *, protocol, logs, **kwargs):
    return {"agg_value": None, "summary": protocol._upstream_model_utility(logs)}


@register_metric
@unlearning_metric("covubench_summary")
def covubench_metric_summary(model, *, protocol, logs, **kwargs):
    return {"agg_value": None, "summary": protocol._summarize(logs)}
