"""Benchmark summary statistics with original filtering and denominators."""

import numpy as np

from evals.metrics import register_metric

from .base import unlearning_metric


def clear_statistics(self, logs):
    summary = {}
    for task_name, task_logs in logs.items():
        if not isinstance(task_logs, dict):
            continue
        for metric_name, value in task_logs.items():
            if (
                metric_name in self.metrics
                and isinstance(value, dict)
                and "agg_value" in value
            ):
                continue
            if metric_name == "generated_text":
                continue
            values = self._flatten_numeric(value)
            if not values:
                continue
            summary[f"{task_name}/{metric_name}/mean"] = float(np.mean(values))
            summary[f"{task_name}/{metric_name}/count"] = len(values)

    return summary


def fiu_statistics(self, logs):
    summary = {}
    for task_name, task_logs in logs.items():
        if not isinstance(task_logs, dict):
            continue
        for metric_name, value in task_logs.items():
            if (
                metric_name in self.metrics
                and isinstance(value, dict)
                and "agg_value" in value
            ):
                continue
            if metric_name == "generated_text":
                continue
            values = self._flatten_numeric(value)
            if not values:
                continue
            summary[f"{task_name}/{metric_name}/mean"] = float(np.mean(values))
            summary[f"{task_name}/{metric_name}/count"] = len(values)

        if "avg_gt_loss" in task_logs:
            avg_gt_loss = np.array(self._flatten_numeric(task_logs["avg_gt_loss"]))
            if avg_gt_loss.size:
                summary[f"{task_name}/gt_probability/mean"] = float(
                    np.mean(
                        [
                            self._answer_probability_scorer.score(loss).value
                            for loss in avg_gt_loss
                        ]
                    )
                )
        if "avg_paraphrased_loss" in task_logs and "average_perturb_loss" in task_logs:
            paraphrased = np.array(
                list(task_logs["avg_paraphrased_loss"].values()), dtype=float
            )
            perturbed = np.array(
                list(task_logs["average_perturb_loss"].values()), dtype=float
            )
            if paraphrased.size and perturbed.size:
                perturbed_mean = perturbed.mean(axis=-1)
                ratio = np.exp(perturbed_mean - paraphrased)
                mode = "forget" if "forget" in task_name else "retain"
                transformed = [
                    self._truth_ratio_transform_scorer.score(value, mode=mode).value
                    for value in ratio
                ]
                summary[f"{task_name}/aggregate_truth_ratio/mean"] = float(
                    np.mean(transformed)
                )
    return summary


@register_metric
@unlearning_metric("clear_statistics")
def clear_metric_statistics(model, *, protocol, logs, **kwargs):
    return {"agg_value": None, "summary": protocol._summary_statistics(logs)}


@register_metric
@unlearning_metric("fiubench_statistics")
def fiubench_metric_statistics(model, *, protocol, logs, **kwargs):
    return {"agg_value": None, "summary": protocol._summary_statistics(logs)}
