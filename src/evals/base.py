"""Configured metric execution and storage, following OpenUnlearning."""

from evals.utils.storage import EvaluationLogs


class Evaluator(EvaluationLogs):
    def __init__(self, name, eval_cfg, **kwargs):
        super().__init__(eval_cfg, name=name, **kwargs)
        from evals.metrics import get_metrics

        self.metrics_cfg = eval_cfg.get("metrics", {})
        self.metrics = get_metrics(self.metrics_cfg)
        self.metric_cache = {}
        self.model = None

    def metric_names(self, *, phase="evaluation", task=None, split=None):
        """Select configured metrics in YAML order for a benchmark context."""
        for name, config in self.metrics_cfg.items():
            if config.get("phase", "evaluation") != phase:
                continue
            if config.get("task") is not None and config.get("task") != task:
                continue
            selected_split = config.get("split")
            if selected_split is not None and selected_split != split:
                continue
            yield name

    def evaluate_metric(self, name, *, model, cache, **kwargs):
        arguments = {"protocol": self, **kwargs, **dict(self.metrics_cfg[name])}
        return self.metrics[name](
            model,
            metric_name=name,
            cache=cache,
            **arguments,
        )

    def evaluate_metrics(
        self,
        model,
        *,
        scope="default",
        phase="evaluation",
        task=None,
        split=None,
        cache=None,
        **kwargs,
    ):
        """Execute every configured metric selected for this context."""
        cache = self.metric_cache.setdefault(scope, {}) if cache is None else cache
        if split is not None:
            kwargs.setdefault("split", split)
        for name in self.metric_names(phase=phase, task=task, split=split):
            result = self.evaluate_metric(name, model=model, cache=cache, **kwargs)
            yield name, result

    def merge_metric_result(self, logs, name, result):
        # Existing benchmark configurations retain their source log layout.
        if self.metrics_cfg[name].get("output") == "merge":
            logs.update(result.get("value_by_index", {}))
        else:
            logs[name] = result

    def metric_summary(self, logs, prefix=""):
        summary = {}
        for name, result in logs.items():
            if not isinstance(result, dict):
                continue
            key = f"{prefix}/{name}" if prefix else name
            if name in self.metrics and "agg_value" in result:
                if result["agg_value"] is not None:
                    summary[key] = result["agg_value"]
            elif name not in {"details", "value_by_index", "metadata"}:
                summary.update(self.metric_summary(result, key))
        return summary

    def summarize(self, logs):
        if not self.metrics:
            return super().summarize(logs)
        summary = {}
        for name in self.metric_names(phase="summary"):
            result = self.evaluate_metric(
                name,
                model=self.model,
                cache={},
                logs=logs,
            )
            summary.update(result.get("summary", {}))
            if result.get("agg_value") is not None:
                summary[name] = result["agg_value"]
        summary.update(self.metric_summary(logs))
        return summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        """Always recompute; overwrite is accepted for older callers only."""
        self.metric_cache = {}
        self.model = model
        try:
            implementation = getattr(self, "_evaluate", self._evaluate_metric_suite)
            return implementation(model, output_dir, overwrite, **kwargs)
        finally:
            self.metric_cache = {}
            self.model = None

    def _evaluate_metric_suite(self, model, output_dir, overwrite, **kwargs):
        model = self.prepare_model(model)
        output_dir = output_dir or self.eval_cfg.output_dir
        logs_path = self.get_logs_file_path(output_dir)
        summary_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        logs = {}
        for _name, _result in self.evaluate_metrics(model, cache=logs, **kwargs):
            self.save_logs(logs, logs_path)
            self.save_logs(self.summarize(logs), summary_path)
        return self.summarize(logs)
