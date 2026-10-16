"""Shared JSON storage for evaluators and benchmark metric protocols."""

import json
import logging
import os

logger = logging.getLogger("evaluator")


class EvaluationLogs:
    name = "Evaluator"

    def __init__(self, eval_cfg, *, name=None, **kwargs):
        self.eval_cfg = eval_cfg
        if name is not None:
            self.name = name

    def get_logs_file_path(self, output_dir, suffix="EVAL"):
        return os.path.join(output_dir, f"{self.name}_{suffix}.json")

    def load_logs_from_file(self, file):
        if os.path.exists(file):
            logger.info("Loading existing evaluations from %s", file)
            with open(file, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def save_logs(self, logs, file, *, indent=4, ensure_ascii=True, sort_keys=True):
        if sort_keys:
            logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w", encoding="utf-8") as handle:
                json.dump(logs, handle, indent=indent, ensure_ascii=ensure_ascii)
        except Exception as exc:
            raise RuntimeError(f"Failed to save {file}: {exc}") from exc

    def save_result(self, output_dir, result, summary, **storage_args):
        self.save_logs(result, self.get_logs_file_path(output_dir), **storage_args)
        self.save_logs(
            summary,
            self.get_logs_file_path(output_dir, suffix="SUMMARY"),
            **storage_args,
        )

    def prepare_model(self, model):
        model.eval()
        return model

    def summarize(self, result):
        return {key: value for key, value in result.items() if key != "details"}
