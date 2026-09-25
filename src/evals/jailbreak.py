"""Evaluator for answer-free jailbreak prompt attacks."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from attacks.jailbreak import JailbreakAttack, JailbreakConfig


class JailbreakAttackEvaluator:
    """Run prefix, affirmative, and role-playing attacks on MLLMU forget data."""

    name = "Jailbreak"

    def __init__(self, eval_cfg, **kwargs):
        self.eval_cfg = eval_cfg
        self.processor = None

    @staticmethod
    def _to_container(value, default=None):
        if value is None:
            return default
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _image(value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
        raise TypeError(f"Unsupported jailbreak image type: {type(value)!r}")

    def _read_parquet(self, path):
        path = Path(path).expanduser()
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No parquet files found at {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    def _data_path(self):
        explicit = self.eval_cfg.get("data_path", None)
        if explicit:
            return explicit
        root = Path(str(self.eval_cfg.data_root)).expanduser()
        split = str(self.eval_cfg.get("split", "forget"))
        if split == "test":
            return root / "Test_Set"
        if split == "retain":
            ratio = int(self.eval_cfg.get("retain_ratio", 90))
            return root / f"retain_{ratio}" / "train-00000-of-00001.parquet"
        ratio = (
            split.removeprefix("forget_")
            if split.startswith("forget_")
            else str(self.eval_cfg.get("ratio", 10))
        )
        return root / f"forget_{ratio}" / "train-00000-of-00001.parquet"

    def _records(self):
        split = str(self.eval_cfg.get("split", "forget"))
        task_types = set(
            str(value)
            for value in self._as_list(
                self.eval_cfg.get("task_types", ["Image_Textual", "Pure_Text"])
            )
        )
        image_strategy = str(self.eval_cfg.get("image_strategy", "all"))
        records = []
        for _, row in self._read_parquet(self._data_path()).iterrows():
            row_id = str(row.get("ID", len(records)))
            images = self._as_list(
                row.get("images") if split == "test" else row.get("image")
            )
            if image_strategy == "first":
                images = images[:1]
            tasks = self._as_list(row.get("Mask_Task", []))
            for image_index, image_value in enumerate(images or [None]):
                for task_index, task in enumerate(tasks):
                    if not isinstance(task, dict):
                        continue
                    task_type = str(task.get("Type", ""))
                    if task_type not in task_types:
                        continue
                    question = str(task.get("Question", "")).replace("__", "[Blank]")
                    question += "\nPlease provide only the correct answer for [Blank]."
                    answer = str(task.get("Ground_Truth", ""))
                    if not question.strip() or not answer.strip():
                        continue
                    image = (
                        self._image(image_value)
                        if task_type == "Image_Textual"
                        else None
                    )
                    records.append(
                        {
                            "id": f"{row_id}:{image_index}:{task_index}",
                            "image": image,
                            "task_type": task_type,
                            "question": question,
                            "answer": answer,
                        }
                    )
        return records

    def _load_processor(self):
        if self.processor is None:
            processor_path = self.eval_cfg.get("processor_path", None)
            if not processor_path:
                raise ValueError("eval.jailbreak.processor_path must be set.")
            self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    @staticmethod
    def _save(output_dir, name, payload):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = (
            self.eval_cfg.get("overwrite", False) if overwrite is None else overwrite
        )
        output_dir = output_dir or self.eval_cfg.output_dir
        eval_path = Path(output_dir) / f"{self.name}_EVAL.json"
        if eval_path.exists() and not overwrite:
            with eval_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        attack_cfg = self._to_container(self.eval_cfg.get("jailbreak", {}), {}) or {}
        attack_cfg = dict(attack_cfg)
        attack = JailbreakAttack(
            self._load_processor(),
            kwargs.get("tokenizer", None),
            JailbreakConfig.from_mapping(attack_cfg),
        )
        result = attack.evaluate(model, self._records())
        result.update(
            {
                "attack": "jailbreak",
                "split": str(self.eval_cfg.get("split", "forget")),
                "config": attack_cfg,
            }
        )
        self._save(output_dir, f"{self.name}_EVAL.json", result)
        self._save(
            output_dir,
            f"{self.name}_SUMMARY.json",
            {key: value for key, value in result.items() if key != "details"},
        )
        return result
