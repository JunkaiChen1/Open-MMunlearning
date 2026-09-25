"""Evaluator for the FigStep-style visual prompt attack."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from attacks.figstep import FigStepAttack, FigStepConfig


class FigStepAttackEvaluator:
    """Run a black-box FigStep attack on image-conditioned MLLMU records."""

    name = "FigStep"

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
        raise TypeError(f"Unsupported FigStep image type: {type(value)!r}")

    def _read_parquet(self, path):
        path = Path(path).expanduser()
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No parquet files found at {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    def _data_path(self):
        if self.eval_cfg.get("data_path", None):
            return self.eval_cfg.data_path
        root = Path(str(self.eval_cfg.data_root)).expanduser()
        split = str(self.eval_cfg.get("split", "forget"))
        if split == "test":
            return root / "Test_Set"
        if split == "retain":
            ratio = int(self.eval_cfg.get("retain_ratio", 90))
            return root / f"retain_{ratio}" / "train-00000-of-00001.parquet"
        ratio = (
            str(split.removeprefix("forget_"))
            if split.startswith("forget_")
            else str(self.eval_cfg.get("ratio", 10))
        )
        return root / f"forget_{ratio}" / "train-00000-of-00001.parquet"

    def _reference_ids(self):
        path = self.eval_cfg.get("reference_ids_path", None)
        if not path:
            return None
        frame = self._read_parquet(path)
        return {str(value) for value in frame["ID"].tolist()}

    def _records(self):
        split = str(self.eval_cfg.get("split", "forget"))
        ids = self._reference_ids()
        image_strategy = str(self.eval_cfg.get("image_strategy", "all"))
        max_records = self.eval_cfg.get("max_records", None)
        records = []
        for _, row in self._read_parquet(self._data_path()).iterrows():
            row_id = str(row.get("ID", len(records)))
            if ids is not None and row_id not in ids:
                continue
            images = self._as_list(
                row.get("images") if split == "test" else row.get("image")
            )
            if image_strategy == "first":
                images = images[:1]
            tasks = self._as_list(row.get("Mask_Task", []))
            for image_index, image_value in enumerate(images):
                for task_index, task in enumerate(tasks):
                    if (
                        not isinstance(task, dict)
                        or task.get("Type") != "Image_Textual"
                    ):
                        continue
                    question = str(task.get("Question", "")).replace("__", "[Blank]")
                    question += "\nPlease provide only the correct answer for [Blank]."
                    answer = str(task.get("Ground_Truth", ""))
                    if question.strip() and answer.strip():
                        records.append(
                            {
                                "id": f"{row_id}:{image_index}:{task_index}",
                                "image": self._image(image_value),
                                "question": question,
                                "answer": answer,
                            }
                        )
                        if max_records and len(records) >= int(max_records):
                            return records
        return records

    def _load_processor(self):
        if self.processor is None:
            processor_path = self.eval_cfg.get("processor_path", None)
            if not processor_path:
                raise ValueError("eval.figstep.processor_path must be set.")
            self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    @staticmethod
    def _save(output_dir, name, payload):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = (
            self.eval_cfg.get("overwrite", False) if overwrite is None else overwrite
        )
        output_dir = output_dir or self.eval_cfg.output_dir
        eval_path = Path(output_dir) / "FigStep_EVAL.json"
        if eval_path.exists() and not overwrite:
            with eval_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        attack_cfg = self._to_container(self.eval_cfg.get("figstep", {}), {}) or {}
        attack_cfg = dict(attack_cfg)
        attack_cfg.setdefault(
            "artifacts_dir", str(Path(output_dir) / "checkpoint" / "figstep_prompts")
        )
        attack = FigStepAttack(
            self._load_processor(),
            kwargs.get("tokenizer", None),
            FigStepConfig.from_mapping(attack_cfg),
        )
        records = self._records()
        result = attack.evaluate(model, records, output_dir=output_dir)
        result.update(
            {
                "attack": "FigStep",
                "split": str(self.eval_cfg.get("split", "forget")),
                "reference_ids_path": self.eval_cfg.get("reference_ids_path", None),
                "config": attack_cfg,
            }
        )
        self._save(output_dir, "FigStep_EVAL.json", result)
        self._save(
            output_dir,
            "FigStep_SUMMARY.json",
            {key: value for key, value in result.items() if key != "details"},
        )
        return result
