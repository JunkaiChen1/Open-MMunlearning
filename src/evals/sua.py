"""SUA universal visual perturbation evaluator."""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from attacks.sua import SUAConfig, SUAUniversalPerturbationAttack


logger = logging.getLogger("evaluator")


class SUAAttackEvaluator:
    """Train/evaluate the SUA universal perturbation against a loaded model.

    This evaluator is deliberately separate from the ordinary MLLMU benchmark
    evaluator because fitting a universal perturbation is an expensive attack,
    not a routine per-split metric. It can be enabled as a second evaluator in
    an evaluation experiment and writes its own ``SUA_EVAL.json`` files.
    """

    name = "SUA"

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

    def _read_parquet(self, path):
        path = Path(path).expanduser()
        files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No parquet files found at {path}")
        return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

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
        raise TypeError(f"Unsupported MLLMU image type: {type(value)!r}")

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

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
        if split.startswith("forget_"):
            ratio = split.removeprefix("forget_")
        else:
            ratio = int(self.eval_cfg.get("ratio", 5))
        return root / f"forget_{ratio}" / "train-00000-of-00001.parquet"

    def _records(self):
        dataframe = self._read_parquet(self._data_path())
        split = str(self.eval_cfg.get("split", "forget"))
        image_strategy = str(self.eval_cfg.get("image_strategy", "all"))
        max_records = self.eval_cfg.get("max_records", None)
        records = []
        for _, row in dataframe.iterrows():
            row_id = str(row.get("ID", len(records)))
            image_values = row.get("images") if split == "test" else row.get("image")
            image_values = self._as_list(image_values)
            if image_strategy == "first":
                image_values = image_values[:1]
            elif image_strategy == "random" and image_values:
                image_values = [image_values[0]]
            tasks = self._as_list(row.get("Mask_Task", []))
            for image_index, image_value in enumerate(image_values):
                for task_index, task in enumerate(tasks):
                    if (
                        not isinstance(task, dict)
                        or task.get("Type") != "Image_Textual"
                    ):
                        continue
                    question = str(task.get("Question", "")).replace("__", "[Blank]")
                    question += "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
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
                raise ValueError("eval.sua.processor_path must be set.")
            self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    def _save(self, output_dir, result, suffix="EVAL"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.name}_{suffix}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return path

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = (
            self.eval_cfg.get("overwrite", False) if overwrite is None else overwrite
        )
        output_dir = output_dir or self.eval_cfg.output_dir
        eval_path = Path(output_dir) / f"{self.name}_EVAL.json"
        if eval_path.exists() and not overwrite:
            with eval_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        tokenizer = kwargs.get("tokenizer", None)
        processor = self._load_processor()
        sua_cfg = self._to_container(self.eval_cfg.get("sua", {}), {})
        sua_cfg = dict(sua_cfg or {})
        checkpoint_path = sua_cfg.get("checkpoint_path") or str(
            Path(output_dir) / "checkpoint" / "sua_perturbation.pt"
        )
        sua_cfg["checkpoint_path"] = checkpoint_path
        config = SUAConfig.from_mapping(sua_cfg)
        attack = SUAUniversalPerturbationAttack(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            config=config,
        )
        records = self._records()
        if Path(checkpoint_path).exists() and not bool(
            self.eval_cfg.get("refit", False)
        ):
            logger.info("Loading existing SUA perturbation from %s", checkpoint_path)
            attack.load(checkpoint_path)
        else:
            logger.info("Fitting SUA perturbation on %s records", len(records))
            attack.fit(records)
        result = attack.evaluate(records)
        result.update(
            {
                "attack": "SUA",
                "split": str(self.eval_cfg.get("split", "forget")),
                "checkpoint_path": str(checkpoint_path),
                "config": sua_cfg,
            }
        )
        self._save(output_dir, result, suffix="EVAL")
        summary = {key: value for key, value in result.items() if key != "details"}
        self._save(output_dir, summary, suffix="SUMMARY")
        return summary
