"""Evaluators for general-purpose multimodal benchmarks.

The evaluator deliberately keeps dataset loading and model execution generic.
POPE, MM-Vet, MMBench, GQA, and VQAv2 differ mainly in their answer protocol, so each
benchmark is selected through configuration rather than a separate evaluator
implementation.
"""

from __future__ import annotations

import base64
import ast
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from evals.scorers import ExactMatchScorer, RougeScorer


logger = logging.getLogger("evaluator")


class MultimodalBenchmarkEvaluator:
    """Run supported multimodal benchmarks with one normalized evaluation path."""

    name = "MultimodalBenchmark"

    def __init__(self, eval_cfg, **kwargs):
        self.eval_cfg = eval_cfg
        self.name = str(eval_cfg.get("name", self.name))
        self.benchmark = str(eval_cfg.get("benchmark", "custom")).lower()
        self.processor = None
        self._exact_match = ExactMatchScorer()
        self._rouge = RougeScorer(("rougeL",), aggregation="fmeasure")

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
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [value]

    @staticmethod
    def _present(value):
        """Return whether a tabular/JSON value contains usable text."""
        if value is None:
            return False
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return False
        return str(value).strip().casefold() not in {"", "nan", "none"}

    @staticmethod
    def _normalize_text(value):
        text = str(value or "").strip().casefold()
        text = re.sub(r"\s+", " ", text)
        return text.strip(" .,!?:;\n\t")

    @staticmethod
    def _decode_base64(value):
        if value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, base64.binascii.Error):
            return None

    @staticmethod
    def _answer_list(value):
        """Parse an annotation answer list from JSON/TSV representations."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none"}:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [text]

    def _load_image(self, value):
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(BytesIO(value["bytes"])).convert("RGB")
            if value.get("path") is not None:
                value = value["path"]
            else:
                raise ValueError(f"Unsupported image mapping keys: {value.keys()}")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(BytesIO(value)).convert("RGB")
        if not isinstance(value, str):
            raise ValueError(f"Unsupported image value type: {type(value)!r}")
        raw = self._decode_base64(value)
        if raw is not None:
            return Image.open(BytesIO(raw)).convert("RGB")
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = Path(str(self.eval_cfg.get("image_root", "."))) / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Multimodal benchmark image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _read_rows(self):
        if self.benchmark == "gqa":
            return self._read_gqa_rows()
        if self.benchmark == "vqav2":
            return self._read_vqav2_rows()
        path = Path(str(self.eval_cfg.data_path)).expanduser()
        fmt = str(self.eval_cfg.get("format", "auto")).lower()
        if path.is_dir():
            patterns = {"parquet": "*.parquet", "tsv": "*.tsv", "jsonl": "*.jsonl"}
            files = sorted(path.glob(patterns.get(fmt, "*")))
            if not files:
                raise FileNotFoundError(f"No benchmark files found in {path}")
            if fmt == "parquet" or files[0].suffix == ".parquet":
                return pd.concat([pd.read_parquet(file) for file in files]).to_dict("records")
            path = files[0]
        if fmt == "auto":
            fmt = path.suffix.lstrip(".").lower()
        if fmt in {"jsonl", "ndjson"}:
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if fmt == "json":
            text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # Some POPE releases use a .json suffix for JSON Lines.
                payload = [json.loads(line) for line in text.splitlines() if line.strip()]
            if isinstance(payload, dict):
                # MM-Vet annotations are keyed by sample id.
                return [
                    dict(value, id=key) if isinstance(value, dict) else value
                    for key, value in payload.items()
                ]
            return payload
        if fmt in {"tsv", "csv"}:
            max_samples = self.eval_cfg.get("max_samples", None)
            return pd.read_csv(
                path,
                sep="\t" if fmt == "tsv" else ",",
                nrows=int(max_samples) if max_samples is not None else None,
            ).to_dict("records")
        if fmt == "parquet":
            return pd.read_parquet(path).to_dict("records")
        raise ValueError(f"Unsupported multimodal benchmark format: {fmt}")

    def _read_gqa_rows(self):
        """Join GQA questions with the compact image parquet mirror.

        The downloaded GQA mirror stores question rows and image bytes in two
        parquet files. Keeping the bytes in the normalized row lets
        ``_load_image`` handle them without extracting another archive.
        """
        instructions_path = Path(
            str(self.eval_cfg.get("data_path", ""))
        ).expanduser()
        images_path = Path(
            str(self.eval_cfg.get("images_path", ""))
        ).expanduser()
        if not instructions_path.exists():
            raise FileNotFoundError(f"GQA instructions file not found: {instructions_path}")
        if not images_path.exists():
            raise FileNotFoundError(f"GQA images file not found: {images_path}")
        instruction_rows = pd.read_parquet(instructions_path).to_dict("records")
        image_rows = pd.read_parquet(images_path).to_dict("records")
        image_by_id = {str(row["id"]): row["image"] for row in image_rows}
        rows = []
        for row in instruction_rows:
            item = dict(row)
            image_id = str(item.get("imageId", ""))
            if image_id in image_by_id:
                item["image"] = image_by_id[image_id]
            item["answers"] = [item.get("answer", "")]
            types = item.get("types")
            if isinstance(types, dict):
                item["category"] = types.get("structural", types.get("semantic", "all"))
            return_type = item.get("category")
            item["category"] = str(return_type) if self._present(return_type) else "all"
            rows.append(item)
        return rows

    def _read_vqav2_rows(self):
        """Join VQAv2 validation questions and ten human answers."""
        questions_path = Path(str(self.eval_cfg.get("data_path", ""))).expanduser()
        annotations_path = Path(str(self.eval_cfg.get("annotations_path", ""))).expanduser()
        if not questions_path.exists():
            raise FileNotFoundError(f"VQAv2 questions file not found: {questions_path}")
        if not annotations_path.exists():
            raise FileNotFoundError(f"VQAv2 annotations file not found: {annotations_path}")
        questions_payload = json.loads(questions_path.read_text(encoding="utf-8"))
        annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        annotations = {
            str(item["question_id"]): item for item in annotations_payload.get("annotations", [])
        }
        rows = []
        for question in questions_payload.get("questions", []):
            question_id = str(question["question_id"])
            annotation = annotations.get(question_id, {})
            image_id = int(question["image_id"])
            rows.append(
                {
                    "id": question_id,
                    "question": question.get("question", ""),
                    "answer": annotation.get("multiple_choice_answer", ""),
                    "answers": [item.get("answer", "") for item in annotation.get("answers", [])],
                    "answer_type": annotation.get("answer_type", "all"),
                    "image": f"val2014/COCO_val2014_{image_id:012d}.jpg",
                }
            )
        return rows

    @staticmethod
    def _options(row):
        options = row.get("options", row.get("choices", None))
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                options = None
        if isinstance(options, dict):
            return {str(key).upper(): str(value) for key, value in options.items()}
        if isinstance(options, (list, tuple)):
            return {chr(ord("A") + index): str(value) for index, value in enumerate(options)}
        values = {}
        for key in "ABCDEF":
            if MultimodalBenchmarkEvaluator._present(row.get(key)):
                values[key] = str(row[key])
        return values

    def _normalize_row(self, row, index):
        scienceqa_choices = row.get("choices", None)
        scienceqa_answer = row.get("answer", None)
        scienceqa_id = str(row.get("id", row.get("index", index)))
        scienceqa_split = str(self.eval_cfg.get("split", "val"))
        if self.benchmark == "scienceqa":
            # ScienceQA stores the gold answer as a zero-based choice index
            # and stores validation images under ``val/<problem_id>/``.
            if isinstance(scienceqa_choices, str):
                try:
                    scienceqa_choices = json.loads(scienceqa_choices)
                except json.JSONDecodeError:
                    scienceqa_choices = []
            if not isinstance(scienceqa_choices, (list, tuple)):
                scienceqa_choices = []
            try:
                answer_index = int(scienceqa_answer)
            except (TypeError, ValueError):
                answer_index = -1
            if 0 <= answer_index < len(scienceqa_choices):
                row = dict(row)
                row["answer"] = chr(ord("A") + answer_index)
                row["options"] = list(scienceqa_choices)
            if self._present(row.get("image")):
                row = dict(row)
                row["image"] = f"{scienceqa_split}/{scienceqa_id}/{row['image']}"

        question = row.get("question", row.get("Question", row.get("text", "")))
        answer = row.get(
            "answer",
            row.get(
                "Answer",
                row.get(
                    "correct_answer",
                    row.get("label", row.get("most_common_answer", "")),
                ),
            ),
        )
        references = self._answer_list(row.get("answers", None))
        if not self._present(answer) and references:
            answer = references[0]
        image = row.get("image", row.get("image_path", row.get("img", None)))
        if image is None and row.get("image_base64") is not None:
            image = row["image_base64"]
        if not self._present(image):
            image = None
        question = str(question).strip() if self._present(question) else ""
        # ``none`` is a valid natural-language answer in GQA/VQAv2 (for
        # example, questions asking whether an object or text is present), so
        # it must not be treated as a missing tabular value here.
        answer_present = self._present(answer) or (
            self.benchmark in {"gqa", "vqav2"}
            and str(answer).strip().casefold() == "none"
        )
        answer = str(answer).strip() if answer_present else ""
        category = row.get(
            "category",
            row.get(
                "capability",
                row.get("l2-category", row.get("answer_type", "all")),
            ),
        )
        hint = row.get("hint", "")
        return {
            "id": str(row.get("id", row.get("index", index))),
            "question": question,
            "answer": answer,
            "references": references or [answer] if self._present(answer) else [],
            "image": image,
            "hint": str(hint).strip() if self._present(hint) else "",
            "options": self._options(row),
            "category": str(category) if self._present(category) else "all",
        }

    def _records(self):
        rows = self._read_rows()
        records = [self._normalize_row(row, index) for index, row in enumerate(rows)]
        if self.benchmark == "scienceqa":
            split = str(self.eval_cfg.get("split", "val"))
            records = [
                record
                for record, row in zip(records, rows)
                if str(row.get("split", split)).casefold() == split.casefold()
            ]
        max_samples = self.eval_cfg.get("max_samples", None)
        records = [record for record in records if record["question"] and record["answer"]]
        if not records:
            raise ValueError(f"No valid records found for {self.name}")
        if max_samples is not None:
            max_samples = min(int(max_samples), len(records))
            sample_seed = self.eval_cfg.get("sample_seed", None)
            if sample_seed is not None:
                # Sampling happens after normalization/filtering so every
                # selected index is a valid multimodal evaluation example.
                rng = np.random.default_rng(int(sample_seed))
                indices = rng.choice(len(records), size=max_samples, replace=False)
                records = [records[int(index)] for index in indices]
            else:
                # Preserve the historical behavior when no seed is supplied.
                records = records[:max_samples]
        return records

    def _load_processor(self):
        if self.processor is None:
            processor_path = self.eval_cfg.get("processor_path", None)
            if not processor_path:
                raise ValueError(f"{self.name} requires processor_path")
            self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    def _prompt(self, record):
        question = record["question"]
        if record.get("hint"):
            question = f"{record['hint']}\n{question}"
        if self.benchmark == "mmbench" or record["options"]:
            options = "\n".join(f"{key}: {value}" for key, value in record["options"].items())
            question = f"{question}\n{options}\nAnswer with one option letter only."
        else:
            answer_instruction = (
                "Answer with a short answer only."
                if self.benchmark in {"gqa", "vqav2"}
                else "Answer briefly and directly."
            )
            question = f"{question}\n{answer_instruction}"
        image = record["image"]
        style = str(self.eval_cfg.get("prompt_style", "auto")).lower()
        if style == "auto":
            model_family = str(self.eval_cfg.get("model_family", "")).lower()
            model_path = str(self.eval_cfg.get("processor_path", "")).lower()
            style = "chat_template" if "qwen" in f"{model_family} {model_path}" else "raw"
        if style == "chat_template":
            processor = self._load_processor()
            content = []
            if image is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": content}]
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ), image
        prefix = "USER: <image>\n" if image is not None else "USER: "
        return f"{prefix}{question}\nASSISTANT:", image

    def _generate(self, model, tokenizer, record):
        processor = self._load_processor()
        prompt, image_value = self._prompt(record)
        images = self._load_image(image_value) if image_value is not None else None
        kwargs = {"text": prompt, "return_tensors": "pt"}
        if images is not None:
            # Match the existing MLLMU processor contract: one image per
            # example, with Gemma's processor requiring an extra nesting.
            kwargs["images"] = (
                [[images]]
                if processor.__class__.__name__ == "Gemma3Processor"
                else [images]
            )
        inputs = processor(**kwargs)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_args = self._to_container(self.eval_cfg.get("generation_args", {}), {}) or {}
        with __import__("torch").no_grad():
            outputs = model.generate(**inputs, **generation_args)
        input_length = inputs["input_ids"].shape[-1]
        sequences = getattr(outputs, "sequences", outputs)
        generated_ids = (
            sequences[:, input_length:]
            if sequences.shape[-1] > input_length
            else sequences
        )
        decoder = processor.decode if hasattr(processor, "decode") else tokenizer.decode
        text = decoder(generated_ids[0], skip_special_tokens=True).strip()
        for marker in ("ASSISTANT:", "Assistant:", "Answer:"):
            if marker in text:
                text = text.split(marker, 1)[1].strip()
        return text

    @staticmethod
    def _choice(answer, options):
        text = str(answer).strip().upper()
        valid = set(options)
        match = re.search(r"\b([A-F])\b", text)
        if match and match.group(1) in valid:
            return match.group(1)
        return text[:1] if text[:1] in valid else None

    @staticmethod
    def _mmvet_references(answer):
        """Expand MM-Vet's ``<OR>``/``<AND>`` answer convention."""
        return [part.strip() for part in str(answer).split("<OR>") if part.strip()]

    @classmethod
    def _mmvet_match(cls, prediction, answer):
        prediction_text = cls._normalize_text(prediction)
        for alternative in cls._mmvet_references(answer):
            required = [
                cls._normalize_text(part)
                for part in alternative.split("<AND>")
                if cls._normalize_text(part)
            ]
            if required and all(part in prediction_text for part in required):
                return 1.0
        return 0.0

    @staticmethod
    def _vqa_normalize(value):
        text = str(value or "").casefold().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        digit_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        tokens = [
            digit_map.get(token, token)
            for token in text.split()
            if token not in {"a", "an", "the"}
        ]
        return " ".join(tokens)

    @classmethod
    def _vizwiz_score(cls, prediction, references):
        normalized_prediction = cls._vqa_normalize(prediction)
        normalized_references = [cls._vqa_normalize(item) for item in references]
        if not normalized_references:
            return 0.0
        annotator_scores = []
        for index in range(len(normalized_references)):
            matches = sum(
                normalized_prediction == reference
                for other_index, reference in enumerate(normalized_references)
                if other_index != index
            )
            annotator_scores.append(min(1.0, matches / 3.0))
        return float(np.mean(annotator_scores))

    def _score_record(self, record, generated):
        if self.benchmark == "pope":
            prediction = self._normalize_text(generated)
            target = self._normalize_text(record["answer"])
            prediction_match = re.search(r"\b(yes|no)\b", prediction)
            target_match = re.search(r"\b(yes|no)\b", target)
            prediction = prediction_match.group(1) if prediction_match else prediction
            target = target_match.group(1) if target_match else target
            return float(prediction == target), prediction, target
        if record["options"] or self.benchmark == "mmbench":
            prediction = self._choice(generated, record["options"])
            target = self._choice(record["answer"], record["options"]) or record["answer"].upper()
            return float(prediction == target), prediction, target
        if self.benchmark == "mmvet":
            return self._mmvet_match(generated, record["answer"]), generated, record["answer"]
        if self.benchmark == "vizwiz":
            return (
                self._vizwiz_score(generated, record["references"]),
                generated,
                record["answer"],
            )
        if self.benchmark == "vqav2":
            return (
                self._vizwiz_score(generated, record["references"]),
                generated,
                record["answer"],
            )
        if self.benchmark == "gqa":
            prediction = self._normalize_text(generated)
            target = self._normalize_text(record["answer"])
            return float(prediction == target), prediction, target
        exact = self._exact_match.score(generated, record["answer"]).value
        return float(exact), generated, record["answer"]

    def _summary(self, records, details):
        scores = [item["score"] for item in details]
        result = {"count": len(details), "accuracy": float(np.mean(scores)) if scores else 0.0}
        categories = sorted({item["category"] for item in details})
        result["category_accuracy"] = {
            category: float(
                np.mean(
                    [item["score"] for item in details if item["category"] == category]
                )
            )
            for category in categories
        }
        if self.benchmark == "pope":
            tp = sum(item["prediction"] == "yes" and item["target"] == "yes" for item in details)
            fp = sum(item["prediction"] == "yes" and item["target"] == "no" for item in details)
            fn = sum(item["prediction"] == "no" and item["target"] == "yes" for item in details)
            tn = sum(item["prediction"] == "no" and item["target"] == "no" for item in details)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            result.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    "yes_rate": (tp + fp) / len(details) if details else 0.0,
                    "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                }
            )
        if self.benchmark == "mmvet":
            result["rougeL"] = (
                float(np.mean([item["rougeL"] for item in details]))
                if details
                else 0.0
            )
        if self.benchmark in {"vizwiz", "vqav2"}:
            result["vqa_accuracy"] = result["accuracy"]
        return result

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        output_dir = Path(output_dir or self.eval_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.name}_EVAL.json"
        if output_path.exists() and not (overwrite or self.eval_cfg.get("overwrite", False)):
            return json.loads(output_path.read_text(encoding="utf-8"))
        tokenizer = kwargs.get("tokenizer")
        if hasattr(model, "eval"):
            model.eval()
        records = self._records()
        details = []
        for index, record in enumerate(records):
            generated = self._generate(model, tokenizer, record)
            score, prediction, target = self._score_record(record, generated)
            item = {
                "id": record["id"],
                "category": record["category"],
                "question": record["question"],
                "generated_answer": generated,
                "target": target,
                "prediction": prediction,
                "score": score,
            }
            if self.benchmark in {"vizwiz", "vqav2"}:
                item["references"] = record["references"]
            if self.benchmark == "mmvet":
                rouge_scores = [
                    self._rouge.score(generated, reference).value
                    for reference in self._mmvet_references(record["answer"])
                ]
                item["rougeL"] = max(rouge_scores, default=0.0)
            details.append(item)
            if (index + 1) % 100 == 0:
                logger.info("%s evaluated %d/%d records", self.name, index + 1, len(records))
        result = self._summary(records, details)
        result.update({"benchmark": self.name, "details": details})
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = output_dir / f"{self.name}_SUMMARY.json"
        summary_path.write_text(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"details", "confusion_matrix"}
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result
