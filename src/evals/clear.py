import json
import logging
import math
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from PIL import Image
from transformers import AutoProcessor

from evals.base import Evaluator
from evals.scorers import (
    AnswerProbabilityScorer,
    ContrastiveProbabilityScorer,
    JensenShannonDistanceScorer,
    LossMIAScorer,
    PerturbationTruthRatioScorer,
    RougeScorer,
    StandardMinKPlusPlusScorer,
    StandardMinKScorer,
    TruthRatioTransformScorer,
    TwoSampleKSScorer,
    ZlibMIAScorer,
    token_log_probability_statistics,
)


logger = logging.getLogger("evaluator")
IGNORE_INDEX = -100


FILE_TO_TASK = {
    "eval_real_faces_wo_options.json": "Real Faces",
    "eval_real_world_wo_options.json": "Real World",
    "eval_log.json": "Retain",
    "eval_retain_facerec.json": "Retain FaceRec",
    "eval_log_forget.json": "Forget",
    "eval_forget_facerec.json": "Forget FaceRec",
    "eval_text_forget.json": "Pure Text Forget",
    "eval_text_retain.json": "Pure Text Retain",
}


# Kept in the source order from CLEAR/mm/dataset.py.  The random caption
# question is part of the input protocol, not a presentation detail.
CAPTION_QUESTIONS = (
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
)


class CLEAREvaluator(Evaluator):
    """Evaluator for CLEAR's multimodal unlearning protocol.

    The original CLEAR evaluation writes one JSON file per task plus an
    `eval_log_aggregated.json` file. This evaluator keeps that file contract
    while running through OpenUnlearning's shared model loading entrypoint.
    """

    def __init__(self, eval_cfg, **kwargs):
        self.name = "CLEAR"
        self.eval_cfg = eval_cfg
        self.processor = None
        self._records_cache = {}
        self._image_cache = {}
        self.rng = random.Random(int(self.eval_cfg.get("seed", 0)))
        self._answer_probability_scorer = AnswerProbabilityScorer()
        self._contrastive_probability_scorer = ContrastiveProbabilityScorer()
        self._truth_ratio_scorer = PerturbationTruthRatioScorer()
        self._truth_ratio_transform_scorer = TruthRatioTransformScorer()
        mia_sign = float(self.eval_cfg.get("generation_mia_sign", 1.0))
        self._loss_mia_scorer = LossMIAScorer(sign=mia_sign)
        self._zlib_mia_scorer = ZlibMIAScorer(sign=mia_sign)
        min_k_ratio = float(self.eval_cfg.get("generation_mia_min_k_ratio", 0.2))
        self._standard_min_k_scorer = StandardMinKScorer(ratio=min_k_ratio)
        self._standard_min_k_plus_plus_scorer = StandardMinKPlusPlusScorer(
            ratio=min_k_ratio
        )
        self._ks_scorer = TwoSampleKSScorer()
        self._js_scorer = JensenShannonDistanceScorer(
            alignment="auto",
            bins=int(self.eval_cfg.get("js_bins", 50)),
        )
        self._rouge_scorer = RougeScorer(("rouge1", "rougeL"), aggregation="recall")
        self._active_prompt_style = None

    def prepare_model(self, model):
        model.eval()
        return model

    def _to_container(self, value, default=None):
        if value is None:
            return default
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _as_list(self, value):
        value = self._to_container(value, value)
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _value_at(self, value, index, default=None):
        value = self._to_container(value, default)
        if isinstance(value, list):
            if not value:
                return default
            if index < len(value):
                return value[index]
            return value[-1]
        return value if value is not None else default

    def _load_processor(self):
        if self.processor is not None:
            return self.processor
        processor_path = self.eval_cfg.get("processor_path", None)
        if not processor_path:
            raise ValueError("eval.clear.processor_path must be set.")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        if self.eval_cfg.get("padding_side", None) is not None:
            self.processor.tokenizer.padding_side = self.eval_cfg.padding_side
        if getattr(self.processor.tokenizer, "pad_token", None) is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        if self.eval_cfg.get("processor_do_pad", True):
            setattr(self.processor, "do_pad", True)
        return self.processor

    def _model_family(self, model):
        configured = self.eval_cfg.get("model_family", None)
        if configured:
            return str(configured).lower()
        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        class_name = model.__class__.__name__.lower()
        text = f"{model_type} {class_name}"
        if "llava" in text:
            return "llava"
        if "qwen" in text:
            return "qwen"
        if "gemma" in text:
            return "gemma"
        if "mllama" in text or "llama-3.2" in text:
            return "llama-3.2"
        return "default"

    def _effective_prompt_style(self, model):
        style = str(self.eval_cfg.get("prompt_style", "auto")).lower()
        if style != "auto":
            return style
        family = self._model_family(model)
        if "llava" in family:
            return "raw"
        return "chat_template"

    def _model_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_to_device(self, inputs, device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _read_parquet(self, path):
        import pyarrow.parquet as pq

        path = Path(path)
        parquet_files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        rows = []
        for file_path in parquet_files:
            rows.extend(pq.read_table(file_path).to_pylist())
        return rows

    def _read_json_or_jsonl(self, path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            with open(path, "r", encoding="utf-8") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "train", "records", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
            return list(data.values())
        raise ValueError(f"Unsupported JSON data in {path}: {type(data)}")

    def _load_hf_dataset(self, dataset_path, split):
        from datasets import load_dataset

        if split is None:
            raise ValueError(
                f"split must be set when loading HuggingFace dataset {dataset_path}"
            )
        dataset = load_dataset(
            dataset_path,
            split=split,
            trust_remote_code=bool(self.eval_cfg.get("dataset_trust_remote_code", False)),
        )
        return [dict(row) for row in dataset]

    def _load_records(self, data_path, split=None):
        if data_path in (None, "", "null", "None"):
            return []
        data_path = str(data_path)
        cache_key = f"{data_path}::{split}"
        if cache_key in self._records_cache:
            return self._records_cache[cache_key]

        path = Path(data_path)
        if path.exists():
            if path.is_dir() or path.suffix == ".parquet":
                records = self._read_parquet(path)
            elif path.suffix in (".json", ".jsonl"):
                records = self._read_json_or_jsonl(path)
            else:
                raise ValueError(f"Unsupported CLEAR data file: {data_path}")
        else:
            records = self._load_hf_dataset(data_path, split)

        self._records_cache[cache_key] = records
        return records

    def _mapping_get(self, mapping, key, default=None):
        if not isinstance(mapping, dict) or key is None:
            return default
        if key in mapping:
            return mapping[key]
        key_lower = str(key).lower()
        for existing_key, value in mapping.items():
            if str(existing_key).lower() == key_lower:
                return value
        return default

    def _get_field(self, qa, row, key, default=None):
        for candidate in self._as_list(key):
            value = self._mapping_get(qa, candidate, None)
            if value is not None:
                return value
            value = self._mapping_get(row, candidate, None)
            if value is not None:
                return value
        return default

    def _format_text(self, value):
        if value is None:
            return ""
        text = str(value).strip()
        if self.eval_cfg.get("capitalize_text", False):
            text = text.capitalize()
        return text

    def _format_answer_values(self, raw_value):
        values = []
        for item in self._as_list(raw_value):
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        return values

    def _question_for_qa(self, qa, row, spec):
        strategy = str(spec.get("question_strategy", "column")).lower()
        if strategy == "random_faces":
            return str(
                spec.get(
                    "face_question",
                    self.eval_cfg.get(
                        "face_question", "The name of the person on the image is "
                    ),
                )
            )
        if strategy == "random_caption":
            questions = self._as_list(
                spec.get(
                    "caption_questions",
                    self.eval_cfg.get(
                        "caption_questions",
                        CAPTION_QUESTIONS,
                    ),
                )
            )
            return str(self.rng.choice(questions)) if questions else "Describe this image."

        question_key = spec.get(
            "question_key", self.eval_cfg.get("metadata_question_key", "question")
        )
        question = self._get_field(qa, row, question_key)
        if question is None and self.eval_cfg.get("fallback_question_keys", True):
            question = self._get_field(
                qa,
                row,
                [
                    "question",
                    "Question",
                    self.eval_cfg.get("metadata_question_key", "Question"),
                ],
            )
        return self._format_text(question)

    def _question_for_sample(self, sample):
        """Draw a caption question at the same granularity as CLEAR's dataset."""
        if sample.get("question_strategy") == "random_caption":
            questions = sample.get("caption_questions", CAPTION_QUESTIONS)
            return str(self.rng.choice(questions)) if questions else "Describe this image."
        return str(sample.get("question", ""))

    def _answer_values(self, sample, key, fallback_keys=None):
        row = sample["row"]
        qa = sample["qa"]
        values = self._format_answer_values(self._get_field(qa, row, key))
        if values:
            return values
        for fallback_key in self._as_list(fallback_keys):
            values = self._format_answer_values(
                self._get_field(qa, row, fallback_key)
            )
            if values:
                return values
        return []

    def _image_value_for_row(self, row, spec):
        image_key = spec.get("image_key", self.eval_cfg.get("image_key", "image"))
        image_value = self._mapping_get(row, image_key, None)
        if image_value is None:
            image_value = self._mapping_get(row, "image_path", None)
        if image_value is None:
            images_value = self._mapping_get(row, "images", None)
            images = self._as_list(images_value)
            image_value = images[0] if images else None
        return image_value

    def _parse_metadata(self, metadata_value):
        if metadata_value in (None, "", "null", "None"):
            return []
        if isinstance(metadata_value, str):
            parsed = json.loads(metadata_value)
        else:
            parsed = self._to_container(metadata_value, metadata_value)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []

    def _explode_row(self, row, row_index, data_path, spec):
        metadata_key = spec.get(
            "metadata_key", self.eval_cfg.get("metadata_key", "metadata")
        )
        strategy = str(spec.get("question_strategy", "column")).lower()
        if strategy == "random_faces":
            qas = [row]
        else:
            metadata_value = self._mapping_get(row, metadata_key, None)
            qas = self._parse_metadata(metadata_value) if metadata_value is not None else []
            if not qas:
                qas = [row]

        samples = []
        image_value = self._image_value_for_row(row, spec)
        id_key = spec.get("id_key", self.eval_cfg.get("id_key", "ID"))
        name_key = spec.get("name_key", self.eval_cfg.get("name_key", "name"))
        for qa_index, qa in enumerate(qas):
            if not isinstance(qa, dict):
                continue
            if strategy == "random_caption":
                # The upstream Dataset samples this independently on every
                # __getitem__ call; defer the draw until a model input is made.
                question = None
                caption_questions = self._as_list(
                    spec.get(
                        "caption_questions",
                        self.eval_cfg.get("caption_questions", CAPTION_QUESTIONS),
                    )
                )
            else:
                question = self._question_for_qa(qa, row, spec)
                caption_questions = None
            if not question and strategy != "random_caption":
                logger.warning("Skipping CLEAR row %s qa %s: empty question.", row_index, qa_index)
                continue
            person_id = self._get_field(qa, row, id_key, default=row_index)
            samples.append(
                {
                    "index": len(samples),
                    "row_index": row_index,
                    "qa_index": qa_index,
                    "person_id": str(person_id),
                    "name": self._get_field(qa, row, name_key, default=""),
                    "question": question,
                    "question_strategy": strategy,
                    "caption_questions": caption_questions,
                    "category": spec.get("category", str(spec.get("eval_task", "clear"))),
                    "qa": qa,
                    "row": row,
                    "image_value": image_value,
                    "image_cache_key": f"{data_path}:{row_index}",
                    "data_path": data_path,
                }
            )
        return samples

    def _sample_answer_pool(self, samples, answer_key):
        pool = []
        for sample in samples:
            answers = self._answer_values(
                sample,
                answer_key,
                fallback_keys=[
                    self.eval_cfg.get("metadata_answer_key", "Answer"),
                    "answer",
                    "name",
                ],
            )
            if answers:
                pool.append((sample["person_id"], answers[0]))
        return pool

    def _attach_auto_perturbations(self, samples, spec):
        if not self.eval_cfg.get("auto_perturbations", True):
            return
        perturb_key = spec.get("perturbed_answer_key", "perturbed_answers")
        answer_key = spec.get("answer_key", self.eval_cfg.get("metadata_answer_key", "answer"))
        pool = self._sample_answer_pool(samples, answer_key)
        num_perturbations = int(self.eval_cfg.get("num_auto_perturbations", 5))
        for sample in samples:
            existing = self._answer_values(sample, perturb_key)
            if existing:
                continue
            candidates = [
                answer
                for person_id, answer in pool
                if person_id != sample["person_id"] and answer
            ]
            if not candidates:
                continue
            self.rng.shuffle(candidates)
            sample["auto_perturbed_answers"] = candidates[:num_perturbations]

    def _samples_for_task(self, spec):
        data_path = spec.get("data_path", None)
        split = spec.get("split", None)
        records = self._load_records(data_path, split)
        # The CLEAR +tofu files mix image rows and QA-only rows.  Filter before
        # applying the source-row cap so a text task does not consume its
        # budget on the image prefix of a mixed file.
        image_mode = str(spec.get("image_mode", "any")).lower()
        if image_mode in ("image", "text"):
            want_image = image_mode == "image"
            records = [
                row
                for row in records
                if isinstance(row, dict)
                and ((self._image_value_for_row(row, spec) is not None) == want_image)
            ]
        # CLEAR's ``ds_size`` slices the source dataset before a list-valued
        # caption field is expanded by the collator.
        max_rows = spec.get("max_rows", self.eval_cfg.get("max_rows_per_task", None))
        if max_rows is None and self.eval_cfg.get("upstream_compatibility", True):
            max_rows = spec.get(
                "max_samples", self.eval_cfg.get("max_samples_per_task", None)
            )
        if max_rows is not None:
            records = records[: int(max_rows)]
        samples = []
        for row_index, row in enumerate(records):
            if not isinstance(row, dict):
                continue
            samples.extend(self._explode_row(row, row_index, data_path, spec))

        self._attach_auto_perturbations(samples, spec)
        max_samples = spec.get("max_samples", None)
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        for index, sample in enumerate(samples):
            sample["index"] = index
        return samples

    def _resolve_image_path(self, image_path, data_path):
        raw = str(image_path)
        if os.path.isabs(raw) and os.path.exists(raw):
            return raw

        image_root = Path(self.eval_cfg.get("image_root", self.eval_cfg.get("data_root", ".")))
        data_parent = Path(str(data_path)).parent
        clean = raw.replace("\\", "/").lstrip("./")
        candidates = [
            image_root / raw,
            image_root / clean,
            data_parent / raw,
            data_parent / clean,
            Path.cwd() / raw,
            Path.cwd() / clean,
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not resolve CLEAR image {image_path}. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _resize_image(self, image):
        max_pixels = self.eval_cfg.get("image_max_pixels", None)
        if max_pixels is None:
            return image
        width, height = image.size
        pixels = width * height
        if pixels <= int(max_pixels):
            return image
        scale = (int(max_pixels) / pixels) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _load_image_value(self, image_value, data_path):
        image_value = self._to_container(image_value, image_value)
        if image_value is None:
            return None
        if isinstance(image_value, Image.Image):
            image = image_value.convert("RGB")
        elif isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                image = Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            elif image_value.get("path") is not None:
                image_path = self._resolve_image_path(image_value["path"], data_path)
                image = Image.open(image_path).convert("RGB")
            else:
                raise ValueError(f"Unsupported CLEAR image dict keys: {image_value.keys()}")
        elif isinstance(image_value, (bytes, bytearray)):
            image = Image.open(BytesIO(image_value)).convert("RGB")
        elif isinstance(image_value, str):
            image_path = self._resolve_image_path(image_value, data_path)
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError(f"Unsupported CLEAR image value: {type(image_value)}")
        return self._resize_image(image)

    def _load_sample_image(self, sample):
        cache_key = sample.get("image_cache_key")
        if (
            self.eval_cfg.get("cache_images", True)
            and cache_key
            and cache_key in self._image_cache
        ):
            return self._image_cache[cache_key]
        image = self._load_image_value(sample.get("image_value"), sample.get("data_path"))
        if self.eval_cfg.get("cache_images", True) and cache_key and image is not None:
            self._image_cache[cache_key] = image
        return image

    def _processor_images(self, images):
        processor = self._load_processor()
        if processor.__class__.__name__ == "Gemma3Processor":
            return [[image] for image in images]
        return images

    def _chat_template_text(self, question, image, answer=None):
        processor = self._load_processor()
        messages = []
        system_prompt = self.eval_cfg.get("system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = []
        if image is not None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer)}],
                }
            )
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )

    def _raw_text(self, question, image, answer=None):
        system_prompt = self.eval_cfg.get(
            "raw_system_prompt",
            "A chat between a curious human and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions. ",
        )
        question_start = self.eval_cfg.get("question_start_tag", "USER: ")
        answer_tag = self.eval_cfg.get("answer_tag", " ASSISTANT: ")
        image_token = self.eval_cfg.get("image_token", "<image>")
        image_part = f"{image_token}\n" if image is not None else ""
        text = f"{system_prompt}{question_start}{image_part}{question}{answer_tag}"
        if answer is not None:
            text += str(answer)
        return text

    def _build_text_and_images(self, sample, model, answer=None, question=None):
        image = self._load_sample_image(sample)
        question = self._question_for_sample(sample) if question is None else str(question)
        style = self._effective_prompt_style(model)
        if style == "chat_template":
            text = self._chat_template_text(question, image, answer=answer)
        elif style == "raw":
            text = self._raw_text(question, image, answer=answer)
        else:
            raise ValueError(f"Unknown CLEAR prompt_style: {style}")
        return text, [image] if image is not None else []

    def _prepare_inputs(self, text, images, tokenizer):
        processor = self._load_processor()
        kwargs = {"text": text, "return_tensors": "pt"}
        max_length = self.eval_cfg.get("max_length", None)
        if max_length is not None:
            kwargs.update(
                {
                    "max_length": int(max_length),
                    "truncation": self.eval_cfg.get("truncation", True),
                }
            )
        if images:
            kwargs["images"] = self._processor_images(images)
        try:
            return processor(**kwargs)
        except Exception:
            if images:
                raise
            tokenizer_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("text", "images", "return_tensors")
            }
            return tokenizer(text, return_tensors="pt", **tokenizer_kwargs)

    def _attention_mask(self, batch, tokenizer):
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            return attention_mask
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones_like(batch["input_ids"])
        return (batch["input_ids"] != pad_token_id).long()

    def _make_labels(self, full_batch, prompt_batch, tokenizer, prompt_text=None):
        if (
            self.eval_cfg.get("upstream_compatibility", True)
            and prompt_text is not None
        ):
            # Match mm_data_collator_preprocessor: count the formatted question
            # with tokenizer.tokenize, then mask its text-token prefix.
            num_question_tokens = len(
                tokenizer.tokenize(prompt_text, add_special_tokens=True)
            )
            prompt_attention = self._attention_mask(prompt_batch, tokenizer)
            # Modern multimodal processors replace one textual image marker
            # with hundreds of visual-token positions.  In that case the
            # tokenizer-level count above cannot describe the actual prompt
            # span; use the processor batch below instead.
            if all(
                int(length.item()) == num_question_tokens
                for length in prompt_attention.sum(dim=1)
            ):
                labels = full_batch["input_ids"].clone()
                for label in labels:
                    non_pad_tokens = (
                        label != tokenizer.pad_token_id
                    ).nonzero(as_tuple=True)[0]
                    if len(non_pad_tokens) == 0:
                        label[:] = IGNORE_INDEX
                    elif tokenizer.padding_side == "left":
                        label[: non_pad_tokens[0] + num_question_tokens] = IGNORE_INDEX
                    else:
                        label[
                            non_pad_tokens[0] : non_pad_tokens[0] + num_question_tokens
                        ] = IGNORE_INDEX
                        label[non_pad_tokens[-1] + 1 :] = IGNORE_INDEX
                return labels
        labels = full_batch["input_ids"].clone()
        prompt_lengths = self._attention_mask(prompt_batch, tokenizer).sum(dim=1)
        padding_side = getattr(tokenizer, "padding_side", "right")
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            if padding_side == "left":
                full_len = full_batch["attention_mask"][row_idx].sum().item()
                prompt_start = max(0, labels.shape[1] - full_len)
                prompt_end = min(prompt_start + prompt_len, labels.shape[1])
                labels[row_idx, prompt_start:prompt_end] = IGNORE_INDEX
            else:
                labels[row_idx, : min(prompt_len, labels.shape[1])] = IGNORE_INDEX
        labels[full_batch["attention_mask"] == 0] = IGNORE_INDEX
        return labels

    def _batch_loss(self, logits, labels):
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX, reduction="none"
        )
        return loss_function(shifted_logits.transpose(-1, -2), shifted_labels).sum(-1)

    def _remove_image_tokens(self, input_ids, logits, model):
        """Align expanded VLM logits with text-space labels as CLEAR does."""
        config = getattr(model, "config", None)
        image_token_ids = {
            value
            for value in (
                getattr(config, "image_token_id", None),
                getattr(config, "image_token_index", None),
            )
            if value is not None
        }
        if not image_token_ids:
            return logits
        batch_size, sequence_length = input_ids.shape
        aligned_logits = []
        for row_index in range(batch_size):
            positions = torch.empty(0, dtype=torch.long, device=input_ids.device)
            for image_token_id in image_token_ids:
                positions = (input_ids[row_index] == image_token_id).nonzero(
                    as_tuple=True
                )[0]
                if len(positions):
                    break
            if len(positions) == 0:
                aligned = logits[row_index, -sequence_length:, :]
            else:
                start = int(positions[0].item())
                right_length = sequence_length - start
                aligned = torch.cat(
                    [
                        logits[row_index, :start, :],
                        logits[row_index, -right_length:, :],
                    ],
                    dim=0,
                )
            aligned_logits.append(aligned)
        return torch.stack(aligned_logits)

    def _answer_loss(
        self,
        model,
        tokenizer,
        sample,
        answer,
        question=None,
        *,
        compute_token_stats=False,
    ):
        question = self._question_for_sample(sample) if question is None else question
        full_text, images = self._build_text_and_images(
            sample, model, answer=answer, question=question
        )
        prompt_text, prompt_images = self._build_text_and_images(
            sample, model, answer=None, question=question
        )
        full_batch = self._prepare_inputs(full_text, images, tokenizer)
        prompt_batch = self._prepare_inputs(prompt_text, prompt_images, tokenizer)
        labels = self._make_labels(
            full_batch, prompt_batch, tokenizer, prompt_text=prompt_text
        )
        num_tokens = (labels != IGNORE_INDEX).sum(-1)
        if num_tokens.sum().item() == 0:
            return None

        device = self._model_device(model)
        full_batch = self._move_to_device(full_batch, device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(**full_batch, labels=labels)

        logits = self._remove_image_tokens(full_batch["input_ids"], outputs.logits, model)
        loss = self._batch_loss(logits, labels)
        result = {
            "prompt_text": prompt_text,
            "loss": float(loss[0].detach().cpu().item()),
            "avg_loss": float(
                (loss[0] / num_tokens[0].to(loss.device)).detach().cpu().item()
            ),
            "num_tokens": int(num_tokens[0].detach().cpu().item()),
        }
        if compute_token_stats:
            result.update(token_log_probability_statistics(logits, labels))
        return result

    def _clean_answer(self, text):
        text = str(text)
        for marker in ("ASSISTANT:", "Assistant:", "assistant:", "Answer:", "### Answer:"):
            if marker in text:
                text = text.split(marker)[-1]
        text = text.strip()
        if self.eval_cfg.get("truncate_generation_at_period", False):
            period_index = text.find(".")
            if period_index != -1:
                text = text[: period_index + 1]
        return text.strip()

    def _generate(self, model, tokenizer, sample, answer=None, question=None):
        question = self._question_for_sample(sample) if question is None else question
        if (
            self.eval_cfg.get("upstream_compatibility", True)
            and self.eval_cfg.get("upstream_text_only_generation", True)
            and answer is not None
        ):
            # CLEAR/mm/eval.py decodes the labelled multimodal input, strips
            # the answer, then calls generate on text tokens only.
            full_text, images = self._build_text_and_images(
                sample, model, answer=answer, question=question
            )
            full_batch = self._prepare_inputs(full_text, images, tokenizer)
            processor = self._load_processor()
            decoder = (
                processor.batch_decode
                if hasattr(processor, "batch_decode")
                else tokenizer.batch_decode
            )
            decoded_input = decoder(
                full_batch["input_ids"], skip_special_tokens=True
            )[0]
            question = decoded_input[: decoded_input.rfind(str(answer))]
            generation_inputs = tokenizer.batch_encode_plus(
                [question], add_special_tokens=True, return_tensors="pt", padding=True
            )
            generation_inputs = self._move_to_device(
                generation_inputs, self._model_device(model)
            )
            generation_args = self._to_container(
                self.eval_cfg.get("generation_args", {}), {}
            )
            generation_args.update(
                {
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            )
            with torch.no_grad():
                outputs = model.generate(**generation_inputs, **generation_args)
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            generated_ids = sequences[:, generation_inputs["input_ids"].shape[-1] :]
            generated = decoder(generated_ids, skip_special_tokens=True)[0]
            return question, generated

        prompt, images = self._build_text_and_images(
            sample, model, answer=None, question=question
        )
        inputs = self._prepare_inputs(prompt, images, tokenizer)
        device = self._model_device(model)
        inputs = self._move_to_device(inputs, device)
        input_len = inputs["input_ids"].shape[-1]
        generation_args = self._to_container(
            self.eval_cfg.get("generation_args", {}), {}
        )
        if (
            "pad_token_id" not in generation_args
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            generation_args["pad_token_id"] = tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = (
            sequences[:, input_len:] if sequences.shape[-1] > input_len else sequences
        )
        processor = self._load_processor()
        decoder = processor.decode if hasattr(processor, "decode") else tokenizer.decode
        generated = decoder(generated_ids[0], skip_special_tokens=True)
        return prompt, self._clean_answer(generated)

    def _rouge_recall(self, generated_outputs, ground_truths, indices):
        rouge1_recall = {}
        rouge_l_recall = {}
        for generated, ground_truth, index in zip(generated_outputs, ground_truths, indices):
            scores = self._rouge_scorer.score(
                str(generated), str(ground_truth)
            ).details
            rouge1_recall[str(index)] = float(scores["rouge1"])
            rouge_l_recall[str(index)] = float(scores["rougeL"])
        return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}

    def _perturb_answer_values(self, sample, spec):
        perturb_answers = self._answer_values(
            sample, spec.get("perturbed_answer_key", "perturbed_answers")
        )
        if perturb_answers:
            return perturb_answers
        return sample.get("auto_perturbed_answers", [])

    def _evaluate_perturbation_ratio(self, samples, model, tokenizer, spec):
        logs = {
            "average_perturb_loss": {},
            "avg_paraphrased_loss": {},
            "truth_ratio": {},
            "paraphrased_loss": {},
            "perturb_loss": {},
            "num_token_paraphrased": {},
            "num_token_perturb": {},
        }
        answer_key = spec.get("answer_key", self.eval_cfg.get("metadata_answer_key", "answer"))
        base_answer_key = spec.get("base_answer_key", answer_key)
        for sample in samples:
            index = str(sample["index"])
            base_answers = self._answer_values(
                sample,
                base_answer_key,
                fallback_keys=[
                    answer_key,
                    self.eval_cfg.get("metadata_answer_key", "Answer"),
                    "answer",
                    "name",
                ],
            )
            perturb_answers = self._perturb_answer_values(sample, spec)
            if not base_answers or not perturb_answers:
                logger.warning(
                    "Skipping CLEAR perturbation metric for sample %s due to missing answers.",
                    index,
                )
                continue

            base_loss = self._answer_loss(model, tokenizer, sample, base_answers[0])
            if base_loss is None:
                logger.warning(
                    "Skipping CLEAR perturbation metric for sample %s due to empty base labels.",
                    index,
                )
                continue

            perturb_losses = []
            for answer in perturb_answers:
                loss = self._answer_loss(model, tokenizer, sample, answer)
                if loss is not None:
                    perturb_losses.append(loss)
            if not perturb_losses:
                logger.warning(
                    "Skipping CLEAR perturbation metric for sample %s due to empty perturb labels.",
                    index,
                )
                continue

            avg_perturb = [loss["avg_loss"] for loss in perturb_losses]
            perturb_loss = [loss["loss"] for loss in perturb_losses]
            num_token_perturb = [loss["num_tokens"] for loss in perturb_losses]
            logs["average_perturb_loss"][index] = avg_perturb
            logs["avg_paraphrased_loss"][index] = base_loss["avg_loss"]
            logs["truth_ratio"][index] = self._truth_ratio_scorer.score(
                base_loss["avg_loss"], avg_perturb
            ).value
            logs["paraphrased_loss"][index] = base_loss["loss"]
            logs["perturb_loss"][index] = perturb_loss
            logs["num_token_paraphrased"][index] = base_loss["num_tokens"]
            logs["num_token_perturb"][index] = num_token_perturb
        return logs

    def _normalize_gt_loss(self, logs):
        if "avg_gt_loss" not in logs or "average_perturb_loss" not in logs:
            return
        normalized = {}
        for index, avg_gt_loss in logs["avg_gt_loss"].items():
            if index not in logs["average_perturb_loss"]:
                continue
            probability = self._contrastive_probability_scorer.score(
                avg_gt_loss,
                logs["average_perturb_loss"][index],
            ).value
            normalized[index] = (
                float("inf") if probability == 0 else float(-math.log(probability))
            )
        logs["normalized_gt_loss"] = normalized

    def _evaluate_task(self, spec, model, tokenizer):
        samples = self._samples_for_task(spec)
        logger.info(
            "Running CLEAR task=%s split=%s samples=%s",
            spec.get("eval_task"),
            spec.get("split"),
            len(samples),
        )
        # CLEAR's +tofu QA rows have only question/answer fields; they do not
        # provide paraphrased or perturbed answers for a truth-ratio score.
        if str(spec.get("image_mode", "any")).lower() == "text":
            logs = {}
        else:
            logs = self._evaluate_perturbation_ratio(samples, model, tokenizer, spec)
        logs.update({"avg_gt_loss": {}, "gt_loss": {}, "num_token_gt": {}})
        generation_mia = bool(self.eval_cfg.get("generation_mia", False))
        generation_min_k_mia = bool(
            self.eval_cfg.get("generation_mia_min_k", False)
        )
        if generation_mia:
            logs.update({"loss_mia": {}, "zlib_mia": {}})
        if generation_min_k_mia:
            logs.update({"min_k_20_mia": {}, "min_k_plus_plus_20_mia": {}})
        if self.eval_cfg.get("save_generated_text", True):
            logs["generated_text"] = {}

        answer_key = spec.get("answer_key", self.eval_cfg.get("metadata_answer_key", "answer"))
        generated_outputs = []
        ground_truths = []
        indices = []
        for sample in samples:
            index = str(sample["index"])
            answers = self._answer_values(
                sample,
                answer_key,
                fallback_keys=[
                    self.eval_cfg.get("metadata_answer_key", "Answer"),
                    "answer",
                    "name",
                ],
            )
            if not answers:
                logger.warning("Skipping CLEAR sample %s due to missing answer.", index)
                continue
            answer = answers[0]
            question = self._question_for_sample(sample)
            loss_result = self._answer_loss(
                model,
                tokenizer,
                sample,
                answer,
                question=question,
                compute_token_stats=generation_min_k_mia,
            )
            if loss_result is None:
                logger.warning("Skipping CLEAR sample %s due to empty labels.", index)
                continue
            prompt, generated = self._generate(
                model, tokenizer, sample, answer=answer, question=question
            )

            logs["avg_gt_loss"][index] = loss_result["avg_loss"]
            logs["gt_loss"][index] = loss_result["loss"]
            logs["num_token_gt"][index] = loss_result["num_tokens"]
            if generation_mia:
                logs["loss_mia"][index] = self._loss_mia_scorer.score(
                    loss_result["avg_loss"]
                ).value
                logs["zlib_mia"][index] = self._zlib_mia_scorer.score(
                    loss_result["avg_loss"], answer
                ).value
            if generation_min_k_mia:
                logs["min_k_20_mia"][index] = self._standard_min_k_scorer.score(
                    loss_result["token_log_probs"]
                ).value
                logs["min_k_plus_plus_20_mia"][index] = (
                    self._standard_min_k_plus_plus_scorer.score(
                        loss_result["token_log_probs"],
                        loss_result["mu"],
                        loss_result["sigma"],
                    ).value
                )
            if self.eval_cfg.get("save_generated_text", True):
                logs["generated_text"][index] = [
                    prompt,
                    generated,
                    answer,
                    sample["category"],
                ]
            generated_outputs.append(generated)
            ground_truths.append(answer)
            indices.append(index)

        logs.update(self._rouge_recall(generated_outputs, ground_truths, indices))
        if "eval_log" not in str(spec.get("eval_task", "")):
            self._normalize_gt_loss(logs)
        return logs

    def _task_specs_from_lists(self):
        data_paths = self._as_list(self.eval_cfg.get("data_path", []))
        split_list = self._as_list(
            self.eval_cfg.get("split_list", self.eval_cfg.get("splits", []))
        )
        if not data_paths or not split_list:
            return []
        specs = []
        for index, data_path in enumerate(data_paths):
            specs.append(
                {
                    "enabled": True,
                    "data_path": data_path,
                    "split": self._value_at(split_list, index),
                    "question_key": self._value_at(
                        self.eval_cfg.get("question_key"), index, "question"
                    ),
                    "question_strategy": self._value_at(
                        self.eval_cfg.get("question_strategy"), index, "column"
                    ),
                    "answer_key": self._value_at(
                        self.eval_cfg.get("answer_key"), index, "answer"
                    ),
                    "base_answer_key": self._value_at(
                        self.eval_cfg.get("base_answer_key"), index, "answer"
                    ),
                    "perturbed_answer_key": self._value_at(
                        self.eval_cfg.get("perturbed_answer_key"), index, "options"
                    ),
                    "eval_task": self._value_at(
                        self.eval_cfg.get("eval_task"), index, f"eval_{index}"
                    ),
                }
            )
        return specs

    def _task_specs(self):
        task_specs = self._to_container(self.eval_cfg.get("task_specs", None), None)
        specs = task_specs if task_specs is not None else self._task_specs_from_lists()
        eval_task_ids = self.eval_cfg.get("eval_task_ids", None)
        if str(eval_task_ids) in ("None", "null", ""):
            eval_task_ids = None
        eval_task_ids = set(int(item) for item in self._as_list(eval_task_ids)) if eval_task_ids is not None else None

        resolved_specs = []
        for index, spec in enumerate(specs):
            if eval_task_ids is not None and index not in eval_task_ids:
                continue
            if not spec.get("enabled", True):
                continue
            pure_text_partition = spec.get("pure_text_partition", None)
            if pure_text_partition:
                try:
                    forget_ratio = int(str(self.eval_cfg.get("forget_ratio", "10")))
                except ValueError as exc:
                    raise ValueError(
                        "eval.clear.forget_ratio must be an integer for pure-text eval."
                    ) from exc
                if pure_text_partition == "forget":
                    pure_text_split = f"forget{forget_ratio:02d}+tofu"
                elif pure_text_partition == "retain":
                    pure_text_split = f"retain{100 - forget_ratio}+tofu"
                else:
                    raise ValueError(
                        "CLEAR pure_text_partition must be 'forget' or 'retain'."
                    )
                spec["split"] = pure_text_split
                if spec.get("data_path", None) in (None, "", "null", "None"):
                    pure_text_root = self.eval_cfg.get("pure_text_data_root", None)
                    if pure_text_root in (None, "", "null", "None"):
                        raise ValueError(
                            "eval.clear.pure_text_data_root must be set when "
                            "pure_text_eval is enabled."
                        )
                    spec["data_path"] = str(Path(str(pure_text_root)) / pure_text_split)
            if spec.get("data_path", None) in (None, "", "null", "None"):
                logger.info("Skipping disabled CLEAR task spec %s.", spec.get("eval_task", index))
                continue
            spec.setdefault("eval_task", f"eval_{index}")
            spec.setdefault("split", None)
            spec.setdefault("question_strategy", "column")
            resolved_specs.append(spec)
        if not resolved_specs:
            raise ValueError("eval.clear must define at least one enabled task spec.")
        return resolved_specs

    def _flatten_numeric(self, value):
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            values = [value]
        flattened = []
        for item in values:
            if isinstance(item, (list, tuple, np.ndarray)):
                flattened.extend(
                    float(x)
                    for x in item
                    if isinstance(x, (int, float, np.integer, np.floating))
                    and not math.isnan(float(x))
                )
            elif isinstance(item, (int, float, np.integer, np.floating)):
                item = float(item)
                if not math.isnan(item):
                    flattened.append(item)
        return flattened

    def _dict_values_to_array(self, value):
        if isinstance(value, dict):
            value = list(value.values())
        try:
            return np.array(value, dtype=float)
        except (TypeError, ValueError):
            return np.array(self._flatten_numeric(value), dtype=float)

    def _generated_rouge_recall(self, generated_text):
        rouge1_recall = {}
        rouge_l_recall = {}
        for index, pair in generated_text.items():
            if len(pair) < 3:
                continue
            _, generated, ground_truth, *_ = pair
            scores = self._rouge_scorer.score(
                str(generated), str(ground_truth)
            ).details
            rouge1_recall[str(index)] = float(scores["rouge1"])
            rouge_l_recall[str(index)] = float(scores["rougeL"])
        return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}

    def _harmonic_mean(self, values):
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

    def _task_name(self, task_file):
        mapping = self._to_container(self.eval_cfg.get("file_to_task", {}), {})
        if isinstance(mapping, dict) and task_file in mapping:
            return mapping[task_file]
        return FILE_TO_TASK.get(task_file, task_file.replace(".json", ""))

    def _task_truth_distribution(self, task_result):
        """Build the per-sample CLEAR truth-ratio distribution.

        ``average_perturb_loss`` is a mapping from sample id to a list of
        perturbation losses.  The perturbations must be averaged *within each
        sample* before comparing them with that sample's paraphrase loss;
        averaging the flattened mapping would mix unrelated samples.
        """
        paraphrase_values = task_result.get("avg_paraphrased_loss", {})
        perturbed_values = task_result.get("average_perturb_loss", {})

        if isinstance(paraphrase_values, dict) and isinstance(perturbed_values, dict):
            pairs = [
                (paraphrase_values[index], perturbed_values[index])
                for index in paraphrase_values
                if index in perturbed_values
            ]
        else:
            paraphrase_items = self._flatten_numeric(paraphrase_values)
            if isinstance(perturbed_values, (list, tuple, np.ndarray)):
                perturbed_items = list(perturbed_values)
            else:
                perturbed_items = [perturbed_values]
            if len(paraphrase_items) != len(perturbed_items):
                return np.array([])
            pairs = list(zip(paraphrase_items, perturbed_items))

        ratios = []
        for paraphrase_value, perturbation_values in pairs:
            base = self._flatten_numeric(paraphrase_value)
            perturbations = self._flatten_numeric(perturbation_values)
            if len(base) != 1 or not perturbations:
                continue
            difference = float(np.mean(perturbations)) - float(base[0])
            try:
                ratios.append(math.exp(difference))
            except OverflowError:
                ratios.append(float("inf"))
        return np.asarray(ratios, dtype=float)

    def _probability_metric(self, task_file, task_result):
        avg_gt_loss = self._dict_values_to_array(task_result["avg_gt_loss"])
        if avg_gt_loss.size == 0:
            return None
        true_probs = np.asarray(
            [self._answer_probability_scorer.score(loss).value for loss in avg_gt_loss],
            dtype=float,
        )
        if "eval_log" in task_file or "average_perturb_loss" not in task_result:
            return float(true_probs.mean())
        false_values = task_result["average_perturb_loss"]
        if isinstance(false_values, dict):
            false_values = list(false_values.values())
        if len(false_values) != true_probs.shape[0]:
            return float(true_probs.mean())
        contrastive_probabilities = []
        for target_loss, value in zip(avg_gt_loss, false_values):
            losses = self._flatten_numeric(value)
            if not losses:
                return float(true_probs.mean())
            contrastive_probabilities.append(
                self._contrastive_probability_scorer.score(target_loss, losses).value
            )
        return float(np.mean(contrastive_probabilities))

    def _compute_model_utility(self, eval_results):
        aggregated_results = {}
        for task_file, task_result in eval_results.items():
            if not isinstance(task_result, dict) or "avg_gt_loss" not in task_result:
                continue
            task_name = self._task_name(task_file)
            probability = self._probability_metric(task_file, task_result)
            if probability is not None:
                aggregated_results[f"Prob. {task_name}"] = probability

            if not task_result.get("rougeL_recall") and task_result.get(
                "generated_text"
            ):
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
                aggregated_results[f"Truth Ratio {task_name}"] = float(
                    np.mean(transformed)
                )

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
            value
            for key, value in aggregated_results.items()
            if "Pure Text Retain" in key
        ]
        pure_text_utility = self._harmonic_mean(text_utility_values)
        if pure_text_utility is not None:
            aggregated_results["Pure Text Utility"] = pure_text_utility

        if self.eval_cfg.get("include_extended_group_metrics", False):
            self._add_group_metrics(aggregated_results)
        return aggregated_results

    def _add_group_metrics(self, aggregated_results):
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

    def _evaluate_forget_quality(self, unlearned_data, retained_data):
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
            logger.warning(
                "scipy is not installed; skipping CLEAR forget-quality metrics."
            )
            return {}
        return {
            "KS test p-value": float(ks_result.details["pvalue"]),
            "JS metric": float(js_result.value),
        }

    def _load_retain_result(self):
        retain_result = self.eval_cfg.get("retain_result", None)
        if retain_result in (None, "", "null", "None"):
            return None
        with open(retain_result, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def summarize(self, logs):
        summary = {}
        for task_name, task_logs in logs.items():
            if not isinstance(task_logs, dict):
                continue
            for metric_name, value in task_logs.items():
                if metric_name == "generated_text":
                    continue
                values = self._flatten_numeric(value)
                if not values:
                    continue
                summary[f"{task_name}/{metric_name}/mean"] = float(np.mean(values))
                summary[f"{task_name}/{metric_name}/count"] = len(values)

        summary.update(self._compute_model_utility(logs))
        retained_data = self._load_retain_result()
        if retained_data is not None:
            summary.update(self._evaluate_forget_quality(logs, retained_data))
        return summary

    def _safe_split_name(self, split):
        split = "none" if split is None else str(split)
        return split.replace("/", "_").replace(" ", "_")

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        model = self.prepare_model(model)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("CLEAREvaluator requires a tokenizer.")
        self._load_processor()

        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        with open_dict(self.eval_cfg):
            self.eval_cfg.output_dir = output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info("***** Running %s evaluation suite *****", self.name)
        logger.info("Fine-grained evaluations will be saved to: %s", logs_file_path)
        logger.info("Aggregated evaluations will be summarised in: %s", summary_file_path)

        for spec in self._task_specs():
            task_key = f"{spec['eval_task']}.json"
            split_file_path = os.path.join(
                output_dir,
                f"{self._safe_split_name(spec.get('split'))}_{spec['eval_task']}.json",
            )
            if not overwrite and task_key in logs and logs[task_key]:
                logger.info("Skipping %s, already evaluated.", task_key)
                continue
            if os.path.exists(split_file_path) and not overwrite:
                logger.info("Loading existing CLEAR task log from %s", split_file_path)
                with open(split_file_path, "r", encoding="utf-8") as handle:
                    task_logs = json.load(handle)
            else:
                task_logs = self._evaluate_task(spec, model, tokenizer)
                os.makedirs(output_dir, exist_ok=True)
                with open(split_file_path, "w", encoding="utf-8") as handle:
                    json.dump(task_logs, handle, indent=4)
            logs[task_key] = task_logs
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        for aggregate_name in ("CLEAR_eval_log_aggregated.json", "eval_log_aggregated.json"):
            aggregate_path = os.path.join(output_dir, aggregate_name)
            with open(aggregate_path, "w", encoding="utf-8") as handle:
                json.dump(logs, handle, indent=4)

        summary = self.summarize(logs)
        self.save_logs(summary, summary_file_path)
        return summary
