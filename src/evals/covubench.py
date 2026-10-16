import logging
import math
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from evals.base import Evaluator
from evals.scorers import (
    KeywordRecallScorer,
    RougeScorer,
    SemanticDissimilarityScorer,
    normalize_text,
    parse_keywords as _parse_keywords,
)


logger = logging.getLogger("evaluator")
_KEYWORD_SCORER = KeywordRecallScorer()
_SEMANTIC_SCORER = SemanticDissimilarityScorer()


def parse_keywords(value):
    """Parse CoVUBench's JSON keyword field into a flat list of phrases."""
    return _parse_keywords(value)


def _normalize_keyword_text(value):
    return normalize_text(value)


def keyword_match_details(prediction, keywords):
    """Return keyword phrases, match flags, and per-answer keyword recall."""
    details = _KEYWORD_SCORER.match_details(prediction, keywords)
    return details["keywords"], details["matches"], details["recall"]


def keyword_recall(prediction, keywords):
    """Compute the CoVUBench keyword-recall Exact Match for one answer."""
    return keyword_match_details(prediction, keywords)[2]


def semantic_dissimilarity_scores(prediction_embeddings, answer_embeddings):
    """Compute pairwise cosine dissimilarity on CoVUBench's 0-100 scale."""
    return _SEMANTIC_SCORER.values(prediction_embeddings, answer_embeddings)


def _mean(values):
    values = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


class CoVUBenchEvaluator(Evaluator):
    """Evaluate the first five metrics from the CoVUBench protocol."""

    _ROW_COLUMNS = (
        "image",
        "question",
        "answer",
        "name",
        "type",
        "keywords",
        "question_type",
    )

    def __init__(self, eval_cfg, **kwargs):
        self.name = "COVUBENCH"
        self.eval_cfg = eval_cfg
        self.processor = None
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._embedding_device = None
        self._rouge_scorer = RougeScorer(
            ("rougeL",),
            use_stemmer=bool(self.eval_cfg.get("rouge_use_stemmer", True)),
            aggregation="recall",
        )

    def prepare_model(self, model):
        model.eval()
        return model

    def _to_container(self, value, default=None):
        if value is None:
            return default
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _load_processor(self):
        if self.processor is not None:
            return self.processor
        processor_path = self.eval_cfg.get("processor_path", None)
        if not processor_path:
            raise ValueError("eval.covubench.processor_path must be set.")
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=bool(
                self.eval_cfg.get("processor_trust_remote_code", False)
            ),
        )
        processor_tokenizer = getattr(self.processor, "tokenizer", None)
        if processor_tokenizer is not None:
            padding_side = self.eval_cfg.get("padding_side", None)
            if padding_side is not None:
                processor_tokenizer.padding_side = padding_side
            if (
                getattr(processor_tokenizer, "pad_token", None) is None
                and getattr(processor_tokenizer, "eos_token", None) is not None
            ):
                processor_tokenizer.pad_token = processor_tokenizer.eos_token
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
        return "raw" if self._model_family(model) == "llava" else "chat_template"

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

    def _split_pattern(self, split):
        configured = self.eval_cfg.get(f"{split}_pattern", None)
        if configured:
            return str(configured)
        if split == "forget":
            return f"forget{int(self.eval_cfg.get('ratio', 5))}-*.parquet"
        return f"{split}-*.parquet"

    def _split_files(self, split):
        data_root = Path(str(self.eval_cfg.data_root)).expanduser()
        files = sorted(data_root.glob(self._split_pattern(split)))
        if not files:
            raise FileNotFoundError(
                f"No CoVUBench {split} parquet files matched "
                f"{data_root / self._split_pattern(split)}"
            )
        return files

    def _concepts_for_split(self, split):
        import pyarrow.parquet as pq

        concepts = set()
        for file_path in self._split_files(split):
            names = pq.read_table(file_path, columns=["name"])["name"].to_pylist()
            concepts.update(str(name) for name in names)
        return concepts

    def _row_is_selected(self, split, row, forget_concepts):
        if not self.eval_cfg.get("filter_by_forget_concepts", True):
            return True
        concept = str(row.get("name", ""))
        if split == "test":
            return concept in forget_concepts
        if split == "retain":
            return concept not in forget_concepts
        return True

    def _iter_split_rows(self, split, forget_concepts):
        import pyarrow.parquet as pq

        batch_size = int(self.eval_cfg.get("parquet_batch_size", 8))
        max_samples = self.eval_cfg.get("max_samples_per_split", None)
        max_samples = int(max_samples) if max_samples is not None else None
        raw_index = 0
        selected_count = 0
        for file_path in self._split_files(split):
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size, columns=list(self._ROW_COLUMNS)
            ):
                for row in batch.to_pylist():
                    sample_id = f"{split}:{raw_index:06d}"
                    raw_index += 1
                    if not self._row_is_selected(split, row, forget_concepts):
                        continue
                    if max_samples is not None and selected_count >= max_samples:
                        return
                    row["_sample_id"] = sample_id
                    row["_source_file"] = str(file_path)
                    selected_count += 1
                    yield row

    def _selected_count(self, split, forget_concepts):
        import pyarrow.parquet as pq

        count = 0
        max_samples = self.eval_cfg.get("max_samples_per_split", None)
        max_samples = int(max_samples) if max_samples is not None else None
        for file_path in self._split_files(split):
            names = pq.read_table(file_path, columns=["name"])["name"].to_pylist()
            for name in names:
                if self._row_is_selected(split, {"name": name}, forget_concepts):
                    count += 1
                    if max_samples is not None and count >= max_samples:
                        return count
        return count

    def _resize_image(self, image):
        max_pixels = self.eval_cfg.get("image_max_pixels", None)
        if max_pixels is None:
            return image
        width, height = image.size
        pixels = width * height
        if pixels <= int(max_pixels):
            return image
        scale = math.sqrt(int(max_pixels) / pixels)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _resolve_image_path(self, path_value, source_file):
        raw_path = Path(str(path_value)).expanduser()
        candidates = [
            raw_path,
            Path(source_file).parent / raw_path,
            Path(str(self.eval_cfg.data_root)) / raw_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not resolve CoVUBench image {path_value} from {source_file}."
        )

    def _load_image(self, row):
        image_value = row.get("image")
        if isinstance(image_value, Image.Image):
            image = image_value.convert("RGB")
        elif isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            if image_bytes:
                with Image.open(BytesIO(image_bytes)) as source_image:
                    image = source_image.convert("RGB")
            elif image_value.get("path"):
                image_path = self._resolve_image_path(
                    image_value["path"], row["_source_file"]
                )
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
            else:
                raise ValueError(
                    "CoVUBench image field contains neither bytes nor path."
                )
        elif isinstance(image_value, (bytes, bytearray)):
            with Image.open(BytesIO(image_value)) as source_image:
                image = source_image.convert("RGB")
        elif image_value:
            image_path = self._resolve_image_path(image_value, row["_source_file"])
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        else:
            raise ValueError("CoVUBench multimodal sample has no image.")
        return self._resize_image(image)

    def _uses_image(self, question_type):
        configured = self._to_container(
            self.eval_cfg.get("image_question_types", ["multi_modal", "caption"]),
            ["multi_modal", "caption"],
        )
        return str(question_type) in {str(value) for value in configured}

    def _chat_template_text(self, question, include_image):
        processor = self._load_processor()
        messages = []
        system_prompt = self.eval_cfg.get("system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content = []
        if include_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _raw_text(self, question, include_image):
        system_prompt = self.eval_cfg.get(
            "raw_system_prompt",
            "A chat between a curious human and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the human's questions. ",
        )
        question_start = self.eval_cfg.get("question_start_tag", "USER: ")
        answer_tag = self.eval_cfg.get("answer_tag", " ASSISTANT: ")
        image_token = self.eval_cfg.get("image_token", "<image>")
        image_part = f"{image_token}\n" if include_image else ""
        return f"{system_prompt}{question_start}{image_part}{question}{answer_tag}"

    def _build_prompt_and_images(self, row, model):
        question = str(row.get("question", "")).strip()
        include_image = self._uses_image(row.get("question_type", ""))
        image = self._load_image(row) if include_image else None
        style = self._effective_prompt_style(model)
        if style == "chat_template":
            prompt = self._chat_template_text(question, include_image)
        elif style == "raw":
            prompt = self._raw_text(question, include_image)
        else:
            raise ValueError(f"Unknown CoVUBench prompt_style: {style}")
        return prompt, [image] if image is not None else []

    def _processor_images(self, images):
        processor = self._load_processor()
        if processor.__class__.__name__ == "Gemma3Processor":
            return [[image] for image in images]
        return images

    def _prepare_inputs(self, prompt, images, tokenizer):
        processor = self._load_processor()
        processor_kwargs = {"text": prompt, "return_tensors": "pt"}
        max_length = self.eval_cfg.get("max_length", None)
        if max_length is not None:
            processor_kwargs.update(
                {
                    "max_length": int(max_length),
                    "truncation": bool(self.eval_cfg.get("truncation", True)),
                }
            )
        if images:
            processor_kwargs["images"] = self._processor_images(images)
        try:
            return processor(**processor_kwargs)
        except Exception:
            if images:
                raise
            tokenizer_kwargs = {
                key: value
                for key, value in processor_kwargs.items()
                if key not in ("text", "images", "return_tensors")
            }
            return tokenizer(prompt, return_tensors="pt", **tokenizer_kwargs)

    def _clean_answer(self, value):
        text = str(value)
        for marker in (
            "### Answer:",
            "ASSISTANT:",
            "Assistant:",
            "assistant:",
            "Answer:",
        ):
            if marker in text:
                text = text.split(marker)[-1]
        for end_marker in ("<|im_end|>", "</s>"):
            if end_marker in text:
                text = text.split(end_marker)[0]
        return text.strip()

    def _generate(self, model, tokenizer, row):
        prompt, images = self._build_prompt_and_images(row, model)
        inputs = self._prepare_inputs(prompt, images, tokenizer)
        inputs = self._move_to_device(inputs, self._model_device(model))
        input_length = inputs["input_ids"].shape[-1]
        generation_args = self._to_container(
            self.eval_cfg.get("generation_args", {}), {}
        )
        if (
            "pad_token_id" not in generation_args
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            generation_args["pad_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_args)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = (
            sequences[:, input_length:]
            if sequences.shape[-1] > input_length
            else sequences
        )
        processor = self._load_processor()
        decoder = processor.decode if hasattr(processor, "decode") else tokenizer.decode
        prediction = decoder(generated_ids[0], skip_special_tokens=True)
        return prompt, self._clean_answer(prediction)

    def _sample_log(self, row, model, tokenizer, split):
        prompt, prediction = self._generate(model, tokenizer, row)
        keywords, keyword_matches, em_score = keyword_match_details(
            prediction, row.get("keywords")
        )
        question_type = str(row.get("question_type", ""))
        image_value = row.get("image")
        image_path = image_value.get("path") if isinstance(image_value, dict) else None
        sample_log = {
            "sample_id": row["_sample_id"],
            "source_file": row["_source_file"],
            "image_path": image_path,
            "name": str(row.get("name", "")),
            "type": str(row.get("type", "")),
            "question_type": question_type,
            "uses_image": self._uses_image(question_type),
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "prediction": prediction,
            "keywords": keywords,
            "keyword_matches": keyword_matches,
            "keyword_recall": em_score,
        }
        if split == "retain":
            rouge = self._rouge_scorer.score(
                prediction, sample_log["answer"]
            )
            sample_log["rougeL_recall"] = float(rouge.value)
        if self.eval_cfg.get("save_prompts", False):
            sample_log["prompt"] = prompt
        return sample_log

    def _load_embedding_encoder(self):
        if self._embedding_model is not None:
            return self._embedding_tokenizer, self._embedding_model
        model_path = self.eval_cfg.get("embedding_model_path", None)
        if not model_path:
            raise ValueError("eval.covubench.embedding_model_path must be set.")
        load_args = {
            "local_files_only": bool(
                self.eval_cfg.get("embedding_local_files_only", True)
            ),
            "trust_remote_code": bool(
                self.eval_cfg.get("embedding_trust_remote_code", False)
            ),
        }
        self._embedding_tokenizer = AutoTokenizer.from_pretrained(
            model_path, **load_args
        )
        self._embedding_model = AutoModel.from_pretrained(model_path, **load_args)
        self._embedding_device = torch.device(
            str(self.eval_cfg.get("embedding_device", "cpu"))
        )
        if self._embedding_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA embedding device was requested, but CUDA is unavailable."
            )
        self._embedding_model.to(self._embedding_device)
        self._embedding_model.eval()
        return self._embedding_tokenizer, self._embedding_model

    def _encode_texts(self, texts):
        tokenizer, model = self._load_embedding_encoder()
        batch_size = int(self.eval_cfg.get("embedding_batch_size", 32))
        max_length = int(self.eval_cfg.get("embedding_max_length", 256))
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = [str(value) for value in texts[start : start + batch_size]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._embedding_device) for key, value in inputs.items()
            }
            with torch.inference_mode():
                outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            attention_mask = attention_mask.expand(outputs.last_hidden_state.size())
            masked_embeddings = outputs.last_hidden_state * attention_mask
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).clamp(
                min=1e-9
            )
            embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
        if not embeddings:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(embeddings, dim=0)

    def _release_embedding_encoder(self):
        self._embedding_tokenizer = None
        self._embedding_model = None
        embedding_device = self._embedding_device
        self._embedding_device = None
        if embedding_device is not None and embedding_device.type == "cuda":
            torch.cuda.empty_cache()

    def _ensure_divergence(self, split_log, checkpoint):
        samples = list(split_log.get("samples", {}).values())
        missing = [
            sample for sample in samples if "semantic_dissimilarity" not in sample
        ]
        try:
            if missing:
                predictions = [sample["prediction"] for sample in missing]
                answers = [sample["answer"] for sample in missing]
                embeddings = self._encode_texts(predictions + answers)
                midpoint = len(missing)
                prediction_embeddings = embeddings[:midpoint]
                answer_embeddings = embeddings[midpoint:]
                semantic_scores = _SEMANTIC_SCORER.score_many(
                    prediction_embeddings, answer_embeddings
                )
                for sample, semantic_score in zip(missing, semantic_scores):
                    sample["semantic_similarity"] = float(
                        semantic_score.details["cosine_similarity"]
                    )
                    sample["semantic_dissimilarity"] = float(semantic_score.value)
            split_log["divergence_completed"] = all(
                "semantic_dissimilarity" in sample for sample in samples
            )
            checkpoint()
        finally:
            self._release_embedding_encoder()

    def _run_signature(self, forget_concepts):
        return {
            "version": 1,
            "ratio": int(self.eval_cfg.get("ratio", 5)),
            "data_root": str(Path(str(self.eval_cfg.data_root)).resolve()),
            "split_patterns": {
                split: self._split_pattern(split)
                for split in ("forget", "test", "retain")
            },
            "forget_concepts": sorted(forget_concepts),
            "filter_by_forget_concepts": bool(
                self.eval_cfg.get("filter_by_forget_concepts", True)
            ),
            "max_samples_per_split": self.eval_cfg.get("max_samples_per_split", None),
            "prompt_style": str(self.eval_cfg.get("prompt_style", "auto")),
            "generation_args": self._to_container(
                self.eval_cfg.get("generation_args", {}), {}
            ),
            "embedding_model_path": str(self.eval_cfg.get("embedding_model_path", "")),
        }

    def _evaluate_split(
        self,
        split,
        split_log,
        forget_concepts,
        model,
        tokenizer,
        checkpoint,
    ):
        expected_count = self._selected_count(split, forget_concepts)
        split_log.setdefault("samples", {})
        split_log["expected_count"] = expected_count
        split_log["completed"] = False
        existing = split_log["samples"]
        checkpoint_interval = max(1, int(self.eval_cfg.get("checkpoint_interval", 10)))
        generated_since_checkpoint = 0
        try:
            for row in self._iter_split_rows(split, forget_concepts):
                sample_id = row["_sample_id"]
                if sample_id in existing:
                    continue
                existing[sample_id] = self._sample_log(row, model, tokenizer, split)
                generated_since_checkpoint += 1
                if generated_since_checkpoint >= checkpoint_interval:
                    split_log["processed_count"] = len(existing)
                    checkpoint()
                    generated_since_checkpoint = 0
        except Exception:
            split_log["processed_count"] = len(existing)
            checkpoint()
            raise
        split_log["processed_count"] = len(existing)
        if len(existing) != expected_count:
            raise RuntimeError(
                f"CoVUBench {split} produced {len(existing)} samples, expected "
                f"{expected_count}."
            )
        split_log["completed"] = True
        checkpoint()

    def _samples(self, logs, split, completed_only=True):
        split_log = logs.get("splits", {}).get(split, {})
        if completed_only and not split_log.get("completed", False):
            return []
        return list(split_log.get("samples", {}).values())

    def _scaled_metrics(self, forget_samples, test_samples, retain_samples):
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

    def _breakdown(self, logs, field):
        split_samples = {
            split: self._samples(logs, split) for split in ("forget", "test", "retain")
        }
        values = sorted(
            {
                str(sample.get(field, ""))
                for samples in split_samples.values()
                for sample in samples
                if str(sample.get(field, ""))
            }
        )
        breakdown = {}
        for value in values:
            filtered = {
                split: [
                    sample for sample in samples if str(sample.get(field, "")) == value
                ]
                for split, samples in split_samples.items()
            }
            metrics, _ = self._scaled_metrics(
                filtered["forget"], filtered["test"], filtered["retain"]
            )
            if metrics:
                breakdown[value] = metrics
        return breakdown

    def summarize(self, logs):
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

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        model = self.prepare_model(model)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("CoVUBenchEvaluator requires a tokenizer.")
        self._load_processor()

        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        forget_concepts = self._concepts_for_split("forget")
        signature = self._run_signature(forget_concepts)
        if logs.get("metadata") and logs["metadata"] != signature:
            raise ValueError(
                "Existing CoVUBench logs were created with a different evaluator "
                "configuration. Set eval.covubench.overwrite=true or use a new "
                "output directory."
            )
        logs["metadata"] = signature
        logs.setdefault("splits", {})

        def checkpoint():
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        logger.info("***** Running %s evaluation suite *****", self.name)
        logger.info("Blocklisted concepts: %s", sorted(forget_concepts))
        logger.info("Fine-grained evaluations will be saved to: %s", logs_file_path)
        logger.info("Aggregated evaluations will be saved to: %s", summary_file_path)

        for split in ("forget", "test", "retain"):
            split_log = logs["splits"].setdefault(split, {})
            if split_log.get("completed", False):
                logger.info("Skipping completed CoVUBench split: %s", split)
            else:
                logger.info(
                    "Running CoVUBench split=%s selected_samples=%s",
                    split,
                    self._selected_count(split, forget_concepts),
                )
                self._evaluate_split(
                    split,
                    split_log,
                    forget_concepts,
                    model,
                    tokenizer,
                    checkpoint,
                )
            if split == "forget" and split_log.get("completed", False):
                self._ensure_divergence(split_log, checkpoint)
                checkpoint()

        checkpoint()
        return self.summarize(logs)
