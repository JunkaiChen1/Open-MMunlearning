import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from PIL import Image
from transformers import AutoProcessor

from evals.base import Evaluator
from evals.scorers import (
    AnswerProbabilityScorer,
    ContrastiveProbabilityScorer,
    KeywordRecallScorer,
    LossMIAScorer,
    MinKPlusPlusScorer,
    MinKScorer,
    PerturbationTruthRatioScorer,
    RougeScorer,
    TruthRatioTransformScorer,
    TwoSampleKSScorer,
    ZlibMIAScorer,
)


logger = logging.getLogger("evaluator")
IGNORE_INDEX = -100


class FIUBenchEvaluator(Evaluator):
    """Evaluator for the original FIUBench protocol.

    The FIUBench release evaluates QA pairs from the first 400 identities,
    filtered by `split.json`, and reports generation, loss-based probability
    statistics, perturbation truth ratios, ROUGE recall, Exact Match, and
    optional Min-K MIA scores. This implementation keeps those benchmark rules
    while using OpenUnlearning's model/processor loading path.
    """

    def __init__(self, eval_cfg, **kwargs):
        self.name = "FIUBENCH"
        self.eval_cfg = eval_cfg
        self.processor = None
        self._records_cache = {}
        self._splits = None
        self._image_cache = {}
        # FIUBench's upstream Exact Match is a direct ``lower()`` substring
        # check for each keyword phrase.
        self._keyword_scorer = KeywordRecallScorer(normalization="lower")
        self._answer_probability_scorer = AnswerProbabilityScorer()
        self._contrastive_probability_scorer = ContrastiveProbabilityScorer()
        self._truth_ratio_scorer = PerturbationTruthRatioScorer()
        self._truth_ratio_transform_scorer = TruthRatioTransformScorer()
        # The historical FIUBench logs use the negative-loss orientation;
        # keep that contract while sharing the model-agnostic MIA formulas.
        self._loss_mia_scorer = LossMIAScorer(sign=-1.0)
        self._zlib_mia_scorer = ZlibMIAScorer(sign=-1.0)
        self._min_k_scorer = MinKScorer()
        self._min_k_plus_plus_scorer = MinKPlusPlusScorer()
        self._ks_scorer = TwoSampleKSScorer()
        self._rouge_scorer = RougeScorer(("rouge1", "rougeL"), aggregation="recall")
        self._warned_gpt = False
        self._active_model_family = None
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
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, ListConfig)):
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
        return value

    def _load_processor(self):
        if self.processor is not None:
            return self.processor
        processor_path = self.eval_cfg.get("processor_path", None)
        if not processor_path:
            raise ValueError("eval.fiubench.processor_path must be set.")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        if self.eval_cfg.get("padding_side", None) is not None:
            self.processor.tokenizer.padding_side = self.eval_cfg.padding_side
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

    def _uses_upstream_llava_io(self):
        """Whether to reproduce FIUBench's tokenizer/CLIP LLaVA path."""
        explicit = self.eval_cfg.get("upstream_llava_io", None)
        if explicit is not None:
            return bool(explicit) and bool(
                self.eval_cfg.get("upstream_compatibility", True)
            )
        family = str(self._active_model_family or "").replace("_", "-")
        return bool(
            self.eval_cfg.get("upstream_compatibility", True)
            and self._active_prompt_style == "raw"
            and family in {"llava-phi", "llava-phi-3-mini"}
        )

    def _raw_prompt_fields(self):
        """Return the model tags used by FIUBench's original data module."""
        family = self._active_model_family or str(
            self.eval_cfg.get("model_family", "")
        ).lower()
        if family in {"llava-phi", "llava_phi"}:
            return {
                "system_prompt": "",
                "question_start": "<|user|>\n",
                "answer_tag": "<|end|>\n<|assistant|>\n",
            }
        return {
            "system_prompt": self.eval_cfg.get(
                "raw_system_prompt",
                "A chat between a curious human and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the human's questions. ",
            ),
            "question_start": self.eval_cfg.get("question_start_tag", "USER: "),
            "answer_tag": self.eval_cfg.get("answer_tag", " ASSISTANT: "),
        }

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

    def _load_json_or_jsonl(self, path):
        path = str(path)
        if path in self._records_cache:
            return self._records_cache[path]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            with open(path, "r", encoding="utf-8") as handle:
                data = [json.loads(line) for line in handle if line.strip()]
        max_people = self.eval_cfg.get("max_people", 400)
        if max_people is not None:
            data = data[: int(max_people)]
        self._records_cache[path] = data
        return data

    def _load_splits(self):
        if self._splits is not None:
            return self._splits
        split_path = self.eval_cfg.get("split_path", None)
        if not split_path:
            raise ValueError("eval.fiubench.split_path must be set.")
        with open(split_path, "r", encoding="utf-8") as handle:
            self._splits = json.load(handle)
        return self._splits

    def _resolve_split_key(self, splits, split):
        if split in splits:
            return split
        normalized = str(split).replace("_", "")
        for key in splits:
            if str(key).replace("_", "") == normalized:
                return key
        return None

    def _split_ratio(self, split, prefix):
        normalized = str(split).replace("_", "")
        if not normalized.startswith(prefix):
            return None
        suffix = normalized[len(prefix):]
        return int(suffix) if suffix.isdigit() else None

    def _filtered_records(self, data_path, split):
        records = self._load_json_or_jsonl(data_path)
        splits = self._load_splits()
        split = str(split)
        split_key = self._resolve_split_key(splits, split)
        if split_key is not None:
            ids = {str(value) for value in splits[split_key]}
            return [record for record in records if str(record.get("unique_id")) in ids]

        retain_ratio = self._split_ratio(split, "retain")
        if retain_ratio is not None and self.eval_cfg.get(
            "allow_generated_retain_split", False
        ):
            forget_ratio = 100 - retain_ratio
            forget_key = self._resolve_split_key(splits, f"forget{forget_ratio}")
            if forget_key is None:
                raise ValueError(
                    f"Cannot build FIUBench {split}: missing forget{forget_ratio} "
                    f"in {self.eval_cfg.split_path}. Available splits: {list(splits.keys())}"
                )
            forget_ids = {str(value) for value in splits[forget_key]}
            return [
                record
                for record in records
                if str(record.get("unique_id")) not in forget_ids
            ]

        if split == "retain":
            forget_ids = set()
            for key, values in splits.items():
                if str(key).startswith("forget"):
                    forget_ids.update(str(value) for value in values)
            return [
                record
                for record in records
                if str(record.get("unique_id")) not in forget_ids
            ]
        raise ValueError(f"Unknown FIUBench split: {split}")

    def _format_question(self, value):
        text = str(value).strip()
        if self.eval_cfg.get("capitalize_text", True):
            return text.capitalize()
        return text

    def _format_answer_values(self, raw_value):
        values = []
        if isinstance(raw_value, (list, tuple, ListConfig)):
            for item in raw_value:
                if item is None or not str(item).strip():
                    continue
                values.append(str(item).strip())
            return values
        if raw_value is None or not str(raw_value).strip():
            return []
        text = str(raw_value).strip()
        if self.eval_cfg.get("capitalize_text", True):
            text = text.capitalize()
        return [text]

    def _qa_values(self, qa, key, fallback_key=None):
        values = self._format_answer_values(qa.get(key))
        if values:
            return values
        if fallback_key and self.eval_cfg.get("fallback_empty_answer_to_answer", True):
            return self._format_answer_values(qa.get(fallback_key))
        return []

    def _question_values(self, qa, key):
        raw = qa.get(key)
        values = []
        for value in self._as_list(raw):
            if value is None or not str(value).strip():
                continue
            values.append(self._format_question(value))
        if values:
            return values
        if key != "question" and self.eval_cfg.get("fallback_empty_question_to_question", True):
            return self._question_values(qa, "question")
        return []

    def _resolve_image_path(self, image_path, data_path):
        raw = str(image_path)
        if os.path.isabs(raw) and os.path.exists(raw):
            return raw

        image_root = Path(self.eval_cfg.get("image_root", self.eval_cfg.data_root))
        data_parent = Path(data_path).parent
        clean = raw.replace("\\", "/").lstrip("./")
        candidates = [
            image_root / raw,
            image_root / clean,
            data_parent / raw,
            data_parent / clean,
        ]
        if clean.startswith("dataset/"):
            stripped = clean[len("dataset/") :]
            candidates.extend([image_root / stripped, data_parent / stripped])
        candidates.append(image_root / Path(clean).name)

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not resolve FIUBench image {image_path}. Tried: "
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

    def _load_image(self, image_path):
        if self.eval_cfg.get("cache_images", True) and image_path in self._image_cache:
            return self._image_cache[image_path]
        image = Image.open(image_path).convert("RGB")
        image = self._resize_image(image)
        if self.eval_cfg.get("cache_images", True):
            self._image_cache[image_path] = image
        return image

    def _samples_for_split(self, data_path, split, question_key):
        records = self._filtered_records(data_path, split)
        samples = []
        for record in records:
            image_path = self._resolve_image_path(record["image_path"], data_path)
            for qa_index, qa in enumerate(record.get("qa_list", [])):
                for question_index, question in enumerate(
                    self._question_values(qa, question_key)
                ):
                    samples.append(
                        {
                            "index": len(samples),
                            "person_id": str(record.get("unique_id")),
                            "qa_index": qa_index,
                            "question_index": question_index,
                            "question": question,
                            "image_path": image_path,
                            "category": "human_face",
                            "qa": qa,
                        }
                    )
        max_samples = self.eval_cfg.get("max_samples_per_split", None)
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        for index, sample in enumerate(samples):
            sample["index"] = index
        return samples

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
                    "content": [{"type": "text", "text": answer}],
                }
            )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )
        return text

    def _raw_text(self, question, image, answer=None):
        fields = self._raw_prompt_fields()
        system_prompt = fields["system_prompt"]
        question_start = fields["question_start"]
        answer_tag = fields["answer_tag"]
        image_token = self.eval_cfg.get("image_token", "<image>")
        image_part = f"{image_token}\n" if image is not None else ""
        text = f"{system_prompt}{question_start}{image_part}{question}{answer_tag}"
        if answer is not None:
            text += str(answer)
        return text

    def _build_text_and_images(self, sample, model, answer=None):
        image = self._load_image(sample["image_path"])
        style = self._effective_prompt_style(model)
        if style == "chat_template":
            text = self._chat_template_text(sample["question"], image, answer=answer)
        elif style == "raw":
            text = self._raw_text(sample["question"], image, answer=answer)
        else:
            raise ValueError(f"Unknown FIUBench prompt_style: {style}")
        return text, [image]

    def _prepare_inputs(self, text, images, tokenizer):
        processor = self._load_processor()
        if self._uses_upstream_llava_io():
            tokenizer_kwargs = {"return_tensors": "pt"}
            max_length = self.eval_cfg.get("max_length", None)
            if max_length is not None:
                tokenizer_kwargs.update(
                    {
                        "max_length": int(max_length),
                        "truncation": self.eval_cfg.get("truncation", True),
                    }
                )
            inputs = dict(tokenizer(text, **tokenizer_kwargs))
            if images:
                image_processor = getattr(processor, "image_processor", None)
                if image_processor is None:
                    raise ValueError(
                        "FIUBench upstream LLaVA mode requires processor.image_processor."
                    )
                image_input = images[0] if len(images) == 1 else images
                if hasattr(image_processor, "preprocess"):
                    inputs.update(
                        image_processor.preprocess(
                            image_input, return_tensors="pt"
                        )
                    )
                else:
                    inputs.update(image_processor(images=image_input, return_tensors="pt"))
            return inputs
        kwargs = {
            "text": text,
            "return_tensors": "pt",
        }
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

    def _make_labels(self, full_batch, prompt_batch, tokenizer, full_text=None):
        if self._uses_upstream_llava_io() and full_text is not None:
            # This is data_module.preprocess_v1 from the release. Its
            # token-count convention is part of FIUBench's reported loss.
            labels = full_batch["input_ids"].clone()
            labels[:, :1] = IGNORE_INDEX
            answer_tag = self._raw_prompt_fields()["answer_tag"]
            instruction = full_text.split(answer_tag)[0].strip(" ")
            instruction_len = len(tokenizer(instruction + answer_tag)["input_ids"]) - 2
            labels[:, 1 : 1 + max(0, instruction_len)] = IGNORE_INDEX
            return labels
        labels = full_batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
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

    def _token_stats(self, logits, labels):
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        valid = shifted_labels[0] != IGNORE_INDEX
        if valid.sum().item() == 0:
            return {}
        selected_logits = shifted_logits[0, valid]
        selected_labels = shifted_labels[0, valid]
        log_probs = F.log_softmax(selected_logits, dim=-1)
        probs = torch.exp(log_probs)
        token_log_probs = log_probs.gather(
            dim=-1, index=selected_labels.unsqueeze(-1)
        ).squeeze(-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        return {
            "token_log_probs": token_log_probs.detach().float().cpu(),
            "mu": mu.detach().float().cpu(),
            "sigma": sigma.detach().float().cpu(),
        }

    def _answer_loss(self, model, tokenizer, sample, answer, compute_token_stats=False):
        full_text, images = self._build_text_and_images(sample, model, answer=answer)
        prompt_text, prompt_images = self._build_text_and_images(sample, model, answer=None)
        full_batch = self._prepare_inputs(full_text, images, tokenizer)
        prompt_batch = self._prepare_inputs(prompt_text, prompt_images, tokenizer)
        labels = self._make_labels(full_batch, prompt_batch, tokenizer, full_text=full_text)
        num_tokens = (labels != IGNORE_INDEX).sum(-1)
        if num_tokens.sum().item() == 0:
            return None

        device = self._model_device(model)
        full_batch = self._move_to_device(full_batch, device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(**full_batch, labels=labels)

        if self._uses_upstream_llava_io():
            valid_labels = labels[labels != IGNORE_INDEX].unsqueeze(0)
            loss_logits = outputs.logits[:, -valid_labels.shape[1] :, :]
            loss = self._batch_loss(loss_logits, valid_labels)
        else:
            loss = self._batch_loss(outputs.logits, labels)
        result = {
            "prompt_text": prompt_text,
            "loss": float(loss[0].detach().cpu().item()),
            "avg_loss": float((loss[0] / num_tokens[0].to(loss.device)).detach().cpu().item()),
            "num_tokens": int(num_tokens[0].detach().cpu().item()),
        }
        if compute_token_stats:
            result.update(self._token_stats(outputs.logits, labels))
            result["zlib_text"] = str(answer)
        return result

    def _clean_answer(self, text):
        text = str(text)
        for marker in ("ASSISTANT:", "Assistant:", "assistant:", "Answer:", "### Answer:"):
            if marker in text:
                text = text.split(marker)[-1]
        text = text.strip()
        if self.eval_cfg.get("truncate_generation_at_period", True):
            period_index = text.find(".")
            if period_index == -1:
                if self.eval_cfg.get("empty_generation_without_period", True):
                    return ""
                return text
            text = text[: period_index + 1]
        return text.strip()

    def _generate(self, model, tokenizer, sample, answer=None):
        if self._uses_upstream_llava_io() and answer is not None:
            # Mirror evaluate_util.run_generation: decode the labelled example,
            # remove the answer, left-pad it again, and reuse its pixel values.
            full_text, images = self._build_text_and_images(
                sample, model, answer=answer
            )
            full_batch = self._prepare_inputs(full_text, images, tokenizer)
            decoded_input = tokenizer.batch_decode(full_batch["input_ids"])[0]
            answer_tag = self._raw_prompt_fields()["answer_tag"].replace("\n", "")
            prompt = decoded_input.split(answer_tag)[0].strip(" ") + answer_tag
            if (self._active_model_family or "").replace("_", "-") == "llava-phi":
                # This is the model-specific normalization in FIUBench's
                # run_generation helper after decoding the labelled prompt.
                question_start = self._raw_prompt_fields()["question_start"]
                prompt = prompt.replace(
                    question_start, f"{question_start} <image>"
                )
                prompt = prompt.replace("<|user|>", "<|user|>\n")
                prompt = prompt.replace("<|end|>", "<|end|>\n")

            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            generation_inputs = tokenizer.batch_encode_plus(
                [prompt], add_special_tokens=True, return_tensors="pt", padding=True
            )
            device = self._model_device(model)
            generation_inputs = self._move_to_device(generation_inputs, device)
            pixel_values = full_batch["pixel_values"].to(device)
            generation_args = self._to_container(
                self.eval_cfg.get("generation_args", {}), {}
            )
            generation_args.update(
                {
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": tokenizer.eos_token_id,
                }
            )
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=generation_inputs["input_ids"],
                    attention_mask=generation_inputs["attention_mask"],
                    pixel_values=pixel_values,
                    **generation_args,
                )
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            generated_ids = sequences[:, generation_inputs["input_ids"].shape[-1] :]
            generated = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            # This exact slice is used in the upstream run_generation helper.
            return prompt, generated[: generated.find(".") + 1]

        prompt, images = self._build_text_and_images(sample, model, answer=None)
        inputs = self._prepare_inputs(prompt, images, tokenizer)
        device = self._model_device(model)
        inputs = self._move_to_device(inputs, device)
        input_len = inputs["input_ids"].shape[-1]
        generation_args = self._to_container(self.eval_cfg.get("generation_args", {}), {})
        if "pad_token_id" not in generation_args and getattr(tokenizer, "eos_token_id", None) is not None:
            generation_args["pad_token_id"] = tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        generated_ids = sequences[:, input_len:] if sequences.shape[-1] > input_len else sequences
        processor = self._load_processor()
        decoder = processor.decode if hasattr(processor, "decode") else tokenizer.decode
        generated = decoder(generated_ids[0], skip_special_tokens=True)
        return prompt, self._clean_answer(generated)

    def _exact_match(self, prediction, keywords):
        value = self._keyword_scorer.score(prediction, keywords).value
        return 0.0 if value is None else float(value)

    def _rouge_recall(self, generated_outputs, ground_truths, indices):
        rouge1_recall = {}
        rouge_l_recall = {}
        for generated, ground_truth, index in zip(
            generated_outputs, ground_truths, indices
        ):
            rouge_text = str(generated)
            if "." in rouge_text:
                rouge_text = rouge_text[: rouge_text.find(".")]
            scores = self._rouge_scorer.score(rouge_text, str(ground_truth)).details
            rouge1_recall[str(index)] = scores["rouge1"]
            rouge_l_recall[str(index)] = scores["rougeL"]
        return {"rouge1_recall": rouge1_recall, "rougeL_recall": rouge_l_recall}

    def _min_k_scores(self, loss_result):
        if "token_log_probs" not in loss_result:
            return None
        min_k = self._min_k_scorer.score(loss_result["token_log_probs"])
        min_k_plus_plus = self._min_k_plus_plus_scorer.score(
            loss_result["token_log_probs"],
            loss_result["mu"],
            loss_result["sigma"],
        )
        return min_k.value, min_k_plus_plus.value

    def _evaluate_perturbation_ratio(
        self, samples, model, tokenizer, base_answer_key, perturbed_answer_key
    ):
        logs = {
            "average_perturb_loss": {},
            "avg_paraphrased_loss": {},
            "truth_ratio": {},
            "paraphrased_loss": {},
            "perturb_loss": {},
            "num_token_paraphrased": {},
            "num_token_perturb": {},
        }
        for sample in samples:
            index = str(sample["index"])
            qa = sample["qa"]
            base_answers = self._qa_values(qa, base_answer_key, fallback_key="answer")
            perturb_answers = self._qa_values(qa, perturbed_answer_key)
            if not base_answers or not perturb_answers:
                logger.warning(
                    "Skipping perturbation metric for sample %s due to missing answers.",
                    index,
                )
                continue

            base_loss = self._answer_loss(model, tokenizer, sample, base_answers[0])
            if base_loss is None:
                logger.warning(
                    "Skipping perturbation metric for sample %s due to empty base labels.",
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
                    "Skipping perturbation metric for sample %s due to empty perturb labels.",
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

    def _evaluate_ape(self, samples, model, tokenizer, answer_key):
        logs = {"exact_match": []}
        if self.eval_cfg.get("save_generated_text", False):
            logs["generated_text"] = {}
        for sample in samples:
            answers = self._qa_values(sample["qa"], answer_key, fallback_key="answer")
            if not answers:
                continue
            prompt, generated = self._generate(
                model, tokenizer, sample, answer=answers[0]
            )
            score = self._exact_match(generated.lower(), sample["qa"].get("keywords", []))
            logs["exact_match"].append(score)
            if self.eval_cfg.get("save_generated_text", False):
                logs["generated_text"][str(sample["index"])] = [
                    prompt,
                    generated,
                    answers[0],
                    sample["category"],
                ]
        return logs

    def _evaluate_standard(
        self,
        samples,
        model,
        tokenizer,
        metrics,
        answer_key,
        base_answer_key,
        perturbed_answer_key,
        normalize_gt=False,
    ):
        logs = self._evaluate_perturbation_ratio(
            samples, model, tokenizer, base_answer_key, perturbed_answer_key
        )
        logs.update(
            {
                "avg_gt_loss": {},
                "gt_loss": {},
                "num_token_gt": {},
                "generated_text": {},
            }
        )
        if "mink" in metrics:
            logs["mink"] = []
            logs["mink++"] = []
            logs["loss"] = []
            logs["zlib"] = []
        if "exact_match" in metrics:
            logs["exact_match"] = []
        if "gpt" in metrics:
            logs["gpt"] = []

        generated_outputs = []
        ground_truths = []
        indices = []
        compute_mink = "mink" in metrics
        for sample in samples:
            index = str(sample["index"])
            answers = self._qa_values(sample["qa"], answer_key, fallback_key="answer")
            if not answers:
                logger.warning("Skipping FIUBench sample %s due to missing answer.", index)
                continue
            answer = answers[0]
            loss_result = self._answer_loss(
                model, tokenizer, sample, answer, compute_token_stats=compute_mink
            )
            if loss_result is None:
                logger.warning("Skipping FIUBench sample %s due to empty labels.", index)
                continue
            prompt, generated = self._generate(model, tokenizer, sample, answer=answer)

            logs["avg_gt_loss"][index] = loss_result["avg_loss"]
            logs["gt_loss"][index] = loss_result["loss"]
            logs["num_token_gt"][index] = loss_result["num_tokens"]
            logs["generated_text"][index] = [
                prompt,
                generated,
                answer,
                sample["category"],
            ]
            generated_outputs.append(generated)
            ground_truths.append(answer)
            indices.append(index)

            if "mink" in metrics:
                scores = self._min_k_scores(loss_result)
                if scores is None:
                    logs["mink"].append(0.0)
                    logs["mink++"].append(0.0)
                else:
                    mink, mink_pp = scores
                    logs["mink"].append(mink)
                    logs["mink++"].append(mink_pp)
                logs["loss"].append(
                    self._loss_mia_scorer.score(loss_result["avg_loss"]).value
                )
                logs["zlib"].append(
                    self._zlib_mia_scorer.score(loss_result["avg_loss"], answer).value
                )

            if "exact_match" in metrics:
                logs["exact_match"].append(
                    self._exact_match(generated.lower(), sample["qa"].get("keywords", []))
                )

            if "gpt" in metrics:
                if not self.eval_cfg.get("enable_gpt", False) and not self._warned_gpt:
                    logger.warning(
                        "FIUBench GPT metric requested, but enable_gpt=false; writing 0.0 scores."
                    )
                    self._warned_gpt = True
                logs["gpt"].append(0.0)

        logs.update(self._rouge_recall(generated_outputs, ground_truths, indices))
        if normalize_gt:
            self._normalize_gt_loss(logs)
        return logs

    def _normalize_metrics(self, metric_list):
        metrics = []
        for metric in self._as_list(metric_list):
            metric = str(metric).lower()
            if metric in ("exact", "match", "em"):
                metric = "exact_match"
            metrics.append(metric)
        return metrics

    def _task_specs(self):
        split_list = self._to_container(self.eval_cfg.get("split_list", None), None)
        if split_list is None:
            split_list = self._to_container(self.eval_cfg.get("splits", None), None)
        if split_list is None:
            ratio = int(self.eval_cfg.get("ratio", 5))
            forget_split = self.eval_cfg.get("forget_split", None) or f"forget{ratio}"
            retain_split = self.eval_cfg.get("retain_split", None) or f"retain{ratio}"
            split_list = [forget_split, retain_split]
        if not split_list:
            raise ValueError("eval.fiubench.split_list must contain at least one split.")

        specs = []
        for index, split in enumerate(split_list):
            data_path = self._value_at(self.eval_cfg.get("data_path"), index)
            if data_path is None:
                data_path = str(Path(self.eval_cfg.data_root) / "full.json")
            specs.append(
                {
                    "split": str(split),
                    "data_path": data_path,
                    "question_key": self._value_at(
                        self.eval_cfg.get("question_key"), index, "question"
                    ),
                    "robust_question_key": self._value_at(
                        self.eval_cfg.get("robust_question_key"), index, "paraphrased_question"
                    ),
                    "answer_key": self._value_at(
                        self.eval_cfg.get("answer_key"), index, "answer"
                    ),
                    "base_answer_key": self._value_at(
                        self.eval_cfg.get("base_answer_key"), index, "paraphrased_answer"
                    ),
                    "perturbed_answer_key": self._value_at(
                        self.eval_cfg.get("perturbed_answer_key"), index, "perturbed_answer"
                    ),
                    "eval_task": self._value_at(
                        self.eval_cfg.get("eval_task"), index, f"eval_{split}_log"
                    ),
                    "metrics": self._normalize_metrics(
                        self._value_at(self.eval_cfg.get("robust_eval"), index, ["rouge"])
                    ),
                }
            )
        return specs

    def _evaluate_split(self, spec, model, tokenizer):
        metrics = spec["metrics"]
        if "ape" in metrics and self.eval_cfg.get("ape_only", True):
            samples = self._samples_for_split(
                spec["data_path"], spec["split"], spec["robust_question_key"]
            )
            logger.info(
                "Running FIUBench APE split=%s samples=%s", spec["split"], len(samples)
            )
            return self._evaluate_ape(samples, model, tokenizer, spec["answer_key"])

        samples = self._samples_for_split(
            spec["data_path"], spec["split"], spec["question_key"]
        )
        logger.info("Running FIUBench split=%s samples=%s", spec["split"], len(samples))
        normalize_gt = self.eval_cfg.get("normalize_gt", None)
        if normalize_gt is None:
            normalize_gt = "eval_retain_log" not in str(spec["eval_task"])
        return self._evaluate_standard(
            samples,
            model,
            tokenizer,
            metrics,
            spec["answer_key"],
            spec["base_answer_key"],
            spec["perturbed_answer_key"],
            normalize_gt=normalize_gt,
        )

    def _flatten_numeric(self, value):
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            values = [value]
        flattened = []
        for item in values:
            if isinstance(item, (list, tuple)):
                flattened.extend(float(x) for x in item if isinstance(x, (int, float)))
            elif isinstance(item, (int, float)):
                flattened.append(float(item))
        return [x for x in flattened if not math.isnan(x)]

    def _upstream_values(self, task_logs, key):
        value = task_logs.get(key, {})
        if isinstance(value, dict):
            value = list(value.values())
        return np.asarray(value, dtype=float)

    def _upstream_model_utility(self, logs):
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
                [
                    self._answer_probability_scorer.score(loss).value
                    for loss in true_losses
                ],
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

    def _upstream_forget_quality(self, logs, retained_logs):
        """Reproduce FIUBench's aggregate_eval_stat.get_forget_quality."""
        task_key = "eval_forget_log.json"
        if task_key not in logs:
            return {}
        unlearned = logs[task_key]
        retained = retained_logs.get(task_key, retained_logs)
        required = {"avg_paraphrased_loss", "average_perturb_loss"}
        if not required.issubset(unlearned) or not required.issubset(retained):
            return {}

        output = {}
        for source_key, output_key in (
            ("mink", "Mink"),
            ("mink++", "Mink++"),
            ("exact_match", "Exact Match"),
        ):
            values = unlearned.get(source_key)
            if values:
                output[output_key] = float(np.mean(values))

        unlearned_paraphrase = self._upstream_values(unlearned, "avg_paraphrased_loss")
        unlearned_perturbed = self._upstream_values(unlearned, "average_perturb_loss")
        retained_paraphrase = self._upstream_values(retained, "avg_paraphrased_loss")
        retained_perturbed = self._upstream_values(retained, "average_perturb_loss")
        if (
            unlearned_perturbed.ndim != 2
            or retained_perturbed.ndim != 2
            or not unlearned_paraphrase.size
            or not retained_paraphrase.size
        ):
            return output
        unlearned_truth = np.exp(
            unlearned_perturbed.mean(axis=-1) - unlearned_paraphrase
        )
        retained_truth = np.exp(retained_perturbed.mean(axis=-1) - retained_paraphrase)
        try:
            ks_result = self._ks_scorer.score(unlearned_truth, retained_truth)
        except ImportError:
            logger.warning("scipy is not installed; skipping FIUBench KS metrics.")
            return output
        output.update(
            {
                "Forget Quality": float(ks_result.details["pvalue"]),
                "KS Test PVal Forget": float(ks_result.details["pvalue"]),
                "KS Test Forget": float(ks_result.details["statistic"]),
            }
        )
        return output

    def _load_retain_result(self):
        path = self.eval_cfg.get("retain_result", None)
        if path in (None, "", "null", "None"):
            return None
        with open(path, "r", encoding="utf-8") as handle:
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
            if (
                "avg_paraphrased_loss" in task_logs
                and "average_perturb_loss" in task_logs
            ):
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
        summary.update(self._upstream_model_utility(logs))
        retained_logs = self._load_retain_result()
        if retained_logs is not None:
            summary.update(self._upstream_forget_quality(logs, retained_logs))
        return summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        model = self.prepare_model(model)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("FIUBenchEvaluator requires a tokenizer.")
        self._load_processor()

        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        with open_dict(self.eval_cfg):
            self.eval_cfg.output_dir = output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        self._active_model_family = self._model_family(model)
        self._active_prompt_style = self._effective_prompt_style(model)

        logger.info("***** Running %s evaluation suite *****", self.name)
        logger.info("Fine-grained evaluations will be saved to: %s", logs_file_path)
        logger.info("Aggregated evaluations will be summarised in: %s", summary_file_path)

        for spec in self._task_specs():
            task_key = f"{spec['eval_task']}.json"
            split_file_path = os.path.join(
                output_dir, f"{spec['split']}_{spec['eval_task']}.json"
            )
            if not overwrite and task_key in logs and logs[task_key]:
                logger.info("Skipping %s, already evaluated.", task_key)
                continue
            if os.path.exists(split_file_path) and not overwrite:
                logger.info("Loading existing FIUBench split log from %s", split_file_path)
                with open(split_file_path, "r", encoding="utf-8") as handle:
                    split_logs = json.load(handle)
            else:
                split_logs = self._evaluate_split(spec, model, tokenizer)
                os.makedirs(output_dir, exist_ok=True)
                with open(split_file_path, "w", encoding="utf-8") as handle:
                    json.dump(split_logs, handle, indent=4)
            logs[task_key] = split_logs
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        aggregate_path = os.path.join(output_dir, "FIUBENCH_eval_log_aggregated.json")
        with open(aggregate_path, "w", encoding="utf-8") as handle:
            json.dump(logs, handle, indent=4)

        summary = self.summarize(logs)
        self.save_logs(summary, summary_file_path)
        return summary
