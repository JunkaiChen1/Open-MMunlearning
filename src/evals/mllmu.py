import json
import logging
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf import open_dict
from PIL import Image
from transformers import AutoProcessor

from evals.base import Evaluator
from evals.scorers import (
    AnswerProbabilityScorer,
    BleuScorer,
    ContainsScorer,
    JensenShannonDistanceScorer,
    LossMIAScorer,
    PerturbationTruthRatioScorer,
    RougeScorer,
    StandardMinKPlusPlusScorer,
    StandardMinKScorer,
    TwoSampleKSScorer,
    ZlibMIAScorer,
    token_log_probability_statistics,
)


logger = logging.getLogger("evaluator")
IGNORE_INDEX = -100


class MLLMUBenchEvaluator(Evaluator):
    """Evaluator for the original MLLMU-Bench protocol.

    This mirrors the benchmark's `eval.py` flow inside OpenUnlearning's evaluator
    registry: four data splits are evaluated on classification, fill-in-the-blank,
    and generation tasks, with aggregate summaries written through the standard
    `*_EVAL.json` and `*_SUMMARY.json` files.
    """

    def __init__(self, eval_cfg, **kwargs):
        self.name = "MLLMU"
        self.eval_cfg = eval_cfg
        self.processor = None
        self._few_shot_df = None
        self._active_model_family = None
        self._prompt_style = None
        self.rng = random.Random(int(self.eval_cfg.get("seed", 0)))
        self._bleu_scorer = BleuScorer()
        # The upstream fill-in-the-blank check is
        # ``ground_truth.lower() in assistant_response.lower()``.
        self._contains_scorer = ContainsScorer(normalization="lower")
        mia_sign = float(self.eval_cfg.get("generation_mia_sign", 1.0))
        self._loss_mia_scorer = LossMIAScorer(sign=mia_sign)
        self._zlib_mia_scorer = ZlibMIAScorer(sign=mia_sign)
        min_k_ratio = float(self.eval_cfg.get("generation_mia_min_k_ratio", 0.2))
        self._standard_min_k_scorer = StandardMinKScorer(ratio=min_k_ratio)
        self._standard_min_k_plus_plus_scorer = StandardMinKPlusPlusScorer(
            ratio=min_k_ratio
        )
        self._generation_rouge_scorer = RougeScorer(
            ("rouge1", "rouge2", "rougeL"),
            aggregation="fmeasure",
        )
        self._answer_probability_scorer = AnswerProbabilityScorer()
        self._truth_ratio_scorer = PerturbationTruthRatioScorer()
        self._ks_scorer = TwoSampleKSScorer()
        self._js_scorer = JensenShannonDistanceScorer(
            alignment="histogram",
            bins=int(self.eval_cfg.get("generation_js_bins", 50)),
        )
        self._truth_ratio_variants = None

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
            raise ValueError("eval.mllmu.processor_path must be set.")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        return self.processor

    def _model_family(self, model):
        configured = self.eval_cfg.get("model_family", None)
        if configured:
            configured = str(configured).lower()
            if configured.startswith("huggingfacem4") or "idefics" in configured:
                return "idefics"
            if "llava" in configured:
                return "llava"
            return configured
        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        class_name = model.__class__.__name__.lower()
        text = f"{model_type} {class_name}"
        if "llava" in text:
            return "llava"
        if "idefics" in text:
            return "idefics"
        if "qwen" in text:
            return "qwen"
        if "gemma" in text:
            return "gemma"
        return "default"

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
        path = Path(path)
        if path.is_dir():
            parquet_files = sorted(path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {path}")
            return pd.concat(
                [pd.read_parquet(file) for file in parquet_files], ignore_index=True
            )
        return pd.read_parquet(path)

    def _load_truth_ratio_variants(self):
        """Load the GPT-generated answer pool used by MLLMU Truth Ratio."""
        if self._truth_ratio_variants is not None:
            return self._truth_ratio_variants
        path = self.eval_cfg.get("generation_truth_ratio_path", None)
        if not path:
            self._truth_ratio_variants = {}
            return self._truth_ratio_variants
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                "MLLMU Truth Ratio sidecar does not exist: "
                f"{path}. Generate it with scripts/build_mllmu_generation_truth_ratio.py."
            )
        variants = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid Truth Ratio sidecar JSON at line {line_number}: {path}"
                    ) from exc
                key = (
                    str(item.get("id")),
                    int(item.get("question_index", -1)),
                    str(item.get("modality", "")),
                )
                if key in variants:
                    raise ValueError(f"Duplicate Truth Ratio sidecar key: {key}")
                paraphrase = str(item.get("paraphrased_answer", "")).strip()
                perturbations = item.get("perturbed_answers")
                if not paraphrase or not isinstance(perturbations, list):
                    raise ValueError(
                        f"Incomplete Truth Ratio sidecar record at line {line_number}"
                    )
                if not all(
                    isinstance(answer, str) and answer.strip()
                    for answer in perturbations
                ):
                    raise ValueError(
                        f"Invalid perturbed answers at line {line_number}: {key}"
                    )
                variants[key] = {
                    "paraphrased_answer": paraphrase,
                    "perturbed_answers": [answer.strip() for answer in perturbations],
                }
        self._truth_ratio_variants = variants
        logger.info(
            "Loaded %d MLLMU Truth Ratio sidecar records from %s", len(variants), path
        )
        return variants

    def _load_few_shot_df(self):
        if self._few_shot_df is not None:
            return self._few_shot_df
        few_shot_path = self.eval_cfg.get("few_shot_path", None)
        if not few_shot_path:
            self._few_shot_df = pd.DataFrame()
        else:
            self._few_shot_df = self._read_parquet(few_shot_path)
        return self._few_shot_df

    def _as_list(self, value):
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _load_image(self, image_value):
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            if image_value.get("path") is not None:
                return Image.open(image_value["path"]).convert("RGB")
        if isinstance(image_value, (bytes, bytearray)):
            return Image.open(BytesIO(image_value)).convert("RGB")
        if isinstance(image_value, str):
            return Image.open(image_value).convert("RGB")
        raise ValueError(f"Unsupported image value: {type(image_value)}")

    def _row_image(self, row, mode):
        if mode == "test" and "images" in row and row["images"] is not None:
            images = self._as_list(row["images"])
            if not images:
                raise ValueError(f"No images found for test sample {row.get('ID')}")
            strategy = self.eval_cfg.get("test_image_strategy", "random")
            image_value = images[0] if strategy == "first" else self.rng.choice(images)
            return self._load_image(image_value)
        return self._load_image(row["image"])

    def _classification_questions(self, classification_task, key):
        if not isinstance(classification_task, dict):
            return []
        return self._as_list(classification_task.get(key, []))

    def _format_options(self, options):
        return "\n".join([f"{key}: {value}" for key, value in options.items()])

    def _question_with_options(self, question, options):
        return f"{question}\n{self._format_options(options)}"

    def _blank_question(self, question):
        return (
            question.replace("__", "[Blank]")
            + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
        )

    def _use_few_shot(self):
        """Return whether MLLMU demonstration examples should be injected."""
        value = self.eval_cfg.get("use_few_shot", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _few_shot_count(self, task_name, family):
        if not self._use_few_shot():
            return 0
        cfg = self._to_container(self.eval_cfg.get("few_shot", {}), {})
        task_cfg = cfg.get(task_name, {}) if isinstance(cfg, dict) else {}
        if not isinstance(task_cfg, dict):
            return int(task_cfg)
        return int(task_cfg.get(family, task_cfg.get("default", 0)))

    def _select_few_shot_ids(self, ids, count):
        ids = [str(value) for value in ids]
        if count <= 0 or not ids:
            return []
        return self.rng.sample(ids, min(count, len(ids)))

    def _few_shots_for_classification(self, id_list, family):
        if not self._use_few_shot():
            return [], [], {}
        few_shot_df = self._load_few_shot_df()
        selected_ids = self._select_few_shot_ids(
            id_list, self._few_shot_count("classification", family)
        )
        if few_shot_df.empty or not selected_ids:
            return [], [], {}

        image_shots, text_shots = [], []
        skipped_indices = {}
        samples = few_shot_df[few_shot_df["ID"].astype(str).isin(selected_ids)]
        for _, row in samples.iterrows():
            row_id = str(row["ID"])
            image = self._row_image(row, mode="few_shot")
            skipped_indices[row_id] = {"image_textual": [], "pure_text": []}
            task = row["Classification_Task"]
            for idx, question_data in enumerate(
                self._classification_questions(task, "Image_Textual_Questions")
            ):
                image_shots.append(
                    {
                        "text": "Question: "
                        + self._question_with_options(
                            question_data["Question"], question_data["Options"]
                        ),
                        "answer": question_data["Correct_Answer"],
                        "image": image,
                    }
                )
                skipped_indices[row_id]["image_textual"].append(idx)
            for idx, question_data in enumerate(
                self._classification_questions(task, "Pure_Text_Questions")
            ):
                text_shots.append(
                    {
                        "text": "Question: "
                        + self._question_with_options(
                            question_data["Question"], question_data["Options"]
                        ),
                        "answer": question_data["Correct_Answer"],
                        "image": None,
                    }
                )
                skipped_indices[row_id]["pure_text"].append(idx)
        return image_shots, text_shots, skipped_indices

    def _few_shots_for_blank(self, id_list, family):
        if not self._use_few_shot():
            return [], [], {}
        few_shot_df = self._load_few_shot_df()
        selected_ids = self._select_few_shot_ids(
            id_list, self._few_shot_count("fill_in_the_blank", family)
        )
        if few_shot_df.empty or not selected_ids:
            return [], [], {}

        image_shots, text_shots = [], []
        skipped_indices = {}
        samples = few_shot_df[few_shot_df["ID"].astype(str).isin(selected_ids)]
        for _, row in samples.iterrows():
            row_id = str(row["ID"])
            image = self._row_image(row, mode="few_shot")
            skipped_indices[row_id] = {"image_textual": [], "pure_text": []}
            for idx, question_data in enumerate(self._as_list(row["Mask_Task"])):
                question_type = question_data["Type"]
                shot = {
                    "text": self._blank_question(question_data["Question"]),
                    "answer": question_data["Ground_Truth"],
                    "image": image if question_type == "Image_Textual" else None,
                    # The original fill prompt spells this one prefix without
                    # a space: ``USER:<image>``.
                    "raw_image_prefix": "USER:<image>\n",
                }
                if question_type == "Image_Textual":
                    image_shots.append(shot)
                    skipped_indices[row_id]["image_textual"].append(idx)
                elif question_type == "Pure_Text":
                    text_shots.append(shot)
                    skipped_indices[row_id]["pure_text"].append(idx)
        return image_shots, text_shots, skipped_indices

    def _raw_prompt(
        self,
        current_text,
        current_image,
        few_shots,
        assistant_prefix="ASSISTANT:",
        text_user_prefix="USER:\n",
        answer=None,
    ):
        parts = []
        images = []
        for shot in few_shots:
            if shot.get("image") is not None:
                prefix = shot.get("raw_image_prefix", "USER: <image>\n")
                parts.append(
                    f"{prefix}{shot['text']}\nCorrect Answer: {shot['answer']}\n"
                )
                images.append(shot["image"])
            else:
                parts.append(
                    f"USER:\n{shot['text']}\nCorrect Answer: {shot['answer']}\n"
                )
        if current_image is not None:
            parts.append(f"USER: <image>\n{current_text}")
            images.append(current_image)
        else:
            parts.append(f"{text_user_prefix}{current_text}")
        if assistant_prefix:
            parts.append(f"\n{assistant_prefix}")
        if answer is not None:
            parts.append(str(answer))
        return "".join(parts), images

    def _chat_template_prompt(
        self, current_text, current_image, few_shots, answer=None
    ):
        processor = self._load_processor()
        messages = []
        system_prompt = self.eval_cfg.get("system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        images = []
        for shot in few_shots:
            content = []
            if shot.get("image") is not None:
                content.append({"type": "image"})
                images.append(shot["image"])
            content.append({"type": "text", "text": shot["text"]})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(shot["answer"])}],
                }
            )

        content = []
        if current_image is not None:
            content.append({"type": "image"})
            images.append(current_image)
        content.append({"type": "text", "text": current_text})
        messages.append({"role": "user", "content": content})
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer)}],
                }
            )

        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None
        )
        return prompt, images

    def _effective_prompt_style(self, model):
        style = str(self.eval_cfg.get("prompt_style", "auto")).lower()
        if style != "auto":
            return style
        family = self._model_family(model)
        return "raw" if family.startswith(("llava", "idefics")) else "chat_template"

    def _build_prompt(
        self,
        current_text,
        current_image=None,
        few_shots=None,
        text_user_prefix="USER:\n",
        raw_assistant_prefix="ASSISTANT:",
        answer=None,
    ):
        few_shots = few_shots or []
        prompt_style = getattr(self, "_prompt_style", None)
        if prompt_style is None:
            prompt_style = str(self.eval_cfg.get("prompt_style", "auto")).lower()
            if prompt_style == "auto":
                prompt_style = "chat_template"
        if prompt_style == "raw":
            return self._raw_prompt(
                current_text,
                current_image,
                few_shots,
                assistant_prefix=raw_assistant_prefix,
                text_user_prefix=text_user_prefix,
                answer=answer,
            )
        try:
            return self._chat_template_prompt(
                current_text, current_image, few_shots, answer=answer
            )
        except Exception as exc:
            if prompt_style == "chat_template":
                raise
            logger.warning("Falling back to raw MLLMU prompt: %s", exc)
            return self._raw_prompt(
                current_text,
                current_image,
                few_shots,
                assistant_prefix=raw_assistant_prefix,
                text_user_prefix=text_user_prefix,
                answer=answer,
            )

    def _processor_images(self, images):
        processor = self._load_processor()
        if processor.__class__.__name__ == "Gemma3Processor":
            return [[image] for image in images]
        return images

    def _prepare_inputs(
        self, prompt, images, tokenizer, input_mode="auto", single_image=False
    ):
        processor = self._load_processor()
        kwargs = {
            "text": prompt,
            "return_tensors": "pt",
        }
        if input_mode == "tokenizer":
            return tokenizer(prompt, return_tensors="pt")
        if input_mode == "processor":
            # MLLMU-Bench passes ``images=None`` for its pure-text fill task.
            # Keep that call shape instead of relying on a processor fallback.
            processed_images = self._processor_images(images) if images else None
            if single_image and processed_images:
                processed_images = processed_images[0]
            kwargs["images"] = processed_images
            return processor(**kwargs)
        if images:
            kwargs["images"] = self._processor_images(images)
        try:
            return processor(**kwargs)
        except Exception:
            if images:
                raise
            return tokenizer(prompt, return_tensors="pt")

    def _attention_mask(self, batch, tokenizer):
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            return attention_mask
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return torch.ones_like(batch["input_ids"])
        return (batch["input_ids"] != pad_token_id).long()

    def _make_answer_labels(self, full_batch, prompt_batch, tokenizer):
        labels = full_batch["input_ids"].clone()
        full_attention = self._attention_mask(full_batch, tokenizer)
        prompt_attention = self._attention_mask(prompt_batch, tokenizer)
        if labels.shape[0] != prompt_attention.shape[0]:
            raise ValueError("MLLMU prompt and answer batches must have equal sizes.")

        padding_side = getattr(tokenizer, "padding_side", "right")
        for row_index in range(labels.shape[0]):
            prompt_length = int(prompt_attention[row_index].sum().item())
            if padding_side == "left":
                full_length = int(full_attention[row_index].sum().item())
                full_start = max(0, labels.shape[1] - full_length)
                prompt_end = min(labels.shape[1], full_start + prompt_length)
                labels[row_index, :prompt_end] = IGNORE_INDEX
            else:
                labels[row_index, : min(prompt_length, labels.shape[1])] = IGNORE_INDEX
        labels[full_attention == 0] = IGNORE_INDEX
        return labels

    def _align_logits_to_labels(self, input_ids, logits, labels, model):
        """Align VLM-expanded logits to the processor's text-token labels."""
        target_length = labels.shape[-1]
        if logits.shape[-2] == target_length:
            return logits
        if logits.shape[-2] < target_length:
            raise ValueError(
                "MLLMU logits are shorter than their labels: "
                f"{logits.shape[-2]} < {target_length}."
            )

        config = getattr(model, "config", None)
        image_token_ids = {
            value
            for value in (
                getattr(config, "image_token_id", None),
                getattr(config, "image_token_index", None),
            )
            if value is not None
        }
        for image_token_id in image_token_ids:
            aligned_rows = []
            found_image_token = False
            for row_index in range(input_ids.shape[0]):
                positions = (input_ids[row_index] == image_token_id).nonzero(
                    as_tuple=True
                )[0]
                if len(positions) == 0:
                    aligned = logits[row_index, -target_length:, :]
                else:
                    found_image_token = True
                    start = int(positions[0].item())
                    right_length = target_length - start
                    aligned = torch.cat(
                        [
                            logits[row_index, :start, :],
                            logits[row_index, -right_length:, :],
                        ],
                        dim=0,
                    )
                aligned_rows.append(aligned)
            if found_image_token:
                return torch.stack(aligned_rows)

        # With visual expansion before the assistant answer, right alignment
        # preserves every labelled answer position even when the model does not
        # expose its image token id in config.
        return logits[:, -target_length:, :]

    def _answer_loss(
        self,
        model,
        tokenizer,
        prompt,
        full_prompt,
        images,
        *,
        input_mode="auto",
        single_image=False,
        compute_token_stats=False,
    ):
        full_batch = self._prepare_inputs(
            full_prompt,
            images,
            tokenizer,
            input_mode=input_mode,
            single_image=single_image,
        )
        prompt_batch = self._prepare_inputs(
            prompt,
            images,
            tokenizer,
            input_mode=input_mode,
            single_image=single_image,
        )
        labels = self._make_answer_labels(full_batch, prompt_batch, tokenizer)
        num_tokens = (labels[..., 1:] != IGNORE_INDEX).sum(-1)
        if num_tokens.sum().item() == 0:
            return None

        device = self._model_device(model)
        full_batch = self._move_to_device(full_batch, device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(**full_batch, labels=labels)
        average_loss = getattr(outputs, "loss", None)
        if average_loss is None:
            raise ValueError("MLLMU generation MIA requires model outputs.loss.")
        if average_loss.numel() != 1:
            average_loss = average_loss.mean()

        average_loss = float(average_loss.detach().cpu().item())
        token_count = int(num_tokens[0].detach().cpu().item())
        result = {
            "loss": float(average_loss * token_count),
            "avg_loss": average_loss,
            "num_tokens": token_count,
        }
        if compute_token_stats:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise ValueError("MLLMU Min-K MIA requires model outputs.logits.")
            aligned_logits = self._align_logits_to_labels(
                full_batch["input_ids"], logits, labels, model
            )
            result.update(token_log_probability_statistics(aligned_logits, labels))
        return result

    def _uses_upstream_raw_generation(self):
        family = (
            self._active_model_family
            or str(self.eval_cfg.get("model_family", "")).lower()
        )
        return bool(
            self.eval_cfg.get("upstream_compatibility", True)
            and self._prompt_style == "raw"
            and family.startswith(("llava", "idefics"))
        )

    def _generate(
        self,
        model,
        tokenizer,
        prompt,
        images,
        *,
        allow_answer_marker=True,
        max_new_tokens=None,
        input_mode="auto",
        decode_mode="processor",
        single_image=False,
    ):
        device = self._model_device(model)
        inputs = self._prepare_inputs(
            prompt,
            images,
            tokenizer,
            input_mode=input_mode,
            single_image=single_image,
        )
        inputs = self._move_to_device(inputs, device)
        input_len = inputs["input_ids"].shape[-1]
        generation_args = self._to_container(
            self.eval_cfg.get("generation_args", {}), {}
        )
        if max_new_tokens is not None:
            generation_args["max_new_tokens"] = int(max_new_tokens)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_args)
        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        processor = self._load_processor()
        decoder = (
            tokenizer.decode
            if decode_mode == "tokenizer"
            else (
                processor.decode if hasattr(processor, "decode") else tokenizer.decode
            )
        )
        if self._uses_upstream_raw_generation():
            # MLLMU-Bench decodes ``outputs[0][2:]`` and then extracts the
            # assistant segment. Keep this literal compatibility path for its
            # supported LLaVA/Idefics models.
            text = decoder(sequences[0][2:], skip_special_tokens=True)
        else:
            generated_ids = (
                sequences[:, input_len:]
                if sequences.shape[-1] > input_len
                else sequences
            )
            text = decoder(generated_ids[0], skip_special_tokens=True)
        return self._clean_answer(text, allow_answer_marker=allow_answer_marker)

    def _clean_answer(self, text, *, allow_answer_marker=True):
        text = str(text)
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[1].strip()
        if allow_answer_marker and "Answer:" in text:
            return text.split("Answer:", 1)[1].strip()
        return text.strip()

    def _extract_choice(self, answer, options):
        answer = str(answer).strip()
        if not answer:
            return None
        strategy = self.eval_cfg.get("choice_parse_strategy", "first_char")
        option_keys = {str(key).upper() for key in options.keys()}
        if strategy == "first_valid":
            for char in answer.upper():
                if char in option_keys:
                    return char
            return None
        first = answer[0].upper()
        return first if first in option_keys else None

    def _bleu(self, ground_truth, predicted_answer):
        return self._bleu_scorer.score(predicted_answer, ground_truth).value

    def _id_list_for_split(self, df, mode, forget_df=None):
        if mode == "test" and forget_df is not None:
            return [str(value) for value in forget_df["ID"].unique().tolist()]
        return [str(value) for value in df["ID"].unique().tolist()]

    def _filtered_samples(self, df, id_list, mode):
        samples = df[df["ID"].astype(str).isin(id_list)]
        max_samples = self.eval_cfg.get("max_samples_per_split", None)
        if max_samples:
            samples = samples.head(int(max_samples))
        return samples

    def _evaluate_classification(self, df, id_list, model, tokenizer, family, mode):
        image_shots, text_shots, skipped = self._few_shots_for_classification(
            id_list, family
        )
        # Upstream classification uses demonstrations for these three splits,
        # while still skipping any sampled questions in every split.
        include_shots = mode in ("forget", "retain_shared", "test")
        prompt_image_shots = image_shots if include_shots else []
        prompt_text_shots = text_shots if include_shots else []
        counts = {
            "image_textual_correct": 0,
            "image_textual_total": 0,
            "pure_text_correct": 0,
            "pure_text_total": 0,
        }
        details = []
        for _, row in df.iterrows():
            row_id = str(row["ID"])
            image = self._row_image(row, mode)
            task = row["Classification_Task"]
            for idx, question_data in enumerate(
                self._classification_questions(task, "Image_Textual_Questions")
            ):
                if idx in skipped.get(row_id, {}).get("image_textual", []):
                    continue
                text = (
                    self._question_with_options(
                        question_data["Question"], question_data["Options"]
                    )
                    + "\nJust give ONE letter representing the answer directly."
                )
                prompt, images = self._build_prompt(text, image, prompt_image_shots)
                generated = self._generate(
                    model,
                    tokenizer,
                    prompt,
                    images,
                    allow_answer_marker=False,
                    input_mode="processor",
                    decode_mode="processor",
                )
                predicted = self._extract_choice(generated, question_data["Options"])
                correct = predicted == question_data["Correct_Answer"]
                counts["image_textual_correct"] += int(correct)
                counts["image_textual_total"] += 1
                details.append(
                    {
                        "id": row_id,
                        "question_type": "Image_Textual",
                        "question": question_data["Question"],
                        "generated_answer": generated,
                        "predicted_answer": predicted,
                        "correct_answer": question_data["Correct_Answer"],
                        "correct": correct,
                    }
                )

            for idx, question_data in enumerate(
                self._classification_questions(task, "Pure_Text_Questions")
            ):
                if idx in skipped.get(row_id, {}).get("pure_text", []):
                    continue
                text = (
                    self._question_with_options(
                        question_data["Question"], question_data["Options"]
                    )
                    + "\nJust give ONE letter representing the answer directly."
                )
                prompt, images = self._build_prompt(text, None, prompt_text_shots)
                generated = self._generate(
                    model,
                    tokenizer,
                    prompt,
                    images,
                    allow_answer_marker=False,
                    max_new_tokens=5 if family == "idefics" else None,
                    input_mode="tokenizer",
                    decode_mode="tokenizer",
                )
                predicted = self._extract_choice(generated, question_data["Options"])
                correct = predicted == question_data["Correct_Answer"]
                counts["pure_text_correct"] += int(correct)
                counts["pure_text_total"] += 1
                details.append(
                    {
                        "id": row_id,
                        "question_type": "Pure_Text",
                        "question": question_data["Question"],
                        "generated_answer": generated,
                        "predicted_answer": predicted,
                        "correct_answer": question_data["Correct_Answer"],
                        "correct": correct,
                    }
                )

        result = {
            "Image-Textual Question Accuracy": self._percent(
                counts["image_textual_correct"], counts["image_textual_total"]
            ),
            "Pure Text Question Accuracy": self._percent(
                counts["pure_text_correct"], counts["pure_text_total"]
            ),
            "counts": counts,
        }
        if self.eval_cfg.get("save_task_details", True):
            result["details"] = details
        return result

    def _evaluate_fill_in_the_blank(self, df, id_list, model, tokenizer, family, mode):
        image_shots, text_shots, skipped = self._few_shots_for_blank(id_list, family)
        counts = {
            "image_textual_correct": 0,
            "image_textual_total": 0,
            "pure_text_correct": 0,
            "pure_text_total": 0,
        }
        details = []
        for _, row in df.iterrows():
            row_id = str(row["ID"])
            image = self._row_image(row, mode)
            for idx, question_data in enumerate(self._as_list(row["Mask_Task"])):
                question_type = question_data["Type"]
                if question_type == "Image_Textual" and idx in skipped.get(
                    row_id, {}
                ).get("image_textual", []):
                    continue
                if question_type == "Pure_Text" and idx in skipped.get(row_id, {}).get(
                    "pure_text", []
                ):
                    continue

                text = self._blank_question(question_data["Question"])
                current_image = image if question_type == "Image_Textual" else None
                few_shots = (
                    image_shots if question_type == "Image_Textual" else text_shots
                )
                prompt, images = self._build_prompt(text, current_image, few_shots)
                generated = self._generate(
                    model,
                    tokenizer,
                    prompt,
                    images,
                    input_mode="processor",
                    decode_mode="processor",
                )
                ground_truth = str(question_data["Ground_Truth"])
                correct = (
                    self._contains_scorer.score(generated, ground_truth).value == 1.0
                )
                if question_type == "Image_Textual":
                    counts["image_textual_correct"] += int(correct)
                    counts["image_textual_total"] += 1
                elif question_type == "Pure_Text":
                    counts["pure_text_correct"] += int(correct)
                    counts["pure_text_total"] += 1
                details.append(
                    {
                        "id": row_id,
                        "question_type": question_type,
                        "question": question_data["Question"],
                        "generated_answer": generated,
                        "ground_truth": ground_truth,
                        "correct": correct,
                    }
                )

        result = {
            "image_textual_accuracy": self._percent(
                counts["image_textual_correct"], counts["image_textual_total"]
            ),
            "pure_text_accuracy": self._percent(
                counts["pure_text_correct"], counts["pure_text_total"]
            ),
            "counts": counts,
        }
        if self.eval_cfg.get("save_task_details", True):
            result["details"] = details
        return result

    def _evaluate_generation(self, df, model, tokenizer, mode):
        totals = {
            "image_textual": {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu": 0.0,
                "answer_probability": 0.0,
                "loss_mia": 0.0,
                "zlib_mia": 0.0,
                "min_k_20_mia": 0.0,
                "min_k_plus_plus_20_mia": 0.0,
                "truth_ratio": 0.0,
                "n": 0,
                "probability_n": 0,
                "mia_n": 0,
                "min_k_mia_n": 0,
                "truth_ratio_n": 0,
            },
            "pure_text": {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "bleu": 0.0,
                "answer_probability": 0.0,
                "loss_mia": 0.0,
                "zlib_mia": 0.0,
                "min_k_20_mia": 0.0,
                "min_k_plus_plus_20_mia": 0.0,
                "truth_ratio": 0.0,
                "n": 0,
                "probability_n": 0,
                "mia_n": 0,
                "min_k_mia_n": 0,
                "truth_ratio_n": 0,
            },
        }
        generation_mia = bool(self.eval_cfg.get("generation_mia", False))
        generation_min_k_mia = bool(self.eval_cfg.get("generation_mia_min_k", False))
        generation_answer_probability = bool(
            self.eval_cfg.get("generation_answer_probability", True)
        )
        generation_truth_ratio = bool(
            self.eval_cfg.get("generation_truth_ratio", False)
        )
        truth_ratio_variants = (
            self._load_truth_ratio_variants() if generation_truth_ratio else {}
        )
        truth_ratio_distribution = {"Image_Textual": [], "Pure_Text": []}
        details = []
        for _, row in df.iterrows():
            row_id = str(row["ID"])
            image = self._row_image(row, mode)
            for question_index, question_data in enumerate(
                self._as_list(row["Generation_Task"])
            ):
                question_type = question_data["Type"]
                question = question_data["Question"]
                ground_truth = str(question_data["Ground_Truth"])
                if question_type == "Image_Textual":
                    text = (
                        question
                        + "\nAnswer the question based on your trained knowledge "
                        "in one sentence accurately in ENGLISH."
                    )
                    current_image = image
                    prompt_kwargs = {"raw_assistant_prefix": "ASSISTANT: "}
                else:
                    text = (
                        question
                        + "\nAnswer the question based on your trained knowledge "
                        "in one sentence in ENGLISH."
                    )
                    current_image = None
                    prompt_kwargs = {"text_user_prefix": "USER: "}

                prompt, images = self._build_prompt(
                    text, current_image, [], **prompt_kwargs
                )
                input_mode = (
                    "processor" if question_type == "Image_Textual" else "tokenizer"
                )
                decode_mode = input_mode
                single_image = bool(
                    question_type == "Image_Textual"
                    and self._uses_upstream_raw_generation()
                    and str(self._active_model_family).startswith("llava")
                )

                loss_result = None
                loss_mia = None
                zlib_mia = None
                min_k_20_mia = None
                min_k_plus_plus_20_mia = None
                answer_probability = None
                if (
                    generation_answer_probability
                    or generation_mia
                    or generation_min_k_mia
                    or generation_truth_ratio
                ):
                    full_prompt, answer_images = self._build_prompt(
                        text,
                        current_image,
                        [],
                        answer=ground_truth,
                        **prompt_kwargs,
                    )
                    loss_result = self._answer_loss(
                        model,
                        tokenizer,
                        prompt,
                        full_prompt,
                        answer_images,
                        input_mode=input_mode,
                        single_image=single_image,
                        compute_token_stats=generation_min_k_mia,
                    )
                    if loss_result is None:
                        logger.warning(
                            "Skipping MLLMU teacher-forced generation metrics for id=%s type=%s "
                            "due to empty answer labels.",
                            row_id,
                            question_type,
                        )
                    else:
                        if generation_answer_probability:
                            answer_probability = self._answer_probability_scorer.score(
                                loss_result["avg_loss"]
                            ).value
                        if generation_mia:
                            loss_mia = self._loss_mia_scorer.score(
                                loss_result["avg_loss"]
                            ).value
                            zlib_mia = self._zlib_mia_scorer.score(
                                loss_result["avg_loss"], ground_truth
                            ).value
                        if generation_min_k_mia:
                            min_k_20_mia = self._standard_min_k_scorer.score(
                                loss_result["token_log_probs"]
                            ).value
                            min_k_plus_plus_20_mia = (
                                self._standard_min_k_plus_plus_scorer.score(
                                    loss_result["token_log_probs"],
                                    loss_result["mu"],
                                    loss_result["sigma"],
                                ).value
                            )

                paraphrase_loss = None
                perturbation_losses = []
                truth_ratio = None
                if generation_truth_ratio:
                    variant_key = (row_id, question_index, question_type)
                    variant = truth_ratio_variants.get(variant_key)
                    if variant is None:
                        raise KeyError(
                            "Missing MLLMU Truth Ratio sidecar record for "
                            f"id={row_id}, question_index={question_index}, "
                            f"modality={question_type}"
                        )
                    paraphrase_full_prompt, paraphrase_images = self._build_prompt(
                        text,
                        current_image,
                        [],
                        answer=variant["paraphrased_answer"],
                        **prompt_kwargs,
                    )
                    paraphrase_loss = self._answer_loss(
                        model,
                        tokenizer,
                        prompt,
                        paraphrase_full_prompt,
                        paraphrase_images,
                        input_mode=input_mode,
                        single_image=single_image,
                    )
                    for perturbation in variant["perturbed_answers"]:
                        perturbation_full_prompt, perturbation_images = (
                            self._build_prompt(
                                text,
                                current_image,
                                [],
                                answer=perturbation,
                                **prompt_kwargs,
                            )
                        )
                        loss = self._answer_loss(
                            model,
                            tokenizer,
                            prompt,
                            perturbation_full_prompt,
                            perturbation_images,
                            input_mode=input_mode,
                            single_image=single_image,
                        )
                        if loss is not None:
                            perturbation_losses.append(loss)
                    if (
                        loss_result is not None
                        and paraphrase_loss is not None
                        and len(perturbation_losses)
                        == len(variant["perturbed_answers"])
                    ):
                        positive_loss = (
                            loss_result["avg_loss"] + paraphrase_loss["avg_loss"]
                        ) / 2.0
                        truth_ratio = self._truth_ratio_scorer.score(
                            positive_loss,
                            [loss["avg_loss"] for loss in perturbation_losses],
                        ).value

                generated = self._generate(
                    model,
                    tokenizer,
                    prompt,
                    images,
                    input_mode=input_mode,
                    decode_mode=decode_mode,
                    single_image=single_image,
                )
                rouge_scores = self._generation_rouge_scorer.score(
                    generated, ground_truth
                ).details
                bleu = self._bleu(ground_truth, generated)
                bucket = (
                    "image_textual" if question_type == "Image_Textual" else "pure_text"
                )
                totals[bucket]["rouge1"] += rouge_scores["rouge1"]
                totals[bucket]["rouge2"] += rouge_scores["rouge2"]
                totals[bucket]["rougeL"] += rouge_scores["rougeL"]
                totals[bucket]["bleu"] += bleu
                totals[bucket]["n"] += 1
                if loss_result is not None and generation_mia:
                    totals[bucket]["loss_mia"] += loss_mia
                    totals[bucket]["zlib_mia"] += zlib_mia
                    totals[bucket]["mia_n"] += 1
                if answer_probability is not None:
                    totals[bucket]["answer_probability"] += answer_probability
                    totals[bucket]["probability_n"] += 1
                if loss_result is not None and generation_min_k_mia:
                    totals[bucket]["min_k_20_mia"] += min_k_20_mia
                    totals[bucket]["min_k_plus_plus_20_mia"] += min_k_plus_plus_20_mia
                    totals[bucket]["min_k_mia_n"] += 1
                if truth_ratio is not None:
                    totals[bucket]["truth_ratio"] += truth_ratio
                    totals[bucket]["truth_ratio_n"] += 1
                    truth_ratio_distribution[question_type].append(truth_ratio)
                detail = {
                    "image_id": row_id,
                    "question type": question_type,
                    "question": question,
                    "generated_answer": generated,
                    "ground_truth": ground_truth,
                }
                if loss_result is not None:
                    detail.update(
                        {
                            "avg_gt_loss": loss_result["avg_loss"],
                            "gt_loss": loss_result["loss"],
                            "num_token_gt": loss_result["num_tokens"],
                        }
                    )
                    if generation_mia:
                        detail.update({"loss_mia": loss_mia, "zlib_mia": zlib_mia})
                    if generation_min_k_mia:
                        detail.update(
                            {
                                "min_k_20_mia": min_k_20_mia,
                                "min_k_plus_plus_20_mia": min_k_plus_plus_20_mia,
                            }
                        )
                    if answer_probability is not None:
                        detail["answer_probability"] = answer_probability
                if generation_truth_ratio:
                    detail["truth_ratio"] = truth_ratio
                    if paraphrase_loss is not None:
                        detail["avg_paraphrase_loss"] = paraphrase_loss["avg_loss"]
                    if perturbation_losses:
                        detail["avg_perturbation_losses"] = [
                            loss["avg_loss"] for loss in perturbation_losses
                        ]
                details.append(detail)

        result = {}
        if totals["image_textual"]["n"]:
            n = totals["image_textual"]["n"]
            result.update(
                {
                    "Average ROUGE-1 (Image_Textual)": totals["image_textual"]["rouge1"]
                    / n,
                    "Average ROUGE-2 (Image_Textual)": totals["image_textual"]["rouge2"]
                    / n,
                    "Average ROUGE-L (Image_Textual)": totals["image_textual"]["rougeL"]
                    / n,
                    "Average BLEU (Image_Textual)": totals["image_textual"]["bleu"] / n,
                }
            )
            if totals["image_textual"]["probability_n"]:
                result["Average Answer Probability (Image_Textual)"] = (
                    totals["image_textual"]["answer_probability"]
                    / totals["image_textual"]["probability_n"]
                )
            if totals["image_textual"]["truth_ratio_n"]:
                result["Average Truth Ratio (Image_Textual)"] = (
                    totals["image_textual"]["truth_ratio"]
                    / totals["image_textual"]["truth_ratio_n"]
                )
        if totals["pure_text"]["n"]:
            n = totals["pure_text"]["n"]
            result.update(
                {
                    "Average ROUGE-1 (Pure_Text)": totals["pure_text"]["rouge1"] / n,
                    "Average ROUGE-2 (Pure_Text)": totals["pure_text"]["rouge2"] / n,
                    "Average ROUGE-L (Pure_Text)": totals["pure_text"]["rougeL"] / n,
                    "Average BLEU (Pure_Text)": totals["pure_text"]["bleu"] / n,
                }
            )
            if totals["pure_text"]["probability_n"]:
                result["Average Answer Probability (Pure_Text)"] = (
                    totals["pure_text"]["answer_probability"]
                    / totals["pure_text"]["probability_n"]
                )
            if totals["pure_text"]["truth_ratio_n"]:
                result["Average Truth Ratio (Pure_Text)"] = (
                    totals["pure_text"]["truth_ratio"]
                    / totals["pure_text"]["truth_ratio_n"]
                )
        if generation_mia and totals["image_textual"]["mia_n"]:
            n = totals["image_textual"]["mia_n"]
            result.update(
                {
                    "Average Loss MIA (Image_Textual)": totals["image_textual"][
                        "loss_mia"
                    ]
                    / n,
                    "Average ZLIB MIA (Image_Textual)": totals["image_textual"][
                        "zlib_mia"
                    ]
                    / n,
                }
            )
        if generation_mia and totals["pure_text"]["mia_n"]:
            n = totals["pure_text"]["mia_n"]
            result.update(
                {
                    "Average Loss MIA (Pure_Text)": totals["pure_text"]["loss_mia"] / n,
                    "Average ZLIB MIA (Pure_Text)": totals["pure_text"]["zlib_mia"] / n,
                }
            )
        if generation_min_k_mia:
            for bucket, label in (
                ("image_textual", "Image_Textual"),
                ("pure_text", "Pure_Text"),
            ):
                n = totals[bucket]["min_k_mia_n"]
                if n:
                    result.update(
                        {
                            f"Average Min-K 20% MIA ({label})": totals[bucket][
                                "min_k_20_mia"
                            ]
                            / n,
                            f"Average Min-K++ 20% MIA ({label})": totals[bucket][
                                "min_k_plus_plus_20_mia"
                            ]
                            / n,
                        }
                    )
        if generation_truth_ratio:
            result["truth_ratio_distribution"] = truth_ratio_distribution
        result["counts"] = {
            "image_textual_total": totals["image_textual"]["n"],
            "pure_text_total": totals["pure_text"]["n"],
            "image_textual_probability_total": totals["image_textual"]["probability_n"],
            "pure_text_probability_total": totals["pure_text"]["probability_n"],
        }
        if generation_mia:
            result["counts"].update(
                {
                    "image_textual_mia_total": totals["image_textual"]["mia_n"],
                    "pure_text_mia_total": totals["pure_text"]["mia_n"],
                }
            )
        if generation_min_k_mia:
            result["counts"].update(
                {
                    "image_textual_min_k_mia_total": totals["image_textual"][
                        "min_k_mia_n"
                    ],
                    "pure_text_min_k_mia_total": totals["pure_text"]["min_k_mia_n"],
                }
            )
        if generation_truth_ratio:
            result["counts"].update(
                {
                    "image_textual_truth_ratio_total": totals["image_textual"][
                        "truth_ratio_n"
                    ],
                    "pure_text_truth_ratio_total": totals["pure_text"]["truth_ratio_n"],
                }
            )
        if self.eval_cfg.get("save_task_details", True):
            result["details"] = details
        return result

    def _percent(self, numerator, denominator):
        return (numerator / denominator) * 100 if denominator else 0.0

    def _split_specs(self):
        data_root = Path(self.eval_cfg.data_root)
        forget_ratio = int(
            self.eval_cfg.get("ratio", self.eval_cfg.get("forget_ratio", 5))
        )
        retain_ratio = self.eval_cfg.get("retain_ratio", None)
        retain_ratio = 100 - forget_ratio if retain_ratio is None else int(retain_ratio)
        forget_path = self.eval_cfg.get("forget_path", None) or str(
            data_root / f"forget_{forget_ratio}" / "train-00000-of-00001.parquet"
        )
        retain_path = self.eval_cfg.get("retain_path", None) or str(
            data_root / f"retain_{retain_ratio}" / "train-00000-of-00001.parquet"
        )
        return {
            "forget": {"path": forget_path, "mode": "forget"},
            "test": {"path": self.eval_cfg.test_path, "mode": "test"},
            "retain_shared": {
                "path": retain_path,
                "mode": "retain_shared",
            },
            "retain_celebrity": {
                "path": self.eval_cfg.celebrity_path,
                "mode": "retain_celebrity",
            },
        }

    def _evaluate_split(self, split_name, spec, model, tokenizer, family, forget_df):
        df = self._read_parquet(spec["path"])
        id_list = self._id_list_for_split(df, spec["mode"], forget_df=forget_df)
        df = self._filtered_samples(df, id_list, spec["mode"])

        result = {}
        configured_tasks = self._to_container(self.eval_cfg.get("tasks", []), [])
        source_task_order = {
            "forget": ("fill_in_the_blank", "classification", "generation"),
            "test": ("classification", "fill_in_the_blank", "generation"),
            "retain_shared": ("fill_in_the_blank", "classification", "generation"),
            "retain_celebrity": ("fill_in_the_blank", "classification", "generation"),
        }
        if self.eval_cfg.get("upstream_compatibility", True):
            tasks = [
                task
                for task in source_task_order[spec["mode"]]
                if task in configured_tasks
            ]
        else:
            tasks = configured_tasks

        for task in tasks:
            if task == "fill_in_the_blank":
                result[task] = self._evaluate_fill_in_the_blank(
                    df, id_list, model, tokenizer, family, spec["mode"]
                )
            elif task == "classification":
                result[task] = self._evaluate_classification(
                    df, id_list, model, tokenizer, family, spec["mode"]
                )
            elif task == "generation":
                result[task] = self._evaluate_generation(
                    df, model, tokenizer, spec["mode"]
                )
                if self.eval_cfg.get("save_generation_details", True):
                    self._save_generation_details(split_name, result[task])
        return result

    def _add_generation_distribution_metrics(self, logs, enabled_splits):
        """Compare each split's Truth Ratio distribution with a reference split."""
        if not self.eval_cfg.get("generation_truth_ratio", False):
            return
        reference_split = str(
            self.eval_cfg.get(
                "generation_distribution_reference_split", "retain_shared"
            )
        )
        reference_result = logs.get(reference_split, {}).get("generation", {})
        reference_distribution = reference_result.get("truth_ratio_distribution", {})
        if not reference_distribution:
            logger.warning(
                "Cannot compute MLLMU KS/JS metrics: reference split %s has no "
                "Truth Ratio distribution.",
                reference_split,
            )
            return

        for split_name in enabled_splits:
            if split_name == reference_split:
                continue
            generation_result = logs.get(split_name, {}).get("generation", {})
            distribution = generation_result.get("truth_ratio_distribution", {})
            if not distribution:
                continue
            for modality in ("Image_Textual", "Pure_Text"):
                current = distribution.get(modality, [])
                reference = reference_distribution.get(modality, [])
                if not current or not reference:
                    continue
                ks_result = self._ks_scorer.score(current, reference)
                js_result = self._js_scorer.score(current, reference)
                generation_result[f"KS Test Statistic ({modality})"] = float(
                    ks_result.details["statistic"]
                )
                generation_result[f"KS Test PValue ({modality})"] = float(
                    ks_result.details["pvalue"]
                )
                generation_result[f"JS Distance ({modality})"] = float(js_result.value)
            generation_result["distribution_reference_split"] = reference_split

    def _save_generation_details(self, split_name, generation_result):
        output_dir = self.eval_cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        details = generation_result.get("details", [])
        file_path = os.path.join(output_dir, f"{split_name}_generation_results.json")
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"Generation_Questions": details, "Description_Questions": []},
                handle,
                indent=4,
            )

    def summarize(self, logs):
        summary = {}
        for split_name, split_result in logs.items():
            for task_name, task_result in split_result.items():
                for metric_name, value in task_result.items():
                    if metric_name in (
                        "details",
                        "counts",
                        "truth_ratio_distribution",
                        "distribution_reference_split",
                    ):
                        continue
                    summary[f"{split_name}/{task_name}/{metric_name}"] = value
        return summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
        model = self.prepare_model(model)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            raise ValueError("MLLMUBenchEvaluator requires a tokenizer.")
        self._load_processor()

        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        with open_dict(self.eval_cfg):
            self.eval_cfg.output_dir = output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        family = self._model_family(model)
        self._active_model_family = family
        self._prompt_style = self._effective_prompt_style(model)
        split_specs = self._split_specs()
        enabled_splits = self._to_container(
            self.eval_cfg.get("splits", list(split_specs.keys())),
            list(split_specs.keys()),
        )
        forget_df = self._read_parquet(split_specs["forget"]["path"])

        logger.info("***** Running %s evaluation suite *****", self.name)
        logger.info("Fine-grained evaluations will be saved to: %s", logs_file_path)
        logger.info(
            "Aggregated evaluations will be summarised in: %s", summary_file_path
        )

        for split_name in enabled_splits:
            if split_name not in split_specs:
                raise ValueError(f"Unknown MLLMU split: {split_name}")
            if not overwrite and split_name in logs and logs[split_name]:
                generation_result = logs[split_name].get("generation", {})
                truth_ratio_ready = "truth_ratio_distribution" in generation_result
                if (
                    not self.eval_cfg.get("generation_truth_ratio", False)
                    or truth_ratio_ready
                ):
                    logger.info("Skipping %s, already evaluated.", split_name)
                    continue
                logger.info(
                    "Re-evaluating %s because Truth Ratio metrics are missing.",
                    split_name,
                )
            logger.info("Running MLLMU split: %s", split_name)
            logs[split_name] = self._evaluate_split(
                split_name, split_specs[split_name], model, tokenizer, family, forget_df
            )
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        self._add_generation_distribution_metrics(logs, enabled_splits)
        summary = self.summarize(logs)
        self.save_logs(logs, logs_file_path)
        self.save_logs(summary, summary_file_path)
        return summary
