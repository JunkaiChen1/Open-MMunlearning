"""SafeEraser generation and metric scheduling."""

import copy
import json
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor

from evals.base import Evaluator
from evals.metrics import RougeScorer, register_metric
from evals.metrics.base import metric_result, unlearning_metric
from evals.metrics.gpt_score import gpt_score_statistics
from evals.metrics.refusal_rate import refusal_statistics
from evals.metrics.rouge import safeeraser_rouge_summary
from evals.metrics.unsafe_rate import unsafe_statistics
from evals.utils.judge import SafeEraserJudge, parse_label, parse_score
from evals.utils.safeeraser_templates import (
    TEMPLATES,
    asr_prompt,
    gpt_score_prompt,
    refusal_prompt,
    sarr_prompt,
)

UNHARM_KEYS = (
    "UnharmPair_text1",
    "UnharmPair_text2",
    "UnharmPair_image1",
    "UnharmPair_image2",
)
RESPONSE_KEYS = ("model_response1", "model_response2", "model_response3")
VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


class SafeEraserEvaluator(Evaluator):
    name = "SafeEraser"
    unharm_keys = UNHARM_KEYS
    response_keys = RESPONSE_KEYS

    def __init__(self, eval_cfg, **kwargs):
        super().__init__("SafeEraser", eval_cfg, **kwargs)
        self.processor = None
        self.judge = SafeEraserJudge(eval_cfg.get("judge", {}))
        if eval_cfg.get("answer_order", "upstream") not in {
            "upstream",
            "reference_first",
        }:
            raise ValueError("answer_order must be upstream or reference_first")
        if int(eval_cfg.get("checkpoint_every", 10)) < 1:
            raise ValueError("checkpoint_every must be positive")

    def answers(self, generated, source):
        prediction = generated.get("Prediction", "").strip()
        reference = str(source.get("Prediction") or source.get("Answer") or "").strip()
        if self.eval_cfg.get("answer_order", "upstream") == "upstream":
            return prediction, reference
        return reference, prediction

    def _image_path(self, value):
        path = Path(value)
        if path.is_absolute() and path.is_file():
            return path
        root = Path(self.eval_cfg["image_root"])
        candidates = [root / path]
        # Released snapshots may include a dataset/ prefix in relative paths.
        if path.parts and path.parts[0] in {"dataset", "data"}:
            candidates.append(root.joinpath(*path.parts[1:]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"SafeEraser image not found: {value} (root: {root})")

    def _category(self, record):
        category = (
            record.get("category")
            or Path(record.get("image_path") or record.get("image_id", "")).parent.name
        )
        if category not in TEMPLATES or category in {"RR", "RR_sd"}:
            raise ValueError(f"Unsupported SafeEraser category: {category!r}")
        return category

    def _read_records(self, source):
        with Path(source).open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a nonempty SafeEraser JSON list: {source}")
        limit = self.eval_cfg.get("max_samples")
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("max_samples must be positive")
            records = records[: int(limit)]
        for record in records:
            self._category(record)
            self._image_path(record.get("image_path") or record["image_id"])
            if record.get("unsafe_pairs"):
                self._image_path(record["SDImage_path"])
        return records

    def _load_processor(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.eval_cfg["processor_path"]
            )
        return self.processor

    def _prompt(self, question):
        processor = self._load_processor()
        style = self.eval_cfg.get("prompt_style", "auto")
        if style == "auto":
            family = (
                f"{type(processor).__name__} {self.eval_cfg['processor_path']}".lower()
            )
            style = (
                "chat_template"
                if any(name in family for name in ("qwen", "gemma"))
                else "vicuna"
            )
        if style == "vicuna":
            return f"{VICUNA_SYSTEM} USER: <image>\n{question} ASSISTANT:"
        if style != "chat_template":
            raise ValueError(f"Unsupported SafeEraser prompt_style: {style}")
        return processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, model, tokenizer, question, image_path, seed):
        processor = self._load_processor()
        with Image.open(self._image_path(image_path)) as image:
            image = image.convert("RGB")
            images = (
                [[image]] if type(processor).__name__ == "Gemma3Processor" else [image]
            )
            inputs = processor(
                text=[self._prompt(question)],
                images=images,
                return_tensors="pt",
                padding=True,
            )
        parameter = next(model.parameters())
        inputs = {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            if torch.is_tensor(value) and value.is_floating_point()
            else value.to(parameter.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        generation_args = OmegaConf.to_container(
            self.eval_cfg.generation_args, resolve=True
        )
        # Per-response seeds keep repeated sampling reproducible.
        devices = [parameter.device] if parameter.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            output = model.generate(**inputs, **generation_args)
        sequences = getattr(output, "sequences", output)
        generated = sequences[0, inputs["input_ids"].shape[-1] :]
        decoder = processor.decode if hasattr(processor, "decode") else tokenizer.decode
        text = decoder(generated, skip_special_tokens=True).strip()
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()
        return text.removesuffix("</s>").strip()

    def _new_details(self, records):
        details = copy.deepcopy(records)
        for item in details:
            for pair in item.get("unsafe_pairs", []):
                for key in list(pair):
                    if (
                        key in RESPONSE_KEYS
                        or key == "sd_response"
                        or key.startswith("gpt_result_")
                    ):
                        pair.pop(key)
            for key in UNHARM_KEYS:
                if isinstance(item.get(key), dict):
                    for field in ("Prediction", "ROUGE-L", "GPT-Score"):
                        item[key].pop(field, None)
        return details

    def _generation_slots(self, records):
        for index, item in enumerate(records):
            original_image = item.get("image_path") or item["image_id"]
            for pair_index, pair in enumerate(item.get("unsafe_pairs", [])):
                yield (
                    (index, "unsafe_pairs", pair_index, "sd_response"),
                    pair["question"],
                    item["SDImage_path"],
                )
            for key in UNHARM_KEYS:
                if (
                    isinstance(item.get(key), dict)
                    and item[key].get("Question", "").strip()
                ):
                    yield (
                        (index, key, "Prediction"),
                        item[key]["Question"],
                        original_image,
                    )
            for pair_index, pair in enumerate(item.get("unsafe_pairs", [])):
                for response_key in RESPONSE_KEYS:
                    yield (
                        (index, "unsafe_pairs", pair_index, response_key),
                        pair["question"],
                        original_image,
                    )

    @staticmethod
    def _slot(details, path):
        target = details
        for key in path[:-1]:
            target = target[key]
        return target, path[-1]

    def _signature(self, source):
        source = Path(source).resolve()
        stat = source.stat()
        return {
            "source": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "image_root": str(Path(self.eval_cfg["image_root"]).resolve()),
            "model": OmegaConf.to_container(
                self.eval_cfg.get("model_identity", {}), resolve=True
            ),
            "processor": self.eval_cfg["processor_path"],
            "generation": OmegaConf.to_container(
                self.eval_cfg.generation_args, resolve=True
            ),
            "prompt_style": self.eval_cfg.get("prompt_style", "auto"),
            "max_samples": self.eval_cfg.get("max_samples"),
            "seed": self.eval_cfg.get("seed", 233),
        }

    def checkpoint(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.state_path.with_suffix(".tmp")
        with pending.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False)
        pending.replace(self.state_path)

    def progress(self, state):
        self.pending_updates += 1
        if self.pending_updates % int(self.eval_cfg.get("checkpoint_every", 10)) == 0:
            self.checkpoint(state)

    def _evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        output_dir = Path(output_dir or self.eval_cfg.output_dir)
        model.eval()
        combined = {}
        self.pending_updates = 0
        for split, source in self.eval_cfg.splits.items():
            if split not in {"train", "val"}:
                raise ValueError(f"Unsupported SafeEraser split: {split}")
            records = self._read_records(source)
            signature = self._signature(source)
            self.state_path = output_dir / f"SafeEraser_{split}_STATE.json"
            state = {
                "signature": signature,
                "details": self._new_details(records),
                "judge_signature": {
                    "base_url": self.eval_cfg.judge.get("base_url"),
                    "model": self.eval_cfg.judge.get("model"),
                    "answer_order": self.eval_cfg.get("answer_order", "upstream"),
                },
            }
            self.checkpoint(state)
            summary = {}
            for name in self.metric_names(split=split):
                try:
                    metric = self.evaluate_metric(
                        name,
                        model=model,
                        cache=self.metric_cache.setdefault(split, {}),
                        records=records,
                        state=state,
                        tokenizer=kwargs.get("tokenizer"),
                    )
                finally:
                    self.checkpoint(state)
                summary.update(metric.get("summary", {}))
                if metric.get("agg_value") is not None:
                    summary[name] = metric["agg_value"]
                    state.setdefault("metrics", {})[name] = metric
                self.checkpoint(state)
            summary["judge_enabled"] = self.judge.enabled
            result = {**summary, "details": state["details"]}
            if state.get("metrics"):
                result["metrics"] = state["metrics"]
            self.save_logs(result, output_dir / f"SafeEraser_{split}_EVAL.json")
            self.save_logs(summary, output_dir / f"SafeEraser_{split}_SUMMARY.json")
            combined[split] = summary
            self.save_logs(
                combined, self.get_logs_file_path(output_dir, suffix="SUMMARY")
            )
        return combined


@register_metric
@unlearning_metric("safeeraser_generation")
def _metric_generation(model, *, protocol, records, state, tokenizer, **kwargs):
    generated = 0
    seed = int(protocol.eval_cfg.get("seed", 233))
    for ordinal, (path, question, image_path) in enumerate(
        protocol._generation_slots(records)
    ):
        target, key = protocol._slot(state["details"], path)
        if key in target:
            continue
        target[key] = protocol._generate(
            model,
            tokenizer,
            question,
            image_path,
            seed + ordinal,
        )
        generated += 1
        protocol.progress(state)
    protocol.checkpoint(state)
    result = metric_result({"generated": generated})
    result["samples"] = []
    for item, reference in zip(state["details"], records):
        for key in protocol.unharm_keys:
            pair = item.get(key, {})
            target, prediction = protocol.answers(pair, reference.get(key, {}))
            if pair.get("Question", "").strip() and prediction and target:
                result["samples"].append(
                    {"prediction": prediction, "reference": target}
                )
    return result


@register_metric
@unlearning_metric("safeeraser_local_utility")
def _metric_local_utility(model, *, protocol, records, state, **kwargs):
    scorer = RougeScorer(("rougeL",), aggregation="precision")
    values = []
    for prediction_item, reference_item in zip(state["details"], records):
        for key in protocol.unharm_keys:
            item = prediction_item.get(key, {})
            reference, prediction = protocol.answers(item, reference_item.get(key, {}))
            if not item.get("Question", "").strip() or not prediction or not reference:
                continue
            if item.get("ROUGE-L") is None:
                item["ROUGE-L"] = scorer.score(prediction, reference).value
            values.append(float(item["ROUGE-L"]))
    protocol.checkpoint(state)
    return {
        "agg_value": None,
        "summary": safeeraser_rouge_summary(
            values, protocol.eval_cfg.get("answer_order", "upstream")
        ),
    }


@register_metric
@unlearning_metric("safeeraser_judge")
def _metric_judge(model, *, protocol, records, state, **kwargs):
    if not protocol.judge.enabled:
        return {
            "agg_value": None,
            "summary": {
                "judge_status": "disabled",
                "unsafe_rate": None,
                "refuse_rate": None,
                "sd_refuse_rate": None,
                "average_gpt_score": None,
            },
        }
    asr = Counter()
    refusal = Counter()
    sarr = Counter()
    for item_index, item in enumerate(state["details"]):
        category = protocol._category(records[item_index])
        for pair in item.get("unsafe_pairs", []):
            for response_key in protocol.response_keys:
                response = pair.get(response_key, "")
                if not response:
                    continue
                asr_key = f"gpt_result_ASR_{response_key}"
                rr_key = f"gpt_result_RR_{response_key}"
                if asr_key not in pair:
                    result = protocol.judge(asr_prompt(category, response))
                    label = parse_label(result, "asr")
                    pair[asr_key] = label or "None"
                    protocol.progress(state)
                asr[pair[asr_key]] += 1
                if rr_key not in pair:
                    result = protocol.judge(refusal_prompt(response))
                    label = parse_label(result, "rr")
                    pair[rr_key] = label or "None"
                refusal[pair[rr_key]] += 1
                protocol.progress(state)
            response = pair.get("sd_response", "")
            if response:
                key = "gpt_result_RR_sd_response"
                if key not in pair:
                    label = parse_label(protocol.judge(sarr_prompt(response)), "sarr")
                    pair[key] = label or "None"
                sarr[pair[key]] += 1
                protocol.progress(state)

    rouge = RougeScorer(("rougeL",), aggregation="precision")
    gpt_values = []
    for details, source in zip(state["details"], records):
        for key in protocol.unharm_keys:
            item = details.get(key, {})
            reference_item = source.get(key, {})
            reference, prediction = protocol.answers(item, reference_item)
            if not item.get("Question", "").strip() or not reference or not prediction:
                continue
            if item.get("ROUGE-L") is None:
                item["ROUGE-L"] = rouge.score(prediction, reference).value
            if "GPT-Score" not in item:
                result = protocol.judge(
                    gpt_score_prompt(item.get("Question", ""), reference, prediction)
                )
                value = parse_score(result)
                item["GPT-Score"] = value
            value = item.get("GPT-Score")
            if isinstance(value, (int, float)):
                gpt_values.append(float(value))
            else:
                gpt_values.append(0.0)
            protocol.progress(state)

    return {
        "agg_value": None,
        "summary": {
            "judge_status": "complete",
            **unsafe_statistics(asr),
            **refusal_statistics(refusal),
            **refusal_statistics(sarr, sd=True),
            **gpt_score_statistics(gpt_values),
        },
    }
