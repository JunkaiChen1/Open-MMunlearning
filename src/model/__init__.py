from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os
import torch
import logging
from model.probe import (
    ProbedGemma3ForConditionalGeneration,
    ProbedLlamaForCausalLM,
    ProbedLlavaForConditionalGeneration,
    ProbedLlavaNextForConditionalGeneration,
    ProbedQwen2_5_VLForConditionalGeneration,
)

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


class Qwen2_5_VLForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL

        return Qwen2_5_VL.from_pretrained(*args, **kwargs)


class LlavaForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import LlavaForConditionalGeneration as Llava

        return Llava.from_pretrained(*args, **kwargs)


class LlavaNextForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import LlavaNextForConditionalGeneration as LlavaNext

        return LlavaNext.from_pretrained(*args, **kwargs)


class Gemma3ForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from transformers import Gemma3ForConditionalGeneration as Gemma3

        return Gemma3.from_pretrained(*args, **kwargs)


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _quantization_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return dtypes[str(value).lower()]
    except KeyError as error:
        supported = ", ".join(dtypes)
        raise ValueError(
            "Unsupported quantization compute_dtype "
            f"'{value}'. Supported values: {supported}."
        ) from error


def get_quantization_config(
    model_cfg: DictConfig,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Build a bitsandbytes configuration from serializable Hydra settings.

    Quantization is deliberately model-scoped so the same train/eval entry
    points can load either a normal checkpoint or a 4-bit inference copy.
    """
    quantization_cfg = model_cfg.get("quantization", None)
    if not quantization_cfg or not quantization_cfg.get("enabled", False):
        return None, {}

    bits = quantization_cfg.get("bits", 4)
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (4, 8):
        raise ValueError("quantization.bits must be either 4 or 8.")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "bitsandbytes quantization requires transformers with "
            "BitsAndBytesConfig and the bitsandbytes package. Install the "
            "project requirements before enabling model.quantization."
        ) from error

    device_map = quantization_cfg.get("device_map", None)
    load_kwargs = {}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if bits == 4:
        compute_dtype = _quantization_dtype(
            quantization_cfg.get("compute_dtype", "bfloat16")
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization_cfg.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quantization_cfg.get("use_double_quant", True),
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quantization_cfg.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=quantization_cfg.get(
                "llm_int8_has_fp16_weight", False
            ),
            llm_int8_enable_fp32_cpu_offload=quantization_cfg.get(
                "llm_int8_enable_fp32_cpu_offload", False
            ),
        )

    logger.info(
        "Loading model with bitsandbytes %s-bit quantization (device_map=%s).",
        bits,
        device_map,
    )
    return quantization_config, load_kwargs


def _as_plain_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _count_trainable_params(model):
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def _enable_input_require_grads(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = (
        model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    )
    if input_embeddings is None:
        return

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def _resolve_lora_target_modules(cfg):
    preset = cfg.pop("module_preset", None)
    custom_target_modules = cfg.pop("custom_target_modules", None)
    explicit_target_modules = cfg.get("target_modules", None)
    if explicit_target_modules is not None:
        return

    if custom_target_modules is not None:
        cfg["target_modules"] = custom_target_modules
        return

    if preset in (None, "all"):
        cfg["target_modules"] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif preset == "attn":
        cfg["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif preset == "llavanext_language":
        cfg["target_modules"] = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
    else:
        raise ValueError(
            f"Unsupported LoRA module_preset: {preset}. "
            "Use one of: all, attn, llavanext_language; or set custom_target_modules."
        )


def apply_peft(model, peft_cfg):
    cfg = _as_plain_dict(peft_cfg)
    if not cfg or not cfg.pop("enabled", False):
        return model

    method = cfg.pop("method", "lora").lower()
    if method != "lora":
        raise ValueError(f"Unsupported PEFT method: {method}")

    prepare_for_gradient_checkpointing = cfg.pop(
        "prepare_model_for_gradient_checkpointing", True
    )
    print_trainable_parameters = cfg.pop("print_trainable_parameters", True)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "PEFT is required for LoRA training. Install it with `pip install peft` "
            "or install the project requirements."
        ) from e

    if hasattr(model, "config"):
        model.config.use_cache = False

    if prepare_for_gradient_checkpointing:
        _enable_input_require_grads(model)

    _resolve_lora_target_modules(cfg)
    cfg = {key: value for key, value in cfg.items() if value is not None}
    model = get_peft_model(model, LoraConfig(**cfg))
    trainable, total = _count_trainable_params(model)
    pct = 100 * trainable / total if total else 0.0
    logger.info(
        "LoRA enabled: trainable params %s / %s (%.4f%%)",
        trainable,
        total,
        pct,
    )
    if print_trainable_parameters and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def load_trainable_peft_adapter(model, adapter_path, peft_cfg=None):
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "PEFT is required to load a trainable LoRA adapter. Install it with "
            "`pip install peft` or install the project requirements."
        ) from e

    cfg = _as_plain_dict(peft_cfg)
    prepare_for_gradient_checkpointing = cfg.pop(
        "prepare_model_for_gradient_checkpointing", True
    )
    print_trainable_parameters = cfg.pop("print_trainable_parameters", True)

    if hasattr(model, "config"):
        model.config.use_cache = False

    if prepare_for_gradient_checkpointing:
        _enable_input_require_grads(model)

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=True,
    )
    trainable, total = _count_trainable_params(model)
    pct = 100 * trainable / total if total else 0.0
    logger.info(
        "Trainable PEFT adapter loaded from %s: trainable params %s / %s (%.4f%%)",
        adapter_path,
        trainable,
        total,
        pct,
    )
    if print_trainable_parameters and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def load_merged_peft_adapter(model, adapter_path):
    """Merge a frozen adapter so head-only training can save a full model."""
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "PEFT is required to merge a LoRA adapter. Install it with "
            "`pip install peft` or install the project requirements."
        ) from e

    logger.info("Merging PEFT adapter from %s into the base model.", adapter_path)
    peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model = peft_model.merge_and_unload()
    # PEFT leaves this metadata on the returned Transformers model even though
    # no adapter layers remain. It causes false "multiple adapters" warnings
    # when SMFA immediately attaches its own adapters.
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_output_embeddings().parameters():
        parameter.requires_grad = True
    return model


UNLEARNING_ARTIFACT_CONFIG = "unlearning_artifact.json"


def _artifact_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _apply_manu_masks(model, masks: Dict[str, torch.Tensor]) -> int:
    modules = dict(model.named_modules())
    applied = 0
    with torch.no_grad():
        for name, mask in masks.items():
            module = modules.get(name)
            if module is None or not hasattr(module, "weight"):
                raise ValueError(f"MANU artifact module not found: {name}")
            if mask.ndim != 1 or mask.shape[0] != module.weight.shape[0]:
                raise ValueError(
                    f"MANU artifact mask {tuple(mask.shape)} does not match "
                    f"{name} weight {tuple(module.weight.shape)}"
                )
            module.weight[mask.to(module.weight.device, dtype=torch.bool)] = 0
            if module.bias is not None:
                module.bias[mask.to(module.bias.device, dtype=torch.bool)] = 0
            applied += int(mask.count_nonzero())
    return applied


def _load_manu_artifact(model, artifact_dir: Path, config: Dict[str, Any]):
    source_adapter = _artifact_path(artifact_dir, config["source_adapter_path"])
    model = load_merged_peft_adapter(model, source_adapter)
    masks_path = _artifact_path(artifact_dir, config["masks_path"])
    masks = torch.load(masks_path, map_location="cpu", weights_only=True)
    applied = _apply_manu_masks(model, masks)
    logger.info("Applied MANU artifact from %s (%d pruned rows).", artifact_dir, applied)
    return model


def _smfa_delta(module, adapter_name: str) -> torch.Tensor:
    return module.scaling[adapter_name] * (
        module.lora_B[adapter_name].weight.float()
        @ module.lora_A[adapter_name].weight.float()
    )


def _sculpt_smfa_delta(delta: torch.Tensor, retain_delta: torch.Tensor, k: float):
    ratio = torch.linalg.vector_norm(delta) / (
        torch.linalg.vector_norm(retain_delta) + 1e-12
    )
    conflict = (delta * retain_delta) < 0
    relative = k * ratio * retain_delta.abs() < delta.abs()
    return delta.masked_fill(conflict & relative, 0)


def _load_smfa_artifact(model, artifact_dir: Path, config: Dict[str, Any]):
    from peft import PeftModel
    from peft.tuners.lora import Linear as LoraLinear

    source_adapter = _artifact_path(artifact_dir, config["source_adapter_path"])
    model = load_merged_peft_adapter(model, source_adapter)
    adapters = tuple(config["adapters"])
    if len(adapters) != 3 or len(set(adapters)) != 3:
        raise ValueError("SMFA artifact must contain three distinct adapters")

    first = adapters[0]
    peft_model = PeftModel.from_pretrained(
        model,
        artifact_dir / first,
        adapter_name=first,
        is_trainable=False,
    )
    for adapter_name in adapters[1:]:
        peft_model.load_adapter(
            artifact_dir / adapter_name,
            adapter_name=adapter_name,
            is_trainable=False,
        )

    multi_name, text_name, retain_name = adapters
    applied = 0
    with torch.no_grad():
        for module in peft_model.modules():
            if not isinstance(module, LoraLinear):
                continue
            if not all(
                name in module.lora_A and name in module.lora_B for name in adapters
            ):
                continue
            retain_delta = _smfa_delta(module, retain_name)
            multi_delta = _sculpt_smfa_delta(
                _smfa_delta(module, multi_name),
                retain_delta,
                float(config["multi_k"]),
            )
            text_delta = _sculpt_smfa_delta(
                _smfa_delta(module, text_name),
                retain_delta,
                float(config["text_k"]),
            )
            module.base_layer.weight.add_(
                ((multi_delta + text_delta) / 2).to(
                    module.base_layer.weight.device,
                    module.base_layer.weight.dtype,
                )
            )
            applied += 1
    if applied == 0:
        raise ValueError("SMFA artifact did not match any LoRA layers")
    model = peft_model.unload()
    if hasattr(model, "peft_config"):
        delattr(model, "peft_config")
    logger.info("Applied SMFA artifact from %s (%d layers).", artifact_dir, applied)
    return model


def load_unlearning_artifact(model, artifact_path):
    artifact_dir = Path(artifact_path)
    config_path = artifact_dir / UNLEARNING_ARTIFACT_CONFIG
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    artifact_type = config.get("type")
    if artifact_type == "manu":
        return _load_manu_artifact(model, artifact_dir, config)
    if artifact_type == "smfa":
        return _load_smfa_artifact(model, artifact_dir, config)
    raise ValueError(f"Unsupported unlearning artifact type: {artifact_type!r}")


def get_model(model_cfg: DictConfig, peft_cfg=None):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    adapter_path = model_cfg.get("adapter_path", None)
    merge_adapter = model_cfg.get("merge_adapter", False)
    torch_dtype = get_dtype(model_args)
    quantization_config, quantization_load_kwargs = get_quantization_config(model_cfg)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    load_kwargs = dict(model_args)
    load_kwargs.update(quantization_load_kwargs)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **load_kwargs,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    if adapter_path:
        artifact_config = Path(str(adapter_path)) / UNLEARNING_ARTIFACT_CONFIG
        if artifact_config.is_file():
            model = load_unlearning_artifact(model, adapter_path)
        elif merge_adapter:
            model = load_merged_peft_adapter(model, adapter_path)
        else:
            model = load_trainable_peft_adapter(model, adapter_path, peft_cfg)
    else:
        model = apply_peft(model, peft_cfg)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(Qwen2_5_VLForConditionalGeneration)
_register_model(LlavaForConditionalGeneration)
_register_model(LlavaNextForConditionalGeneration)
_register_model(Gemma3ForConditionalGeneration)
_register_model(ProbedLlamaForCausalLM)
_register_model(ProbedLlavaForConditionalGeneration)
_register_model(ProbedLlavaNextForConditionalGeneration)
_register_model(ProbedQwen2_5_VLForConditionalGeneration)
_register_model(ProbedGemma3ForConditionalGeneration)
