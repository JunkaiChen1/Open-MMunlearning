"""Probe variants for the text decoder of causal and multimodal models."""

from copy import deepcopy
import gc
import logging
from typing import Optional, Type

import torch
from torch import nn
from transformers import AutoConfig


logger = logging.getLogger(__name__)


def _disable_tied_output_embeddings(config) -> None:
    """Keep the probe head independent from the input embedding matrix."""
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False
    if hasattr(config, "text_config"):
        _disable_tied_output_embeddings(config.text_config)


def _get_decoder(model) -> nn.Module:
    decoder = model.get_decoder()
    if not hasattr(decoder, "layers"):
        raise ValueError(
            f"{type(model).__name__} does not expose decoder layers "
            "compatible with probing."
        )
    return decoder


def _set_decoder_depth(model, decoder: nn.Module, n_layers: int) -> None:
    decoder.layers = nn.ModuleList(decoder.layers[:n_layers])

    # The decoder is the source of truth at runtime. Update every config view so
    # saved probes can be reloaded by the ordinary Transformers auto classes.
    if hasattr(decoder, "config") and hasattr(decoder.config, "num_hidden_layers"):
        decoder.config.num_hidden_layers = n_layers
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = n_layers
    if hasattr(model.config, "text_config") and hasattr(
        model.config.text_config, "num_hidden_layers"
    ):
        model.config.text_config.num_hidden_layers = n_layers


def _load_reference_head(
    reference_model_class: Type,
    head_pretrained_model_name_or_path: str,
    load_kwargs: dict,
    expected_head: nn.Module,
) -> nn.Module:
    # Loading the reference only to copy its output head must not place a second
    # full multimodal model on the active GPU.
    reference_kwargs = dict(load_kwargs)
    for key in ("device_map", "max_memory", "offload_folder", "offload_state_dict"):
        reference_kwargs.pop(key, None)

    reference_model = reference_model_class.from_pretrained(
        head_pretrained_model_name_or_path, **reference_kwargs
    )
    reference_head = reference_model.get_output_embeddings()
    if reference_head.weight.shape != expected_head.weight.shape:
        raise ValueError(
            "The reference output head shape does not match the probed model: "
            f"expected {tuple(expected_head.weight.shape)}, got "
            f"{tuple(reference_head.weight.shape)}."
        )

    copied_head = deepcopy(reference_head).to(
        device=expected_head.weight.device, dtype=expected_head.weight.dtype
    )
    del reference_model
    return copied_head


def _configure_probe_model(
    model,
    *,
    n_layers: int,
    freeze_base_model: bool,
    head_pretrained_model_name_or_path: Optional[str],
    reinitialize_output_head: bool,
    reference_model_class: Type,
    load_kwargs: dict,
):
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer.")

    decoder = _get_decoder(model)
    available_layers = len(decoder.layers)
    retained_layers = min(n_layers, available_layers)
    if retained_layers != n_layers:
        logger.warning(
            "Requested %s decoder layers, but %s only has %s; retaining all layers.",
            n_layers,
            type(model).__name__,
            available_layers,
        )

    _set_decoder_depth(model, decoder, retained_layers)
    _disable_tied_output_embeddings(model.config)

    output_head = model.get_output_embeddings()
    if reinitialize_output_head:
        if head_pretrained_model_name_or_path is None:
            logger.info("Initialising a new output head for %s", type(model).__name__)
            model._init_weights(output_head)
        else:
            logger.info(
                "Initialising the output head for %s from %s",
                type(model).__name__,
                head_pretrained_model_name_or_path,
            )
            output_head = _load_reference_head(
                reference_model_class,
                head_pretrained_model_name_or_path,
                load_kwargs,
                output_head,
            )
            model.set_output_embeddings(output_head)
    elif head_pretrained_model_name_or_path is not None:
        raise ValueError(
            "head_pretrained_model_name_or_path cannot be set when "
            "reinitialize_output_head is false."
        )

    if freeze_base_model:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_output_embeddings().parameters():
            parameter.requires_grad = True

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        "Initialised %s with %s/%s decoder layers and %s trainable parameters.",
        type(model).__name__,
        retained_layers,
        available_layers,
        trainable,
    )
    return model


def _load_probed_model(
    model_class: Type,
    pretrained_model_name_or_path: str,
    head_pretrained_model_name_or_path: Optional[str],
    n_layers: int,
    freeze_base_model: bool,
    reinitialize_output_head: bool,
    **kwargs,
):
    """Load a model lazily, then configure its language decoder as a probe."""
    config, unused_kwargs = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
    )
    _disable_tied_output_embeddings(config)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path, config=config, **unused_kwargs
    )
    return _configure_probe_model(
        model,
        n_layers=n_layers,
        freeze_base_model=freeze_base_model,
        head_pretrained_model_name_or_path=head_pretrained_model_name_or_path,
        reinitialize_output_head=reinitialize_output_head,
        reference_model_class=model_class,
        load_kwargs=unused_kwargs,
    )


class _ProbedDecoderLoader:
    """Lazy handler shared by all decoder probes."""

    @staticmethod
    def _model_class() -> Type:
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        head_pretrained_model_name_or_path: Optional[str] = None,
        n_layers: int = 100,
        freeze_base_model: bool = True,
        reinitialize_output_head: bool = True,
        **kwargs,
    ):
        return _load_probed_model(
            cls._model_class(),
            pretrained_model_name_or_path,
            head_pretrained_model_name_or_path,
            n_layers,
            freeze_base_model,
            reinitialize_output_head,
            **kwargs,
        )


class ProbedLlamaForCausalLM(_ProbedDecoderLoader):
    """A Llama probe that retains decoder layers and trains only its output head."""

    @staticmethod
    def _model_class() -> Type:
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM


class ProbedLlavaForConditionalGeneration(_ProbedDecoderLoader):
    """LLaVA 1.5 probe preserving the vision tower and multimodal projector."""

    @staticmethod
    def _model_class() -> Type:
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration


class ProbedLlavaNextForConditionalGeneration(_ProbedDecoderLoader):
    """LLaVA-NeXT probe preserving all image feature packing and projection."""

    @staticmethod
    def _model_class() -> Type:
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration


class ProbedQwen2_5_VLForConditionalGeneration(_ProbedDecoderLoader):
    """Qwen2.5-VL probe preserving the visual encoder and multimodal RoPE path."""

    @staticmethod
    def _model_class() -> Type:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration


class ProbedGemma3ForConditionalGeneration(_ProbedDecoderLoader):
    """Gemma 3 probe preserving its vision tower and multimodal projector."""

    @staticmethod
    def _model_class() -> Type:
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration
