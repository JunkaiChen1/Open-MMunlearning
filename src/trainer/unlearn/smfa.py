"""Native SMFA (Sculpted Memory Forgetting Adapter) trainer.

SMFA learns separate multimodal-forget, text-forget, and retain adapters in a
single Trainer run. The forget adapters learn IDK targets together with retain
answers. At load time their conflicting deltas are sculpted against the retain
adapter and applied to the fine-tuned base model.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import torch
from accelerate.utils import DistributedType
from peft import PeftModel, get_peft_model

from trainer.unlearn.base import UnlearnTrainer


logger = logging.getLogger(__name__)


class SMFA(UnlearnTrainer):
    """Framework-native three-adapter SMFA training and sculpting."""

    def __init__(
        self,
        multi_k=5.0,
        text_k=10.0,
        lora_adapter_names=None,
        source_adapter_path=None,
        *args,
        **kwargs,
    ):
        model = kwargs.get("model")
        if not isinstance(model, PeftModel):
            raise TypeError("SMFA requires peft=lora so it can maintain three adapters")
        multi_k = float(multi_k)
        text_k = float(text_k)
        if multi_k < 0 or text_k < 0:
            raise ValueError("SMFA multi_k and text_k must be non-negative")
        names = list(lora_adapter_names or ("MFA_multi", "MFA_text", "RA"))
        if len(names) != 3 or len(set(names)) != 3:
            raise ValueError("SMFA needs three distinct adapter names")

        # All unlearning updates must start from the fine-tuned model represented
        # by the incoming default adapter. Merge it before creating SMFA's three
        # independent training adapters.
        source_name = model.active_adapter
        if isinstance(source_name, list):
            source_name = source_name[0]
        source_config = copy.deepcopy(model.peft_config[source_name])
        base_model = model.merge_and_unload(adapter_names=[source_name])
        if hasattr(base_model, "peft_config"):
            delattr(base_model, "peft_config")
        model = get_peft_model(
            base_model,
            copy.deepcopy(source_config),
            adapter_name=names[0],
        )
        for name in names[1:]:
            model.add_adapter(name, copy.deepcopy(source_config))
        # PEFT promotes the adapter created by get_peft_model to fp32, but
        # add_adapter inherits the bf16/fp16 base dtype. Keep all three
        # trainable adapters at the same precision so their optimizer states
        # and sculpted deltas are comparable.
        for parameter_name, parameter in model.named_parameters():
            if parameter.dtype not in (torch.float16, torch.bfloat16):
                continue
            if any(f".{name}." in parameter_name for name in names):
                parameter.data = parameter.data.float()
        kwargs["model"] = model

        super().__init__(*args, **kwargs)
        self.multi_k = multi_k
        self.text_k = text_k
        self.adapter_names = tuple(names)
        self.source_adapter_path = source_adapter_path
        if not self.source_adapter_path:
            raise ValueError("SMFA requires source_adapter_path for reloadable output")
        self._activate(self.adapter_names[0])
        logger.info("SMFA adapters: %s", ", ".join(self.adapter_names))

    def _activate(self, adapter):
        """Select an adapter while keeping all three adapters in the optimizer."""
        self.model.set_adapter(adapter)
        for name, parameter in self.model.named_parameters():
            if "lora_" in name and any(adapter_name in name for adapter_name in self.adapter_names):
                parameter.requires_grad_(True)

    @staticmethod
    def _view(batch, text_only):
        if text_only:
            if "text_only" not in batch:
                raise ValueError(
                    "SMFA requires collator.DataCollatorForMultimodalQADataset "
                    "with include_text_only=true"
                )
            batch = batch["text_only"]
        return batch

    def _run_adapter_loss(self, model, batch, adapter, text_only):
        self._activate(adapter)
        view = self._view(batch, text_only)
        model_inputs = self._prepare_model_inputs(view)
        with self.compute_loss_context_manager():
            outputs = model(**model_inputs)
            loss = outputs.loss
        # LLaVA fp16 can overflow after adversarial updates. Fail the trial
        # rather than writing a checkpoint with poisoned adapter weights.
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SMFA received a non-finite {adapter} loss; use bfloat16 or a "
                "more conservative update configuration"
            )
        return loss

    def _sanitize_gradients(self):
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    "SMFA produced non-finite gradients; use bfloat16 or a more "
                    "conservative update configuration"
                )

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.use_apex:
            raise NotImplementedError("SMFA does not support Apex training")
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        accumulation = self.args.gradient_accumulation_steps
        backward_kwargs = {}
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            backward_kwargs["scale_wrt_gas"] = False

        forget = inputs["forget"]
        if not {"original", "alternate"}.issubset(forget):
            raise ValueError(
                "SMFA requires the forget dataset to provide original and IDK "
                "alternate samples; configure data.forget.*.args.idk_path"
            )

        # The reference implementation minimizes IDK-forget and true retain
        # examples for each MFA adapter, then trains RA on multimodal and text
        # retain examples. Combining the six losses into one optimizer step
        # preserves those objectives within the common Trainer loop.
        losses = []
        stages = (
            (self.adapter_names[0], forget["alternate"], False),
            (self.adapter_names[0], inputs["retain"], False),
            (self.adapter_names[1], forget["alternate"], True),
            (self.adapter_names[1], inputs["retain"], True),
            (self.adapter_names[2], inputs["retain"], False),
            (self.adapter_names[2], inputs["retain"], True),
        )
        loss_scale = accumulation * len(stages)
        for adapter, batch, text_only in stages:
            loss = self._run_adapter_loss(model, batch, adapter, text_only)
            self.accelerator.backward(loss / loss_scale, **backward_kwargs)
            self._sanitize_gradients()
            losses.append(loss.detach())
        return torch.stack(losses).mean() / accumulation

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._activate(self.adapter_names[2])
        retain_inputs = self._prepare_model_inputs(inputs["retain"])
        outputs = model(**retain_inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        if self.accelerator.is_main_process:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            model.save_pretrained(
                output_dir,
                selected_adapters=list(self.adapter_names),
                safe_serialization=getattr(self.args, "save_safetensors", True),
            )
            artifact = {
                "version": 1,
                "type": "smfa",
                "source_adapter_path": str(Path(self.source_adapter_path).resolve()),
                "adapters": list(self.adapter_names),
                "multi_k": self.multi_k,
                "text_k": self.text_k,
            }
            with (Path(output_dir) / "unlearning_artifact.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(artifact, handle, indent=2, sort_keys=True)
                handle.write("\n")
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        self.accelerator.wait_for_everyone()
