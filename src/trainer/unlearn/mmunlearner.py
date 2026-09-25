import logging
import os

import torch
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader

from trainer.unlearn.grad_diff import GradDiff


logger = logging.getLogger(__name__)


def compute_saliency_mask(forget_fisher, preserve_fisher, threshold, epsilon=1e-15):
    """Return the MMUnlearner parameter mask for matching Fisher tensors."""
    masks = {}
    for name, forget_value in forget_fisher.items():
        preserve_value = preserve_fisher.get(name)
        if preserve_value is None:
            continue
        if forget_value.shape != preserve_value.shape:
            raise ValueError(
                f"Fisher shape mismatch for {name}: "
                f"{tuple(forget_value.shape)} != {tuple(preserve_value.shape)}"
            )
        ratio = (forget_value + epsilon) / (preserve_value + epsilon)
        masks[name] = ratio >= threshold
    return masks


class MMUnlearner(GradDiff):
    """Geometry-constrained gradient ascent for multimodal unlearning.

    The saliency stage follows the official MMUnlearner implementation: squared
    forget gradients are divided by squared preserve gradients and thresholded.
    During optimization, the resulting mask is applied only to the forget-loss
    gradient; retain gradients remain unmasked.
    """

    def __init__(
        self,
        saliency_threshold=1.0,
        saliency_epsilon=1e-15,
        mask_num_batches=None,
        mask_parameter_patterns=None,
        preserve_forget_text=True,
        preserve_retain_text=True,
        preserve_retain=True,
        preserve_retain_view="multimodal",
        mask_path=None,
        mask_save_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if saliency_threshold < 0:
            raise ValueError("saliency_threshold must be non-negative")
        if mask_num_batches is not None and mask_num_batches <= 0:
            raise ValueError("mask_num_batches must be positive or null")
        if preserve_retain_view not in ("multimodal", "text_only"):
            raise ValueError(
                "preserve_retain_view must be 'multimodal' or 'text_only'"
            )

        self.saliency_threshold = float(saliency_threshold)
        self.saliency_epsilon = float(saliency_epsilon)
        self.mask_num_batches = mask_num_batches
        self.mask_parameter_patterns = tuple(mask_parameter_patterns or ())
        self.preserve_forget_text = preserve_forget_text
        self.preserve_retain_text = preserve_retain_text
        self.preserve_retain = preserve_retain
        self.preserve_retain_view = preserve_retain_view
        self.mask_path = mask_path
        self.mask_save_path = mask_save_path

        self.gradient_mask = None
        self._mask_hooks = []
        self._mask_forget_gradient = False

    def _analysis_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _select_view(self, batch, view):
        if view == "text_only":
            if "text_only" not in batch:
                raise ValueError(
                    "MMUnlearner needs the collator text-only view. Set "
                    "collator.DataCollatorForMultimodalQADataset.args."
                    "include_text_only=true."
                )
            batch = batch["text_only"]
        return self._prepare_model_inputs(batch)

    def _matches_parameter_scope(self, name):
        if not self.mask_parameter_patterns:
            return True
        return any(pattern in name for pattern in self.mask_parameter_patterns)

    def _eligible_parameters(self, model):
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and self._matches_parameter_scope(name)
        }
        if not parameters:
            patterns = ", ".join(self.mask_parameter_patterns) or "<all>"
            raise ValueError(
                "MMUnlearner found no trainable parameters in mask scope: " + patterns
            )
        return parameters

    def _accumulate_fisher(self, model, parameters, sources):
        fisher = {}
        batches = 0
        model.train()

        for loader, view in sources:
            for batch_index, batch in enumerate(loader):
                if (
                    self.mask_num_batches is not None
                    and batch_index >= self.mask_num_batches
                ):
                    break
                batch = self._prepare_inputs(batch)
                model_inputs = self._select_view(batch, view)
                model.zero_grad(set_to_none=True)
                with self.compute_loss_context_manager():
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                self.accelerator.backward(loss)

                with torch.no_grad():
                    for name, parameter in parameters.items():
                        if parameter.grad is None:
                            continue
                        value = parameter.grad.detach().float().square().cpu()
                        if name in fisher:
                            fisher[name].add_(value)
                        else:
                            fisher[name] = value
                batches += 1
                del outputs, loss, model_inputs, batch

        model.zero_grad(set_to_none=True)
        if batches == 0:
            raise ValueError("MMUnlearner saliency analysis received no batches")
        for value in fisher.values():
            value.div_(batches)
        return fisher, batches

    def _build_gradient_mask(self, model):
        train_dataset = self.train_dataset
        if not hasattr(train_dataset, "forget"):
            raise TypeError(
                "MMUnlearner requires ForgetRetainDataset with a forget split"
            )

        parameters = self._eligible_parameters(model)
        forget_loader = self._analysis_loader(train_dataset.forget)
        forget_fisher, forget_batches = self._accumulate_fisher(
            model, parameters, [(forget_loader, "multimodal")]
        )

        preserve_sources = []
        if self.preserve_forget_text:
            preserve_sources.append((forget_loader, "text_only"))
        if self.preserve_retain_text:
            if getattr(train_dataset, "retain", None) is None:
                raise ValueError(
                    "preserve_retain_text=true requires a retain dataset"
                )
            preserve_sources.append(
                (self._analysis_loader(train_dataset.retain), "text_only")
            )
        if self.preserve_retain:
            if getattr(train_dataset, "retain", None) is None:
                raise ValueError(
                    "preserve_retain=true requires a retain dataset"
                )
            preserve_sources.append(
                (
                    self._analysis_loader(train_dataset.retain),
                    self.preserve_retain_view,
                )
            )
        if not preserve_sources:
            raise ValueError("MMUnlearner requires at least one preserve source")

        preserve_fisher, preserve_batches = self._accumulate_fisher(
            model, parameters, preserve_sources
        )
        masks = compute_saliency_mask(
            forget_fisher,
            preserve_fisher,
            threshold=self.saliency_threshold,
            epsilon=self.saliency_epsilon,
        )
        if not masks:
            raise ValueError(
                "MMUnlearner could not build a mask for any trainable parameter"
            )

        logger.info(
            "MMUnlearner saliency mask built from %d forget and %d preserve batches",
            forget_batches,
            preserve_batches,
        )
        if self.mask_save_path and self.accelerator.is_main_process:
            mask_dir = os.path.dirname(os.path.abspath(self.mask_save_path))
            os.makedirs(mask_dir, exist_ok=True)
            torch.save(
                {
                    "weight": masks,
                    "threshold": self.saliency_threshold,
                    "parameter_patterns": self.mask_parameter_patterns,
                },
                self.mask_save_path,
            )
            logger.info("MMUnlearner saliency mask saved to %s", self.mask_save_path)
        return masks

    def _load_gradient_mask(self):
        mask_data = torch.load(self.mask_path, map_location="cpu", weights_only=True)
        if isinstance(mask_data, dict) and "weight" in mask_data:
            mask_data = mask_data["weight"]
        if not isinstance(mask_data, dict):
            raise TypeError("MMUnlearner mask file must contain a tensor dictionary")
        return mask_data

    @staticmethod
    def _find_mask(name, parameter, masks):
        candidates = []
        if name in masks:
            candidates.append(masks[name])
        else:
            for mask_name, mask in masks.items():
                if name.endswith(mask_name) or mask_name.endswith(name):
                    candidates.append(mask)
        candidates = [mask for mask in candidates if mask.shape == parameter.shape]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous MMUnlearner masks for parameter {name}")
        return candidates[0] if candidates else None

    def _install_gradient_hooks(self, model, masks):
        active = 0
        total = 0
        matched = 0
        device_masks = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            mask = self._find_mask(name, parameter, masks)
            if mask is None:
                continue
            mask = mask.to(device=parameter.device, dtype=torch.bool)
            device_masks[name] = mask
            active += mask.count_nonzero().item()
            total += mask.numel()
            matched += 1

            def mask_hook(gradient, parameter_mask=mask):
                if self._mask_forget_gradient:
                    return gradient * parameter_mask
                return gradient

            self._mask_hooks.append(parameter.register_hook(mask_hook))

        if matched == 0:
            raise ValueError(
                "MMUnlearner mask did not match any trainable model parameter. "
                "Generate the mask with the same model and PEFT configuration."
            )
        self.gradient_mask = device_masks
        logger.info(
            "MMUnlearner installed masks for %d parameters; active fraction %.6f",
            matched,
            active / total if total else 0.0,
        )

    def _ensure_gradient_mask(self, model):
        if self.gradient_mask is not None:
            return
        masks = (
            self._load_gradient_mask()
            if self.mask_path
            else self._build_gradient_mask(model)
        )
        self._install_gradient_hooks(model, masks)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = self._select_view(inputs["forget"], "multimodal")
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = self._select_view(inputs["retain"], "multimodal")
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.use_apex:
            raise NotImplementedError("MMUnlearner does not support Apex training")

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        self._ensure_gradient_mask(model)

        backward_kwargs = {}
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            backward_kwargs["scale_wrt_gas"] = False
        accumulation = self.args.gradient_accumulation_steps

        forget_inputs = self._select_view(inputs["forget"], "multimodal")
        with self.compute_loss_context_manager():
            forget_outputs = model(**forget_inputs)
            forget_loss = -forget_outputs.loss
            if self.args.n_gpu > 1:
                forget_loss = forget_loss.mean()
        scaled_forget_loss = self.gamma * forget_loss / accumulation
        self._mask_forget_gradient = True
        try:
            self.accelerator.backward(scaled_forget_loss, **backward_kwargs)
        finally:
            self._mask_forget_gradient = False

        retain_loss = torch.zeros_like(forget_loss)
        if self.alpha != 0:
            retain_inputs = self._select_view(inputs["retain"], "multimodal")
            with self.compute_loss_context_manager():
                retain_loss = self.compute_retain_loss(model, retain_inputs)
                if self.args.n_gpu > 1:
                    retain_loss = retain_loss.mean()
            self.accelerator.backward(
                self.alpha * retain_loss / accumulation, **backward_kwargs
            )

        reported_loss = (
            self.gamma * forget_loss.detach() + self.alpha * retain_loss.detach()
        ) / accumulation
        return reported_loss
