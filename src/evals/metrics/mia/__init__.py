"""Compatibility layer for the attack implementations."""

from attacks.mia.all_attacks import AllAttacks, Attack
from attacks.mia.gradnorm import GradNormAttack
from attacks.mia.loss import LOSSAttack
from attacks.mia.min_k import MinKProbAttack
from attacks.mia.min_k_plus_plus import MinKPlusPlusAttack
from attacks.mia.reference import ReferenceAttack
from attacks.mia.utils import mia_auc
from attacks.mia.zlib import ZLIBAttack

from evals.metrics.base import unlearning_metric
from transformers import AutoModelForCausalLM
import logging

logger = logging.getLogger("metrics")


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    return mia_auc(LOSSAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"])


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    return mia_auc(MinKProbAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"], k=kwargs["k"])


@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    return mia_auc(MinKPlusPlusAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"], k=kwargs["k"])


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    return mia_auc(GradNormAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"], p=kwargs["p"])


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    return mia_auc(ZLIBAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"], tokenizer=kwargs.get("tokenizer"))


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"], torch_dtype=model.dtype, device_map={"": model.device}
    )
    return mia_auc(ReferenceAttack, model, data=kwargs["data"], collator=kwargs["collators"], batch_size=kwargs["batch_size"], reference_model=reference_model)


__all__ = [
    "AllAttacks", "Attack", "LOSSAttack", "MinKProbAttack", "MinKPlusPlusAttack",
    "GradNormAttack", "ZLIBAttack", "ReferenceAttack", "mia_auc", "mia_loss",
    "mia_min_k", "mia_min_k_plus_plus", "mia_gradnorm", "mia_zlib", "mia_reference",
]
