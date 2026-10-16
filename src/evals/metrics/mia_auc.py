"""MIA score aggregation and ROC AUC with the original label orientation."""

import numpy as np

from evals.base import Evaluator
from evals.metrics import register_metric

from .base import unlearning_metric


class MIAEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MIA", eval_cfg, **kwargs)


def summarize_attack_scores(scores, indices):
    return {
        "agg_value": float(np.mean(scores)),
        "value_by_index": {
            str(index): {"score": float(score)} for index, score in zip(indices, scores)
        },
    }


def aggregate_mia_auc(output):
    from sklearn.metrics import roc_auc_score

    forget_scores = [
        item["score"] for item in output["forget"]["value_by_index"].values()
    ]
    holdout_scores = [
        item["score"] for item in output["holdout"]["value_by_index"].values()
    ]
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array([0] * len(forget_scores) + [1] * len(holdout_scores))
    value = roc_auc_score(labels, scores)
    return {**output, "auc": value, "agg_value": value}


@register_metric
@unlearning_metric("mia_auc")
def mia_auc(
    model,
    *,
    pre_compute=None,
    attack="loss",
    data=None,
    collators=None,
    batch_size=1,
    tokenizer=None,
    reference_model=None,
    k=0.2,
    p=2,
    **kwargs,
):
    if pre_compute is not None:
        return aggregate_mia_auc(
            {key: pre_compute[key] for key in ("forget", "holdout")}
        )
    if data is None or collators is None:
        raise ValueError(
            "mia_auc requires forget/holdout datasets and an indexed collator."
        )
    arguments = _attack_arguments(attack, tokenizer, reference_model, k, p)
    return run_mia_auc(
        get_attacker(attack), model, data, collators, batch_size, **arguments
    )


def load_reference_model(config):
    """Load the same configurable model handlers used by train/eval entry points."""
    if config is None:
        raise ValueError(
            "The reference attack requires a reference_model configuration."
        )
    if not isinstance(config, dict):
        from omegaconf import DictConfig

        if not isinstance(config, DictConfig):
            return config
    from omegaconf import OmegaConf

    from model import get_model

    # get_model consumes parts of its config, so retain the reusable YAML value.
    config = (
        OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        if not isinstance(config, dict)
        else OmegaConf.create(config)
    )
    reference, _ = get_model(config)
    reference.eval()
    return reference


def _attack_arguments(attack, tokenizer, reference_model, k, p):
    arguments = {"k": k, "p": p, "tokenizer": tokenizer}
    if attack == "ref":
        arguments["reference_model"] = load_reference_model(reference_model)
    return arguments


@register_metric
@unlearning_metric("mia_attack")
def mia_attack(
    model,
    *,
    attack="loss",
    data,
    collators,
    batch_size=1,
    tokenizer=None,
    reference_model=None,
    k=0.2,
    p=2,
    **kwargs,
):
    arguments = _attack_arguments(attack, tokenizer, reference_model, k, p)
    return get_attacker(attack)(
        model=model, data=data, collator=collators, batch_size=batch_size, **arguments
    ).attack()


def get_attacker(attack: str):
    from .base import AllAttacks
    from .gradnorm import GradNormAttack
    from .loss_mia import LOSSAttack
    from .min_k import MinKProbAttack
    from .min_k_plus_plus import MinKPlusPlusAttack
    from .reference_loss import ReferenceAttack
    from .zlib_mia import ZLIBAttack

    if isinstance(attack, str) and attack not in AllAttacks._value2member_map_:
        try:
            attack = AllAttacks(attack)
        except ValueError:
            pass
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.GRADNORM: GradNormAttack,
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls


def run_mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys "forget" and "holdout".
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    # Build attack arguments from common parameters and any extras.
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
    }
    attack_args.update(kwargs)

    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
        "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
    }
    return aggregate_mia_auc(output)
