import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    outputs = model(**inputs)
    labels = inputs.get("labels")
    if labels is None:
        raise ValueError("KL retention loss requires labels to identify supervised tokens")
    if outputs.logits.shape[:-1] != labels.shape:
        raise ValueError(
            "KL logits and labels must have matching batch and sequence dimensions, got "
            f"logits {tuple(outputs.logits.shape)} and labels {tuple(labels.shape)}"
        )
    if ref_outputs.logits.shape != outputs.logits.shape:
        raise ValueError(
            "KL current and reference logits must have identical shapes, got "
            f"{tuple(outputs.logits.shape)} and {tuple(ref_outputs.logits.shape)}"
        )

    # For a causal LM, logits at position t predict the label at position t + 1.
    # Restrict retention to answer tokens instead of regularizing prompt, image,
    # and padding positions whose labels are masked with -100.
    valid_tokens = labels[..., 1:].ne(-100)
    if not valid_tokens.any():
        raise ValueError("KL retention loss requires at least one supervised token")

    current_log_probs = F.log_softmax(outputs.logits[..., :-1, :].float(), dim=-1)
    ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :].float(), dim=-1)
    kl_loss = F.kl_div(
        current_log_probs[valid_tokens],
        ref_log_probs[valid_tokens],
        reduction="batchmean",
        log_target=True,
    )
    return kl_loss, outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    outputs = model(**inputs)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].to(shift_logits.device).contiguous()

    with torch.no_grad():
        shift_teacher_logits = ref_model(**inputs).logits[..., :-1, :].contiguous()

    flat_labels = shift_labels.reshape(-1)
    valid_tokens = flat_labels.ne(-100)
    if not valid_tokens.any():
        raise ValueError("UNDIAL requires at least one non-masked answer token")

    vocab_size = shift_logits.size(-1)
    teacher_logits = shift_teacher_logits.reshape(-1, vocab_size)[valid_tokens].float()
    valid_labels = flat_labels[valid_tokens]
    teacher_logits = teacher_logits.scatter_add(
        dim=-1,
        index=valid_labels.unsqueeze(-1),
        src=teacher_logits.new_full((valid_labels.numel(), 1), -float(beta)),
    )
    soft_targets = F.softmax(teacher_logits, dim=-1)

    student_logits = shift_logits.reshape(-1, vocab_size)[valid_tokens].float()
    loss = -(soft_targets * F.log_softmax(student_logits, dim=-1)).sum(dim=-1)
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
