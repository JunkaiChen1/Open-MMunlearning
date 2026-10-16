"""Model forward helpers shared by membership-inference attacks."""

import warnings

import torch
from torch import nn

from data.utils import IGNORE_INDEX


def align_logits_to_labels(input_ids, logits, labels, model, attention_mask=None):
    """Keep answer positions aligned when a model expands image placeholders."""
    text_length = labels.shape[-1]
    extra_length = logits.shape[-2] - text_length
    if extra_length == 0:
        return logits
    if extra_length < 0:
        raise ValueError("MIA logits are shorter than their text-token labels.")

    config = getattr(model, "config", None)
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for key in ("image_token_id", "image_token_index"):
        token_id = getattr(config, key, None)
        if token_id is not None:
            image_mask |= input_ids == token_id
    image_counts = image_mask.sum(-1)
    if not image_mask.any():
        if input_ids.shape[0] != 1:
            raise ValueError(
                "Cannot align expanded MIA batches without image token positions; "
                "configure image_token_id/image_token_index or use batch_size=1."
            )
        # Single-sample visual prefixes preserve the answer's distance from
        # the end of the sequence, as in the MLLMU generation protocol.
        return logits[:, -text_length:, :]

    image_length = getattr(config, "image_seq_length", None)
    if input_ids.shape[0] > 1 and (
        image_length is None
        or int(image_counts.max()) * (image_length - 1) != extra_length
    ):
        raise ValueError(
            "Cannot infer per-row image expansion for this MIA batch; "
            "use batch_size=1 for variable-length image features."
        )
    if torch.all(image_counts == image_counts[0]):
        row_extra = torch.full_like(image_counts, extra_length)
    else:
        # Mixed image/text batches need each row's expansion length, rather
        # than a common shift by the padded batch's maximum expansion.
        row_extra = image_counts * (image_length - 1)

    left_padding = False
    if torch.any(row_extra != extra_length):
        if attention_mask is not None and (attention_mask[:, 0] == 0).any():
            left_padding = True
        elif attention_mask is not None and (attention_mask[:, -1] == 0).any():
            left_padding = False
        else:
            padding_side = getattr(model, "padding_side", None) or getattr(
                config, "padding_side", None
            )
            if padding_side not in ("left", "right"):
                raise ValueError(
                    "Expanded MIA rows require the model's output padding_side "
                    "when the input batch contains no padding; use batch_size=1."
                )
            left_padding = padding_side == "left"

    rows = []
    for row, mask in enumerate(image_mask):
        extra = int(row_extra[row])
        image_positions = mask.nonzero(as_tuple=True)[0]
        positions = torch.arange(text_length, device=logits.device)
        if image_positions.numel():
            first_image = int(image_positions[0])
            last_image = int(image_positions[-1])
            if (labels[row, : last_image + 1] != IGNORE_INDEX).any():
                raise ValueError(
                    "Expanded MIA labels must score the answer after the images."
                )
            # The image slot keeps its final visual logit, which predicts the
            # following text token. Only the post-image answer is supervised.
            positions[first_image:] += extra
        if left_padding:
            positions += extra_length - extra
        rows.append(logits[row].index_select(0, positions))
    return torch.stack(rows)


def evaluate_probability(model, batch):
    """Evaluate model probabilities and average token-level loss for a given batch."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    labels = batch["labels"]
    logits = align_logits_to_labels(
        batch["input_ids"], output.logits, labels, model, batch.get("attention_mask")
    )
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def tokenwise_logprobs(
    model, batch, grad=False, return_labels=False, align_logits=None
):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    if align_logits is not None:
        logits = align_logits(batch["input_ids"], logits, batch["labels"], model)
    else:
        logits = align_logits_to_labels(
            batch["input_ids"],
            logits,
            batch["labels"],
            model,
            batch.get("attention_mask"),
        )
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    # Some model-side image placeholders (for example -200) are not vocabulary
    # entries. Their labels are ignored and their gathered values are unused.
    ignored_placeholders = ((next_tokens < 0) | (next_tokens >= V)) & (
        batch["labels"][:, 1:].unsqueeze(-1) == IGNORE_INDEX
    )
    next_tokens = next_tokens.masked_fill(ignored_placeholders, 0)
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_vocab_logprobs(
    model, batch, grad=False, return_labels=False, align_logits=None
):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    if align_logits is not None:
        logits = align_logits(batch["input_ids"], logits, batch["labels"], model)
    else:
        logits = align_logits_to_labels(
            batch["input_ids"],
            logits,
            batch["labels"],
            model,
            batch.get("attention_mask"),
        )
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts
