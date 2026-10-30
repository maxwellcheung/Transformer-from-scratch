import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def token_cross_entropy(logits,targets,pad_id,):
    valid_mask = targets.ne(pad_id)
    valid_tokens = valid_mask.sum().item()

    if valid_tokens == 0:
        raise ValueError(
            'targets must contain at least one valid token'
        )

    vocab_size = logits.size(-1)

    flat_logits = logits.reshape(
        -1,
        vocab_size,
    )

    flat_targets = targets.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=pad_id,
    )

    return loss


def token_accuracy(
    logits,
    targets,
    pad_id,
):
    valid_mask = targets.ne(pad_id)
    valid_tokens = valid_mask.sum().item()

    if valid_tokens == 0:
        raise ValueError(
            'targets must contain at least one valid token'
        )

    predictions = logits.argmax(dim=-1)

    correct_mask = (
        predictions.eq(targets)
        & valid_mask
    )

    correct_tokens = correct_mask.sum().item()

    return correct_tokens, valid_tokens

def train_step(model, batch, optimizer, pad_id, max_grad_norm,):
    model.train()

    src_tokens, tgt_input, tgt_output = batch

    device = next(
        model.parameters()
    ).device

    src_tokens = src_tokens.to(device)
    tgt_input = tgt_input.to(device)
    tgt_output = tgt_output.to(device)

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        src_tokens,
        tgt_input,
    )

    loss = token_cross_entropy(
        logits,
        tgt_output,
        pad_id,
    )

    correct_tokens, valid_tokens = token_accuracy(
        logits,
        tgt_output,
        pad_id,
    )

    loss.backward()

    gradient_norm = clip_grad_norm_(
        model.parameters(),
        max_grad_norm,
    )

    optimizer.step()

    metrics = {
        'loss': loss.item(),
        'correct_tokens': correct_tokens,
        'valid_tokens': valid_tokens,
        'gradient_norm': gradient_norm.item(),
    }

    return metrics