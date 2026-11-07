from torch.nn.utils import clip_grad_norm_

from seq2seq.training import (
    token_cross_entropy,
    token_accuracy,
)


def train_step(
    model,
    batch,
    optimizer,
    pad_id,
    max_grad_norm,
):
    model.train()

    input_ids, targets = batch

    device = next(
        model.parameters()
    ).device

    input_ids = input_ids.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(input_ids)

    loss = token_cross_entropy(
        logits,
        targets,
        pad_id,
    )

    correct_tokens, valid_tokens = token_accuracy(
        logits,
        targets,
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