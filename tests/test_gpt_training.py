import math

import torch

from gpt.model import GPTConfig, GPTModel


def test_gpt_train_step_updates_model():
    from gpt.training import train_step

    torch.manual_seed(0)

    config = GPTConfig(
        vocab_size=12,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=16,
        dropout=0.0,
        pad_id=0,
    )

    model = GPTModel(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    input_ids = torch.tensor([
        [1, 4, 5, 0],
        [1, 6, 7, 8],
    ])

    targets = torch.tensor([
        [4, 5, 2, 0],
        [6, 7, 8, 2],
    ])

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = train_step(
        model=model,
        batch=(input_ids, targets),
        optimizer=optimizer,
        pad_id=0,
        max_grad_norm=1.0,
    )

    parameters_after = list(model.parameters())

    assert math.isfinite(metrics['loss'])
    assert math.isfinite(metrics['gradient_norm'])

    assert metrics['valid_tokens'] == 7
    assert 0 <= metrics['correct_tokens'] <= 7

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )