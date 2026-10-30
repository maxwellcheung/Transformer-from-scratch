import pytest
import torch
import torch.nn.functional as F
import math

def make_example():
    logits = torch.tensor([
        [
            [0.1, 3.0, 0.2, 0.0],
            [0.1, 0.2, 3.0, 0.0],
            [3.0, 0.1, 0.2, 0.0],
        ],
        [
            [0.1, 3.0, 0.2, 0.0],
            [0.1, 0.2, 0.3, 3.0],
            [0.1, 3.0, 0.2, 0.0],
        ],
    ])

    targets = torch.tensor([
        [1, 2, 0],
        [2, 3, 0],
    ])

    return logits, targets


def test_loss_and_accuracy_ignore_padding():
    from seq2seq.training import (
        token_cross_entropy,
        token_accuracy,
    )

    logits, targets = make_example()

    loss = token_cross_entropy(
        logits,
        targets,
        pad_id=0,
    )

    expected_loss = F.cross_entropy(
        logits.reshape(-1, 4),
        targets.reshape(-1),
        ignore_index=0,
    )

    torch.testing.assert_close(
        loss,
        expected_loss,
    )

    correct_tokens, valid_tokens = token_accuracy(
        logits,
        targets,
        pad_id=0,
    )

    assert correct_tokens == 3
    assert valid_tokens == 4


def test_loss_and_accuracy_reject_all_padding_targets():
    from seq2seq.training import (
        token_cross_entropy,
        token_accuracy,
    )

    logits = torch.randn(2, 3, 4)

    targets = torch.zeros(
        2,
        3,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match='valid token',
    ):
        token_cross_entropy(
            logits,
            targets,
            pad_id=0,
        )

    with pytest.raises(
        ValueError,
        match='valid token',
    ):
        token_accuracy(
            logits,
            targets,
            pad_id=0,
        )


def test_train_step_updates_model_and_returns_metrics():
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )
    from seq2seq.training import train_step

    torch.manual_seed(0)

    config = TransformerConfig(
        src_vocab_size=20,
        tgt_vocab_size=19,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        dropout=0.0,
        src_pad_id=0,
        tgt_pad_id=0,
        max_len=16,
    )

    model = Seq2SeqTransformer(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    batch = (
        torch.tensor([
            [4, 5, 0],
            [6, 7, 8],
        ]),
        torch.tensor([
            [1, 4, 5],
            [1, 6, 0],
        ]),
        torch.tensor([
            [4, 5, 2],
            [6, 2, 0],
        ]),
    )

    weight_before = (
        model.output_projection.weight
        .detach()
        .clone()
    )

    metrics = train_step(
        model,
        batch,
        optimizer,
        pad_id=0,
        max_grad_norm=1.0,
    )

    weight_after = (
        model.output_projection.weight
        .detach()
    )

    assert set(metrics) == {
        'loss',
        'correct_tokens',
        'valid_tokens',
        'gradient_norm',
    }

    assert math.isfinite(metrics['loss'])
    assert metrics['loss'] > 0

    assert metrics['valid_tokens'] == 5

    assert (
        0
        <= metrics['correct_tokens']
        <= metrics['valid_tokens']
    )

    assert math.isfinite(
        metrics['gradient_norm']
    )
    assert metrics['gradient_norm'] >= 0

    assert not torch.allclose(
        weight_before,
        weight_after,
    )


def test_model_can_overfit_single_batch():
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )
    from seq2seq.training import (
        token_cross_entropy,
        train_step,
    )

    torch.manual_seed(0)

    config = TransformerConfig(
        src_vocab_size=10,
        tgt_vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        dropout=0.0,
        src_pad_id=0,
        tgt_pad_id=0,
        max_len=16,
    )

    model = Seq2SeqTransformer(config)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.02,
    )

    src_tokens = torch.tensor([
        [4, 5, 6],
    ])

    tgt_input = torch.tensor([
        [1, 4, 5, 6],
    ])

    tgt_output = torch.tensor([
        [4, 5, 6, 2],
    ])

    batch = (
        src_tokens,
        tgt_input,
        tgt_output,
    )

    model.eval()

    with torch.no_grad():
        initial_logits = model(
            src_tokens,
            tgt_input,
        )

        initial_loss = token_cross_entropy(
            initial_logits,
            tgt_output,
            pad_id=0,
        ).item()

    metrics = None

    for _ in range(100):
        metrics = train_step(
            model,
            batch,
            optimizer,
            pad_id=0,
            max_grad_norm=1.0,
        )

    model.eval()

    with torch.no_grad():
        final_logits = model(
            src_tokens,
            tgt_input,
        )

        final_loss = token_cross_entropy(
            final_logits,
            tgt_output,
            pad_id=0,
        ).item()

    assert final_loss < initial_loss * 0.2
    assert metrics['correct_tokens'] == 4
    assert metrics['valid_tokens'] == 4