import torch
import pytest

def test_full_model_shape_and_finite_backward():
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )

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

    src_tokens = torch.tensor([
        [4, 5, 0],
        [6, 7, 8],
    ])

    tgt_tokens = torch.tensor([
        [1, 4, 5],
        [1, 6, 0],
    ])

    logits = model(
        src_tokens,
        tgt_tokens,
    )

    assert logits.shape == (2, 3, 19)
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def make_test_model():
    from seq2seq.model import (
        Seq2SeqTransformer,
        TransformerConfig,
    )

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
    model.eval()

    return model

def test_full_model_prevents_future_target_leakage():
    torch.manual_seed(0)

    model = make_test_model()

    src_tokens = torch.tensor([
        [4, 5, 6, 0],
    ])

    tgt_tokens = torch.tensor([
        [1, 4, 5, 6],
    ])

    changed_tgt_tokens = tgt_tokens.clone()

    # 只改变最后一个目标 token
    changed_tgt_tokens[0, -1] = 7

    with torch.no_grad():
        logits = model(
            src_tokens,
            tgt_tokens,
        )

        changed_logits = model(
            src_tokens,
            changed_tgt_tokens,
        )

    # 最后一个 token 不能影响前面位置的预测
    torch.testing.assert_close(
        logits[:, :-1],
        changed_logits[:, :-1],
        atol=1e-6,
        rtol=1e-5,
    )

    # 最后一个位置本身应该受到影响
    assert not torch.allclose(
        logits[:, -1],
        changed_logits[:, -1],
    )


def test_full_model_ignores_masked_source_padding():
    torch.manual_seed(0)

    model = make_test_model()

    src_tokens = torch.tensor([
        [4, 5, 6],
    ])

    padded_src_tokens = torch.tensor([
        [4, 5, 6, 0, 0],
    ])

    tgt_tokens = torch.tensor([
        [1, 4, 5, 6],
    ])

    with torch.no_grad():
        logits = model(
            src_tokens,
            tgt_tokens,
        )

        padded_logits = model(
            padded_src_tokens,
            tgt_tokens,
        )

    torch.testing.assert_close(
        logits,
        padded_logits,
        atol=1e-6,
        rtol=1e-5,
    )

def make_config(**overrides):
    from seq2seq.model import TransformerConfig

    values = {
        'src_vocab_size': 20,
        'tgt_vocab_size': 19,
        'd_model': 8,
        'num_heads': 2,
        'd_ff': 16,
        'num_layers': 2,
        'dropout': 0.0,
        'src_pad_id': 0,
        'tgt_pad_id': 0,
        'max_len': 16,
    }

    values.update(overrides)

    return TransformerConfig(**values)

@pytest.mark.parametrize(
    'field,value',
    [
        ('src_vocab_size', 0),
        ('tgt_vocab_size', 0),
        ('d_model', 0),
        ('num_heads', 0),
        ('d_ff', 0),
        ('num_layers', 0),
        ('max_len', 0),
    ],
)
def test_config_rejects_non_positive_values(
    field,
    value,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        make_config(**{field: value})


def test_config_rejects_non_divisible_model_width():
    with pytest.raises(
        ValueError,
        match='d_model.*7.*num_heads.*2',
    ):
        make_config(
            d_model=7,
            num_heads=2,
        )


@pytest.mark.parametrize(
    'dropout',
    [-0.1, 1.0, 1.1],
)
def test_config_rejects_invalid_dropout(dropout):
    with pytest.raises(
        ValueError,
        match='dropout',
    ):
        make_config(dropout=dropout)