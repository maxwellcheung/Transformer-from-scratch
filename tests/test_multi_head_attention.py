import pytest
import torch

def test_multi_head_self_attention_shapes_and_gradients():
    from seq2seq.attention import MultiHeadAttention

    module = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        dropout=0.0,
    )

    x = torch.randn(2, 3, 8, requires_grad=True)

    output, weights = module(x, x, x)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)

    output.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_multi_head_cross_attention_shapes_and_source_mask():
    from seq2seq.attention import MultiHeadAttention

    module = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        dropout=0.0,
    )

    decoder_state = torch.randn(2, 2, 8)
    encoder_output = torch.randn(2, 4, 8)

    source_mask = torch.zeros(
        2, 1, 1, 4,
        dtype=torch.bool,
    )

    source_mask[1, :, :, 3] = True

    output, weights = module(
        decoder_state,
        encoder_output,
        encoder_output,
        source_mask,
    )

    assert output.shape == (2, 2, 8)
    assert weights.shape == (2, 2, 2, 4)

    torch.testing.assert_close(
        weights[1, :, :, 3],
        torch.zeros(2, 2),
    )

def test_multi_head_rejects_non_divisible_width():
    from seq2seq.attention import MultiHeadAttention

    with pytest.raises(ValueError, match='7.*2'):
        MultiHeadAttention(
            d_model=7,
            num_heads=2,
            dropout=0.0,
        )

def test_multi_head_rejects_wrong_input_width():
    from seq2seq.attention import MultiHeadAttention

    module = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        dropout=0.0,
    )

    query = torch.randn(2, 3, 7)
    key = value = torch.randn(2, 3, 8)

    with pytest.raises(
        ValueError,
        match='query.*expected.*8.*got.*7',
    ):
        module(query, key, value)

def test_multi_head_rejects_invalid_mask_shape():
    from seq2seq.attention import MultiHeadAttention

    module = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        dropout=0.0,
    )

    x = torch.randn(2, 3, 8)
    wrong_mask = torch.zeros(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match='mask shape'):
        module(x, x, x, wrong_mask)