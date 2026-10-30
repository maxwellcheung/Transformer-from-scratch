import pytest
import torch

def test_padding_mask_marks_pad_tokens():
    from seq2seq.masks import make_padding_mask

    tokens = torch.tensor([
        [4, 0, 5],
        [0, 0, 6],
    ])

    mask = make_padding_mask(tokens, pad_id=0)

    expected = torch.tensor([
        [[[False, True, False]]],
        [[[True, True, False]]],
    ])

    assert mask.shape == (2, 1, 1, 3)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_causal_mask_blocks_future_positions():
    from seq2seq.masks import make_causal_mask

    mask = make_causal_mask(
        length=3,
        device=torch.device('cpu'),
    )

    expected = torch.tensor([
        [[
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]]
    ])

    assert mask.shape == (1, 1, 3, 3)
    assert torch.equal(mask, expected)

def test_attetion_weights_sum_to_one_and_mask_block_future():
    from seq2seq.attention import scaled_dot_product_attention
    from seq2seq.masks import make_causal_mask

    q = k = v = torch.tensor([
        [[[1.0, 0.0],
        [0.0, 1.0]]]
    ])

    mask = make_causal_mask(2, q.device)

    context, weights = scaled_dot_product_attention(
        q, k, v, mask,
    )

    assert context.shape == q.shape

    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(1, 1, 2),
    )

    assert weights[0, 0, 0, 1].item() == 0.0


def test_attention_returns_zero_for_fully_masked_query():
    from seq2seq.attention import scaled_dot_product_attention

    q = k = v = torch.tensor([
        [[[1.0, 0.0],
        [0.0, 1.0]]]
    ])

    mask = torch.tensor([
        [[
            [True, True],
            [False, False],
        ]]
    ])

    context, weights = scaled_dot_product_attention(
        q, k, v, mask,
    )

    assert torch.isfinite(weights).all()
    assert torch.isfinite(context).all()

    torch.testing.assert_close(
        weights[0, 0, 0],
        torch.zeros(2),
    )

    torch.testing.assert_close(
        context[0, 0, 0],
        torch.zeros(2),
    )

def test_attention_rejects_different_qk_head_dimensions():
    from seq2seq.attention import scaled_dot_product_attention

    q = torch.zeros(1, 1, 2, 3)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match='head dimension'):
        scaled_dot_product_attention(q, k, v)

def test_attention_rejects_different_key_value_lengths():
    from seq2seq.attention import scaled_dot_product_attention

    q = torch.zeros(1, 1, 2, 2)
    k = torch.zeros(1, 1, 3, 2)
    v = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match='key/value length'):
        scaled_dot_product_attention(q, k, v)

def test_attention_rejects_invalid_mask():
    from seq2seq.attention import scaled_dot_product_attention

    q = k = v = torch.zeros(1, 1, 2, 2)

    wrong_shape = torch.zeros(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match='mask shape '):
        scaled_dot_product_attention(q, k, v, wrong_shape,)

    wrong_dtype = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match='bool'):
        scaled_dot_product_attention(q, k, v, wrong_dtype,)