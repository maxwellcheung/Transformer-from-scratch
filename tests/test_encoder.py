import torch


def test_feed_forward_preserves_shape_and_processes_positions_independently():
    from seq2seq.layers import FeedForward

    torch.manual_seed(0)

    feed_forward = FeedForward(
        d_model=8,
        d_ff=16,
        dropout=0.0,
    )
    feed_forward.eval()

    x = torch.randn(1, 3, 8)
    changed_x = x.clone()
    changed_x[0, 1] += 10

    output = feed_forward(x)
    changed_output = feed_forward(changed_x)

    assert output.shape == x.shape

    torch.testing.assert_close(
        output[:, 0],
        changed_output[:, 0],
    )
    torch.testing.assert_close(
        output[:, 2],
        changed_output[:, 2],
    )


def test_encoder_layer_preserves_shape_and_backpropagates():
    from seq2seq.layers import EncoderLayer

    layer = EncoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )

    x = torch.randn(
        2,
        4,
        8,
        requires_grad=True,
    )
    source_mask = torch.zeros(
        2,
        1,
        1,
        4,
        dtype=torch.bool,
    )

    output = layer(x, source_mask)

    assert output.shape == x.shape

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_encoder_layer_ignores_masked_padding_token():
    from seq2seq.layers import EncoderLayer

    torch.manual_seed(0)

    layer = EncoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    layer.eval()

    valid_tokens = torch.randn(1, 3, 8)
    padding_token = torch.randn(1, 1, 8)

    padded_input = torch.cat(
        [valid_tokens, padding_token],
        dim=1,
    )

    valid_mask = torch.zeros(
        1,
        1,
        1,
        3,
        dtype=torch.bool,
    )

    padded_mask = torch.tensor(
        [[[[False, False, False, True]]]]
    )

    valid_output = layer(
        valid_tokens,
        valid_mask,
    )
    padded_output = layer(
        padded_input,
        padded_mask,
    )

    torch.testing.assert_close(
        valid_output,
        padded_output[:, :3],
        atol=1e-6,
        rtol=1e-5,
    )