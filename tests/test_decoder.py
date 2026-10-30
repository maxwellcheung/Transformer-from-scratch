import torch


def test_decoder_layer_accepts_different_source_and_target_lengths():
    from seq2seq.layers import DecoderLayer

    layer = DecoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )

    target_state = torch.randn(
        2,
        3,
        8,
        requires_grad=True,
    )

    memory = torch.randn(
        2,
        5,
        8,
        requires_grad=True,
    )

    target_mask = torch.triu(
        torch.ones(
            1,
            1,
            3,
            3,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    source_mask = torch.zeros(
        2,
        1,
        1,
        5,
        dtype=torch.bool,
    )

    output = layer(
        target_state,
        memory,
        target_mask,
        source_mask,
    )

    assert output.shape == (2, 3, 8)

    output.square().mean().backward()

    assert target_state.grad is not None
    assert memory.grad is not None
    assert torch.isfinite(target_state.grad).all()
    assert torch.isfinite(memory.grad).all()

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

def test_decoder_layer_prevents_future_information_leakage():
    from seq2seq.layers import DecoderLayer

    torch.manual_seed(0)

    layer = DecoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    layer.eval()

    target_state = torch.randn(1, 4, 8)
    changed_target_state = target_state.clone()

    # 只修改最后一个未来 token
    changed_target_state[:, -1] += 100

    memory = torch.randn(1, 5, 8)

    target_mask = torch.triu(
        torch.ones(
            1,
            1,
            4,
            4,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    source_mask = torch.zeros(
        1,
        1,
        1,
        5,
        dtype=torch.bool,
    )

    output = layer(
        target_state,
        memory,
        target_mask,
        source_mask,
    )

    changed_output = layer(
        changed_target_state,
        memory,
        target_mask,
        source_mask,
    )

    # 最后一个 token 的变化不能影响前面三个位置
    torch.testing.assert_close(
        output[:, :-1],
        changed_output[:, :-1],
    )

    # 最后一个位置本身应当发生变化
    assert not torch.allclose(
        output[:, -1],
        changed_output[:, -1],
    )