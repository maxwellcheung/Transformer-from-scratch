import torch

from seq2seq.masks import make_causal_mask


def test_gpt_block_preserves_shape_and_backpropagates():
    from gpt.layers import GPTBlock

    block = GPTBlock(
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

    mask = make_causal_mask(
        length=4,
        device=x.device,
    )

    output = block(x, mask)

    assert output.shape == x.shape

    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gpt_block_prevents_future_information_leakage():
    from gpt.layers import GPTBlock

    torch.manual_seed(0)

    block = GPTBlock(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    block.eval()

    x = torch.randn(1, 4, 8)

    changed_x = x.clone()

    # 只修改最后一个 token 的一个特征。
    changed_x[0, -1, 0] += 100

    mask = make_causal_mask(
        length=4,
        device=x.device,
    )

    output = block(x, mask)
    changed_output = block(
        changed_x,
        mask,
    )

    # 未来 token 的变化不能影响过去位置。
    torch.testing.assert_close(
        output[:, :-1],
        changed_output[:, :-1],
        atol=1e-6,
        rtol=1e-5,
    )

    # 最后位置本身应当发生变化。
    assert not torch.allclose(
        output[:, -1],
        changed_output[:, -1],
    )