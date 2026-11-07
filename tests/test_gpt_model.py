import torch


def make_config():
    from gpt.model import GPTConfig

    return GPTConfig(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=16,
        dropout=0.0,
        pad_id=0,
    )


def test_gpt_model_returns_logits_and_backpropagates():
    from gpt.model import GPTModel

    model = GPTModel(make_config())

    input_ids = torch.tensor([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 0, 0],
    ])

    logits = model(input_ids)

    assert logits.shape == (2, 5, 20)
    assert torch.isfinite(logits).all()

    logits.square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gpt_model_prevents_future_information_leakage():
    from gpt.model import GPTModel

    torch.manual_seed(0)

    model = GPTModel(make_config())
    model.eval()

    input_ids = torch.tensor([
        [1, 2, 3, 4],
    ])

    changed_input_ids = torch.tensor([
        [1, 2, 3, 9],
    ])

    logits = model(input_ids)
    changed_logits = model(changed_input_ids)

    torch.testing.assert_close(
        logits[:, :-1],
        changed_logits[:, :-1],
        atol=1e-6,
        rtol=1e-5,
    )

    assert not torch.allclose(
        logits[:, -1],
        changed_logits[:, -1],
    )