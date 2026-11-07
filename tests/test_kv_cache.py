import torch

from seq2seq.attention import MultiHeadAttention
from seq2seq.masks import make_causal_mask


def test_cached_attention_matches_full_last_position():
    torch.manual_seed(0)

    attention = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        dropout=0.0,
    )
    attention.eval()

    x = torch.randn(1, 4, 8)

    # 无 Cache：一次计算全部 4 个 token。
    full_mask = make_causal_mask(
        length=4,
        device=x.device,
    )

    full_output, _ = attention(
        x,
        x,
        x,
        mask=full_mask,
    )

    # Prefill：先处理前 3 个 token，并得到 Cache。
    prefill_x = x[:, :3]

    prefill_mask = make_causal_mask(
        length=3,
        device=x.device,
    )

    _, _, past_key_value = attention.forward_with_cache(
        prefill_x,
        past_key_value=None,
        mask=prefill_mask,
    )

    # Decode：只处理第 4 个 token。
    new_x = x[:, 3:]

    cached_output, _, present_key_value = (
        attention.forward_with_cache(
            new_x,
            past_key_value=past_key_value,
            mask=None,
        )
    )

    # 增量计算结果应与完整计算最后一个位置一致。
    torch.testing.assert_close(
        cached_output,
        full_output[:, -1:],
        atol=1e-6,
        rtol=1e-5,
    )

    present_key, present_value = present_key_value

    assert present_key.shape == (1, 2, 4, 4)
    assert present_value.shape == (1, 2, 4, 4)



from gpt.layers import GPTBlock


def test_cached_gpt_block_matches_full_last_position():
    torch.manual_seed(0)

    block = GPTBlock(
        d_model=8,
        num_heads=2,
        d_ff=16,
        dropout=0.0,
    )
    block.eval()

    x = torch.randn(1, 4, 8)

    # 无 Cache 的完整计算。
    full_mask = make_causal_mask(
        length=4,
        device=x.device,
    )

    full_output = block(
        x,
        full_mask,
    )

    # Prefill 前三个位置。
    prefill_mask = make_causal_mask(
        length=3,
        device=x.device,
    )

    _, past_key_value = block.forward_with_cache(
        x[:, :3],
        past_key_value=None,
        attention_mask=prefill_mask,
    )

    # 只计算最后一个位置。
    cached_output, present_key_value = (
        block.forward_with_cache(
            x[:, 3:],
            past_key_value=past_key_value,
            attention_mask=None,
        )
    )

    torch.testing.assert_close(
        cached_output,
        full_output[:, -1:],
        atol=1e-6,
        rtol=1e-5,
    )

    present_key, present_value = present_key_value

    assert present_key.shape == (1, 2, 4, 4)
    assert present_value.shape == (1, 2, 4, 4)


from gpt.model import GPTConfig, GPTModel


def test_cached_gpt_model_matches_full_last_position():
    torch.manual_seed(0)

    config = GPTConfig(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=16,
        dropout=0.0,
        pad_id=0,
    )

    model = GPTModel(config)
    model.eval()

    input_ids = torch.tensor([
        [1, 4, 5, 6],
    ])

    # 无 Cache：完整计算全部 token。
    full_logits = model(input_ids)

    # Prefill：处理前三个 token，并获得每层 Cache。
    _, past_key_values = model.forward_with_cache(
        input_ids[:, :3],
        past_key_values=None,
    )

    # Decode：只输入最后一个 token。
    cached_logits, present_key_values = (
        model.forward_with_cache(
            input_ids[:, 3:],
            past_key_values=past_key_values,
        )
    )

    torch.testing.assert_close(
        cached_logits,
        full_logits[:, -1:],
        atol=1e-6,
        rtol=1e-5,
    )

    assert len(present_key_values) == 2

    for key, value in present_key_values:
        assert key.shape == (1, 2, 4, 4)
        assert value.shape == (1, 2, 4, 4)


from gpt.inference import (
    greedy_generate,
    greedy_generate_with_cache,
)


def test_cached_generation_matches_uncached_generation():
    torch.manual_seed(0)

    config = GPTConfig(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=16,
        dropout=0.0,
        pad_id=0,
    )

    model = GPTModel(config)
    model.train()

    prompt_ids = torch.tensor([
        [1, 4, 5],
    ])

    uncached_result = greedy_generate(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=2,
        max_new_tokens=4,
    )

    cached_result = greedy_generate_with_cache(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=2,
        max_new_tokens=4,
    )

    assert torch.equal(
        cached_result,
        uncached_result,
    )

    # 生成结束后应恢复原来的训练模式。
    assert model.training