import torch


def greedy_generate(
    model,
    prompt_ids,
    eos_id,
    max_new_tokens,
):
    was_training = model.training

    device = next(
        model.parameters()
    ).device

    generated = prompt_ids.to(device).clone()

    batch_size = generated.size(0)

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    try:
        model.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(generated)

                last_logits = logits[:, -1, :]

                next_token = last_logits.argmax(
                    dim=-1,
                    keepdim=True,
                )

                # 已经生成 EOS 的样本继续保持为 EOS。
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(
                        next_token,
                        eos_id,
                    ),
                    next_token,
                )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1,
                )

                finished = (
                    finished
                    | next_token.squeeze(1).eq(eos_id)
                )

                if finished.all():
                    break

    finally:
        model.train(was_training)

    return generated


def greedy_generate_with_cache(
    model,
    prompt_ids,
    eos_id,
    max_new_tokens,
):
    was_training = model.training

    device = next(
        model.parameters()
    ).device

    generated = prompt_ids.to(device).clone()

    batch_size = generated.size(0)

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

    try:
        model.eval()

        with torch.no_grad():
            # Prefill：一次处理完整 prompt。
            logits, past_key_values = (
                model.forward_with_cache(
                    generated,
                    past_key_values=None,
                )
            )

            for step in range(max_new_tokens):
                # 当前 logits 的最后位置预测下一个 token。
                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(
                        next_token,
                        eos_id,
                    ),
                    next_token,
                )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1,
                )

                finished = (
                    finished
                    | next_token.squeeze(1).eq(eos_id)
                )

                if finished.all():
                    break

                # 最后一次循环后不再需要预测下一个 token。
                if step == max_new_tokens - 1:
                    break

                # Decode：只输入刚生成的一个 token。
                logits, past_key_values = (
                    model.forward_with_cache(
                        next_token,
                        past_key_values=past_key_values,
                    )
                )

    finally:
        model.train(was_training)

    return generated