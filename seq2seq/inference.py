import torch

def greedy_decode(
    model,
    src_tokens,
    sos_id,
    eos_id,
    max_new_tokens,
):
    if max_new_tokens <= 0:
        raise ValueError(
            'max_new_tokens must be positive, '
            f'got {max_new_tokens}'
        )

    if src_tokens.dim() != 2:
        raise ValueError(
            'src_tokens must have 2 dimensions, '
            f'got {src_tokens.dim()}'
        )

    was_training = model.training

    model.eval()

    try:
        with torch.no_grad():
            device = next(
                model.parameters()
            ).device

            src_tokens = src_tokens.to(device)

            # Encoder 只执行一次。
            memory, src_mask = model.encode(
                src_tokens
            )

            batch_size = src_tokens.size(0)

            generated = torch.full(
                (batch_size, 1),
                sos_id,
                dtype=torch.long,
                device=device,
            )

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=device,
            )

            for _ in range(max_new_tokens):
                decoder_hidden = model.decode(
                    generated,
                    memory,
                    src_mask,
                )

                # 只使用序列最后位置预测下一个 token。
                last_hidden = decoder_hidden[:, -1, :]

                next_logits = model.output_projection(
                    last_hidden
                )

                next_token = next_logits.argmax(
                    dim=-1
                )

                # 已经结束的样本后续只能继续填 EOS。
                next_token = torch.where(
                    finished,
                    torch.full_like(
                        next_token,
                        eos_id,
                    ),
                    next_token,
                )

                generated = torch.cat(
                    [
                        generated,
                        next_token.unsqueeze(1),
                    ],
                    dim=1,
                )

                finished = (
                    finished
                    | next_token.eq(eos_id)
                )

                if finished.all():
                    break

            return generated

    finally:
        # 无论正常返回还是发生异常，都恢复原模式。
        model.train(was_training)