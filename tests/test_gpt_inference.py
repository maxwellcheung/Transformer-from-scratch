import torch
from torch import nn


class ScriptedGPT(nn.Module):
    def __init__(self):
        super().__init__()

        # 用于提供模型所在设备。
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = 6

        # 输入长度 1、2、3 时依次预测这些 token。
        self.next_tokens = [4, 5, 2]

    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )

        step = min(
            sequence_length - 1,
            len(self.next_tokens) - 1,
        )

        token_id = self.next_tokens[step]

        logits[:, -1, token_id] = 10.0

        return logits


def test_greedy_generate_stops_at_eos_and_restores_mode():
    from gpt.inference import greedy_generate

    model = ScriptedGPT()
    model.train()

    prompt_ids = torch.tensor([
        [1],
    ])

    generated = greedy_generate(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=2,
        max_new_tokens=10,
    )

    assert generated.tolist() == [
        [1, 4, 5, 2],
    ]

    # 调用前是训练模式，生成后仍应恢复为训练模式。
    assert model.training