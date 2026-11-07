import torch
from torch import nn

from seq2seq.masks import (
    make_causal_mask,
    make_padding_mask,
)
from .layers import GPTBlock


class GPTConfig:
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len,
        dropout,
        pad_id,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout
        self.pad_id = pad_id

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )

        self.position_embedding = nn.Embedding(
            config.max_len,
            config.d_model,
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList([
            GPTBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
        )

    def forward(self, input_ids):
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)

        if sequence_length > self.config.max_len:
            raise ValueError(
                f'sequence length {sequence_length} exceeds '
                f'max_len {self.config.max_len}'
            )

        # 1：产生 [T] 的位置编号
        position_ids = torch.arange(sequence_length, device=input_ids.device)

        # 2：得到 token 和 position embedding
        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(position_ids)

        # 3：两种 embedding 相加，再经过 dropout
        x = self.embedding_dropout(token_vectors + position_vectors)

        # 4：构造 padding mask 和 causal mask
        padding_mask = make_padding_mask(
            input_ids,
            self.config.pad_id,
        )

        causal_mask = make_causal_mask(
            sequence_length,
            input_ids.device,
        )

        # 5：使用逻辑或合并 Mask
        attention_mask = causal_mask | padding_mask

        # 6：依次经过全部 GPT Block
        for block in self.blocks:
            x = block(x, attention_mask)

        # 7：最终归一化
        hidden = self.final_norm(x)

        # TODO 8：映射到词表，得到 [B,T,V]
        logits = self.lm_head(hidden)

        return logits 

    
    def forward_with_cache(
        self,
        input_ids,
        past_key_values=None,
    ):
        sequence_length = input_ids.size(1)

        if past_key_values is None:
            # Prefill 阶段：没有历史 Cache。
            past_length = 0

            layer_past_key_values = [
                None
                for _ in self.blocks
            ]

            # Prefill 同时处理多个 token，仍需因果 Mask。
            attention_mask = make_causal_mask(
                sequence_length,
                input_ids.device,
            )

        else:
            # 每个 GPT Block 必须对应一个 Cache。
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    'number of layer caches must match '
                    f'number of blocks: {len(self.blocks)}'
                )

            # 所有层的历史长度应当相同。
            first_past_key, _ = past_key_values[0]
            past_length = first_past_key.size(2)

            layer_past_key_values = past_key_values

            # 当前版本的增量阶段每次只接收一个新 token。
            if sequence_length != 1:
                raise ValueError(
                    'cached decoding expects exactly '
                    'one new token'
                )

            # 历史中只有过去和当前位置，没有未来位置。
            attention_mask = None

        total_length = past_length + sequence_length

        if total_length > self.config.max_len:
            raise ValueError(
                f'total sequence length {total_length} '
                f'exceeds max_len {self.config.max_len}'
            )

        # 位置必须从 past_length 开始。
        position_ids = torch.arange(
            past_length,
            total_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(
            input_ids
        )

        position_vectors = self.position_embedding(
            position_ids
        )

        x = token_vectors + position_vectors
        x = self.embedding_dropout(x)

        present_key_values = []

        for block, layer_past in zip(
            self.blocks,
            layer_past_key_values,
        ):
            x, layer_present = block.forward_with_cache(
                x,
                past_key_value=layer_past,
                attention_mask=attention_mask,
            )

            present_key_values.append(
                layer_present
            )

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        return logits, present_key_values