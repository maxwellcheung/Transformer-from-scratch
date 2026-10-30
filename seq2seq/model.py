import math

from torch import nn

from .layers import EncoderLayer, DecoderLayer
from .masks import make_padding_mask, make_causal_mask
from .position import PositionalEncoding


class TransformerConfig:
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout,
        src_pad_id,
        tgt_pad_id,
        max_len,
    ):
        positive_values = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'num_layers': num_layers,
            'max_len': max_len,
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(
                    f'{name} must be positive, '
                    f'got {value}'
                )

        if d_model % num_heads != 0:
            raise ValueError(
                f'd_model {d_model} must be divisible '
                f'by num_heads {num_heads}'
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f'dropout must be in [0, 1), '
                f'got {dropout}'
            )

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.max_len = max_len

class  Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_scale = math.sqrt(config.d_model)

        # 源端和目标端有不同的词表，因此使用两个 Embedding。
        self.src_embedding = nn.Embedding(
            config.src_vocab_size,
            config.d_model,
            padding_idx=config.src_pad_id,
        )

        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size,
            config.d_model,
            padding_idx=config.tgt_pad_id,
        )

        self.src_position = PositionalEncoding(
            config.d_model,
            config.max_len,
            config.dropout,
        )

        self.tgt_position = PositionalEncoding(
            config.d_model,
            config.max_len,
            config.dropout,
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                config.d_model, config.num_heads, config.d_ff, config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                config.d_model, config.num_heads, config.d_ff, config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.output_projection = nn.Linear(
            config.d_model,
            config.tgt_vocab_size,
        )

    def encode(self, src_tokens):
        src_mask = make_padding_mask(src_tokens, self.config.src_pad_id,)

        x = self.src_embedding(src_tokens)

        # Transformer 论文中的 embedding 缩放。
        x = x * self.embedding_scale
        x = self.src_position(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        memory = x

        return memory, src_mask

    def decode(self, tgt_tokens, memory, src_mask,):
        tgt_padding_mask = make_padding_mask(
            tgt_tokens,
            self.config.tgt_pad_id,
        )

        causal_mask = make_causal_mask(
            tgt_tokens.size(1),
            tgt_tokens.device,
        )

        tgt_mask = tgt_padding_mask | causal_mask

        y = self.tgt_embedding(tgt_tokens)
        y = y * self.embedding_scale
        y = self.tgt_position(y)

        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, src_mask,)

        return y

    def forward(self, src_tokens, tgt_tokens):
        memory, src_mask = self.encode(src_tokens)

        decoder_hidden = self.decode(
            tgt_tokens,
            memory,
            src_mask,
        )

        logits = self.output_projection(
            decoder_hidden
        )

        return logits
