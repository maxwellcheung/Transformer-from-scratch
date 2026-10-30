import math
import torch
from torch import nn

def scaled_dot_product_attention(q,k,v,mask=None,dropout=None,):
    for name, tensor in (('q', q),('k',k),('v',v),):
        if tensor.dim() != 4:
            raise ValueError(
                f'{name} must have 4 dimensions, '
                f'got {tensor.dim()}'
            )

    if (q.size(0) != k.size(0) or q.size(0) != v.size(0)
        or q.size(1) != k.size(1) or q.size(1) != v.size(1)):
        raise ValueError(
            'q, k and v must have the same '
            'batch size and number of heads'
        )

    if q.size(-1) != k.size(-1):
        raise ValueError(
            'q and k must have the same head dimension, '
            f'got {q.size(-1)} and {k.size(-1)}'
        )

    if k.size(-2) != v.size(-2):
        raise ValueError(
            'key/value length must match, '
            f'got {k.size(-2)} and {v.size(-2)}'
        )

    batch_size = q.size(0)
    num_heads = q.size(1)
    query_length = q.size(-2)
    key_length = k.size(-2)
    head_dimension = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2,-1),) / math.sqrt(head_dimension)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError(
                f'mask must have bool dtype, got {mask.dtype}'
            )

        allowed_shapes = {
            (batch_size, 1, 1, key_length),
            (1, 1, query_length, key_length),
            (batch_size, 1, query_length, key_length),
        }

        if tuple(mask.shape) not in allowed_shapes:
            raise ValueError(
                f'mask shape must be one of {allowed_shapes}, '
                f'got {tuple(mask.shape)}'
            )

        expanded_mask = mask.expand(
            batch_size,
            num_heads,
            query_length,
            key_length,
        )

        all_masked = expanded_mask.all(
            dim=-1,
            keepdim=True,
        )

        scores = scores.masked_fill(
            expanded_mask,
            float('-inf'),
        )

        # 全屏蔽行不能让 softmax 接收全 -inf。
        scores = torch.where(
            all_masked,
            torch.zeros_like(scores),
            scores,
        )

    weights = torch.softmax(scores, dim=-1)

    if mask is not None:
        weights = weights.masked_fill(
        expanded_mask,
        0.0,
        )

    context_weights = weights

    if dropout is not None:
        context_weights = dropout(context_weights)

    context = torch.matmul(context_weights, v)

    return context, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f'd_model must be positive, got {d_model}'
            )

        if num_heads <= 0:
            raise ValueError(
                f'num_heads must be positive, got {num_heads}'
            )

        if d_model % num_heads != 0:
            raise ValueError(
                f'd_model {d_model} must be divisible '
                f'by num_heads {num_heads}'
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )

        return x.transpose(1,2)

    def _combine_heads(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(2)

        x = x.transpose(1, 2).contiguous()
        return x.reshape(
            batch_size,
            sequence_length,
            self.d_model,
        )
    def _validate_input(self, name, x):
        if x.dim() != 3:
            raise ValueError(
                f'{name} must have 3 dim,'
                f'got {x.dim()}'
            )

        actual_width = x.size(-1)

        if actual_width != self.d_model:
            raise ValueError(
                f'{name} expected model width '
                f'{self.d_model}, got {actual_width}'
            )

    def forward(self, query, key, value, mask=None,):
        self._validate_input('query', query)
        self._validate_input('key', key)
        self._validate_input('value', value)

        if (
            query.size(0) != key.size(0)
            or query.size(0) != value.size(0)
        ):
            raise ValueError(
                'query, key and value must have '
                'the same batch size'
            )

        if key.size(1) != value.size(1):
            raise ValueError(
                'key/value length must match, '
                f'got {key.size(1)} and {value.size(1)}'
            )

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        context, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=self.attention_dropout,
        )

        context = self._combine_heads(context)
        output = self.out_proj(context)

        return output, weights