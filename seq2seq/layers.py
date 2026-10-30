import torch
from torch import nn

from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        attention_output, attention_weights = self.self_attention(
            x, x, x, mask=src_mask)

        attention_output = self.attention_dropout(attention_output)

        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)

        output = self.norm2(x + feed_forward_output)
        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, memory, tgt_mask, src_mask,):
        self_attention_output, _ = self.self_attention(y, y, y, mask=tgt_mask)
        self_attention_output = self.self_attention_dropout(self_attention_output)

        y = self.norm1(y + self_attention_output)

        cross_attention_output, _ = self.cross_attention(y, memory, memory, mask=src_mask)
        cross_attention_output = self.cross_attention_dropout(cross_attention_output)

        y = self.norm2( y + cross_attention_output)

        feed_forward_output = self.feed_forward(y)
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)

        output = self.norm3(y + feed_forward_output)
        return output