from torch import nn

from seq2seq.attention import MultiHeadAttention
from seq2seq.layers import FeedForward


class GPTBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.0,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout=dropout,
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        #1：先对 x 做 norm1
        normalized_x = self.norm1(x)
        #2：归一化结果同时作为 Q、K、V
        attention_output, _ = self.self_attention(normalized_x,normalized_x,normalized_x,mask=attention_mask,)
        #3：attention dropout，然后与原来的 x 残差相加
        x = x + self.attention_dropout(attention_output)
        #4：对更新后的 x 做 norm2
        normalized_x = self.norm2(x)
        #5：送入前馈网络，再做 dropout
        feed_forward_output = self.feed_forward(normalized_x)
        #6：与 x 残差相加并返回
        output = x + self.feed_forward_dropout(feed_forward_output)
        return output

    def forward_with_cache(
        self,
        x,
        past_key_value=None,
        attention_mask=None,
    ):
        normalized_x = self.norm1(x)

        (
            attention_output,
            _,
            present_key_value,
        ) = self.self_attention.forward_with_cache(
            normalized_x,
            past_key_value=past_key_value,
            mask=attention_mask,
        )

        attention_output = self.attention_dropout(
            attention_output
        )

        x = x + attention_output

        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        feed_forward_output = self.feed_forward_dropout(
            feed_forward_output
        )

        output = x + feed_forward_output

        return output, present_key_value