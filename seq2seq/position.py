import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.0):
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f'd_model must be positive, got {d_model}'
            )

        if max_len <= 0:
            raise ValueError(
                f'max_len must be positive, got {max_len}'
            )

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # unsqueeze(1) 让 position 变成列向量 [max_len, 1]，
        # 与行向量 div_term 相乘后得到外积形式的矩阵
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # i 是维度索引，取值 0, 1, 2, ..., d_model/2 - 1
        # 2i 表示偶数维度，2i+1 表示奇数维度
        # 每一对 (2i, 2i+1) 共用同一个频率 div_term[i]
        # 根据原公式进行指数和代数变换
        # 用 exp(log(...)) 的写法：数值上更稳定，尤其是当 d_model 很大时；
        # 避免先计算 10000 ** (2i/d_model) 再取倒数可能带来的精度问题
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -math.log(10000.0) / d_model
        )

        pe = torch.zeros(max_len, d_model)

        # 偶数维度
        pe[:,0::2] = torch.sin(position * div_term)

        # 防止d_model为奇数时形状不匹配报错
        # div_term 是按偶数维度数量生成的，所以它比奇数维度数量多 1，
        # 赋给奇数维度时必须截掉最后一个
        odd_width = pe[:,1::2].shape[1]

        # 奇数维度
        pe[:,1::2] = torch.cos(position * div_term[:odd_width])

        self.register_buffer('pe', pe.unsqueeze(0),)

    def forward(self,x):
        if x.dim() != 3:
            raise ValueError(
                f'expected x to have 3 dim, got {x.dim()}'
            )

        actual_width = x.size(-1)

        if actual_width != self.d_model:
            raise ValueError(
                f'expected model with {self.d_model},'
                f'got {actual_width}'
            )

        sequence_length = x.size(1)

        if sequence_length > self.max_len:
            raise ValueError(
                f'sequence length {sequence_length} '
                f'exceeds max_len {self.max_len}'
            )

        positional = self.pe[:,:sequence_length,:]
        positional = positional.to(dtype=x.dtype)

        return self.dropout(x + positional)