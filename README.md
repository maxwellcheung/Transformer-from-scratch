# Transformer from Scratch

一个面向学习的 PyTorch Transformer 实现，包含 Encoder–Decoder、Decoder-only GPT 与 KV Cache。项目不调用
`nn.Transformer`、`nn.MultiheadAttention` 或 PyTorch 内置的
`scaled_dot_product_attention`，而是从词表、Mask 和缩放点积注意力开始，逐步完成训练、
checkpoint、自回归翻译和增量生成。

当前里程碑：Encoder–Decoder、Decoder-only GPT 与 KV Cache 核心路线均已完成。

## 已实现内容

- 空格级分词词表，以及 `<pad>/<sos>/<eos>/<unk>` 特殊 token
- 翻译 Dataset、teacher forcing 目标错位和动态 padding
- 正弦位置编码（支持奇数 `d_model`）
- padding mask、causal mask 和安全的全屏蔽 attention 行为
- 手写 scaled dot-product attention 与 multi-head attention
- Post-LN Encoder Layer 与 Decoder Layer
- 完整 Encoder–Decoder Transformer
- 忽略 PAD 的 token loss、accuracy、梯度裁剪与训练循环
- CSV 指标记录、checkpoint 保存/恢复和贪心解码
- shape、梯度、Mask 隔离、过拟合与损坏 checkpoint 等自动化测试
- next-token prediction 数据错位与 Decoder-only GPT Dataset
- Pre-LN GPT Block、可学习位置嵌入与完整 GPT 语言模型
- GPT token loss、accuracy、单步训练和无 Cache 贪心生成
- Attention、GPT Block、完整模型三级 KV Cache
- prefill / decode 两阶段缓存生成，以及缓存与非缓存结果一致性测试

## 项目结构

```text
from_scratch/
├── seq2seq/
│   ├── vocab.py          # 词表与 checkpoint 恢复
│   ├── data.py           # Dataset、teacher forcing、padding
│   ├── position.py       # 正弦位置编码
│   ├── masks.py          # padding/causal mask
│   ├── attention.py      # scaled attention 与多头注意力
│   ├── layers.py         # FFN、EncoderLayer、DecoderLayer
│   ├── model.py          # 完整 Encoder–Decoder Transformer
│   ├── training.py       # loss、accuracy、train_step
│   ├── train.py          # 可复现训练入口与 CSV 指标
│   ├── checkpoint.py     # checkpoint 保存与加载
│   └── inference.py      # 贪心解码
├── gpt/
│   ├── data.py           # next-token Dataset 与动态 padding
│   ├── layers.py         # Pre-LN GPT Block
│   ├── model.py          # Decoder-only GPT 与分层 KV Cache
│   ├── training.py       # GPT train_step
│   └── inference.py      # 普通/缓存贪心生成
├── tests/                # 组件与端到端测试
├── docs/
│   ├── encoder_decoder_summary.md
│   └── gpt_kv_cache_summary.md
└── scratch.ipynb         # Notebook 探索记录
```

## 快速验证

在本目录执行：

```bash
python -m pytest -v
```

## 训练一个教学模型

```bash
python -m seq2seq.train \
  --epochs 30 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --d-model 16 \
  --d-ff 32 \
  --num-heads 2 \
  --num-layers 1 \
  --dropout 0.0 \
  --max-samples 3 \
  --device cpu \
  --output-dir outputs/toy_run
```

训练目录会包含：

```text
outputs/toy_run/
├── metrics.csv
└── checkpoint.pt
```

`outputs/` 和模型权重默认不提交到 Git；它们可以通过训练命令重新生成。

## 加载与翻译

```python
import torch

from seq2seq.checkpoint import load_checkpoint
from seq2seq.inference import greedy_decode

model, src_vocab, tgt_vocab, metadata = load_checkpoint(
    "outputs/toy_run/checkpoint.pt",
    device="cpu",
)

src_tokens = torch.tensor([
    src_vocab.encode("i love you")
])

generated = greedy_decode(
    model,
    src_tokens,
    sos_id=tgt_vocab.sos_id,
    eos_id=tgt_vocab.eos_id,
    max_new_tokens=10,
)

print(tgt_vocab.decode(generated[0].tolist()))
```

在三条教学语料上的一次示例实验中，训练到第 30 轮时 token accuracy 达到 `1.0`，
`i love you` 可生成 `我 爱 你`。这只证明模型能够拟合教学数据，不代表真实翻译泛化能力。

## 核心 shape

设 batch、源长度、目标长度、模型宽度、头数和单头宽度分别为
`B、S、T、D、H、Dh`，其中 `Dh = D / H`：

```text
源/目标 embedding          [B,S,D] / [B,T,D]
拆头后的 Q/K/V            [B,H,L,Dh]
self-attention weights     [B,H,L,L]
cross-attention weights    [B,H,T,S]
Encoder memory             [B,S,D]
Decoder hidden             [B,T,D]
目标 logits                [B,T,V_tgt]
```

更多原理、公式、常见错误和复习清单见
[`docs/encoder_decoder_summary.md`](docs/encoder_decoder_summary.md)。

## Decoder-only GPT 与 KV Cache

GPT 训练使用同一序列的一位错开目标：

```text
input_ids  <sos>  i     love  cats
targets    i      love  cats  <eos>
```

普通生成每轮重新计算完整前缀；KV Cache 首先对 prompt 做 prefill，然后每轮只处理一个新
token。项目通过数值测试验证了两条路径的最终 logits 和贪心生成结果一致。

关键 shape：

```text
当前 Query                 [B,H,1,Dh]
历史 K/V + 当前 K/V        [B,H,T,Dh]
增量 attention scores      [B,H,1,T]
每层 Cache                 (key, value)
全模型 Cache               list[(key, value)]
```

完整原理、代码地图、计算量与显存分析、常见错误和复习清单见
[`docs/gpt_kv_cache_summary.md`](docs/gpt_kv_cache_summary.md)。

## 学习结果

- Encoder–Decoder 教学模型可以把 `i love you` 生成成 `我 爱 你`
- GPT 单样本实验达到 loss `0.0004`、token accuracy `1.0`
- 普通生成与 KV Cache 生成均得到 `i love cats`
- 当前完整回归测试结果为 `70 passed`

## 可继续扩展的方向

- GPT checkpoint 与独立训练 CLI
- temperature、top-k 和 top-p 采样
- 批量不等长 prompt 的 Cache 与 padding mask
- 预分配 KV Cache，减少反复 `torch.cat`

