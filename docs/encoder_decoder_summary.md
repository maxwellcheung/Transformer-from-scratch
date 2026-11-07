# Encoder–Decoder Transformer 阶段总结

## 1. 学习目标与完成结果

本阶段采用“Notebook 探索 → 模块化重写 → 组件级测试”的方式，从零实现了一个可训练、
可保存、可重新加载并可贪心解码的英译中 Mini Transformer。

完整链路已经打通：

```text
文本
→ 分词与词表
→ Dataset / DataLoader
→ Encoder–Decoder Transformer
→ token loss 与反向传播
→ CSV 指标与 checkpoint
→ 自回归生成
→ 文本解码
```

教学实验在三条训练样本上达到 token accuracy `1.0`，并能将 `i love you` 生成成
`我 爱 你`。这是训练链路与模型可学习性的验证，不是泛化能力评估。

如果你希望先跟着一个具体句子观察每一步向量和 shape，可以先阅读
[`从“我 爱 你”到 “i love you”`](translation_case_walkthrough.md)，再回到本篇按模块复习。

## 2. 数据与 teacher forcing

词表构建和编码必须使用相同的 token 粒度。本项目统一使用 `text.split()`，避免“构建时按
字符、编码时按单词”造成大量 `<unk>`。

对于目标句子 `我 爱 你`：

```text
tgt_input  = [<sos>, 我, 爱, 你]
tgt_output = [我, 爱, 你, <eos>]
```

两者长度相等但错开一位，使 Decoder 在每个位置根据已知前缀预测下一个 token。不同长度的
样本通过 `pad_sequence(..., batch_first=True)` 补齐；Embedding 的输入必须为整数 ID，
因此 token Tensor 使用 `torch.long`。

## 3. 位置编码

Attention 本身不理解顺序，因此在 token embedding 上加入正弦位置编码：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / D))
PE(pos, 2i+1) = cos(pos / 10000^(2i / D))
```

位置表通过 `register_buffer` 注册：它随模型移动设备和保存 checkpoint，但不是需要优化的
参数。`[1,L,D]` 的位置表能够利用广播与 `[B,L,D]` 的 embedding 相加。

实现中特别处理了奇数 `d_model`，并拒绝超过 `max_len` 的输入。

## 4. Mask

本项目约定布尔 Mask 中 `True` 表示禁止关注。

```text
源 padding mask    [B,1,1,S]
目标 padding mask  [B,1,1,T]
causal mask        [1,1,T,T]
目标组合 mask      [B,1,T,T]
```

目标组合 Mask 是 padding mask 与 causal mask 的逻辑或。causal mask 的上三角为 `True`，
保证位置 `t` 只能读取位置 `0..t`。

全屏蔽行如果直接执行 softmax，会对全 `-inf` 产生 NaN。实现先识别全屏蔽行，保证其
attention weights 和 context 都为零。

## 5. Scaled dot-product attention

```text
scores  = QK^T / sqrt(Dh)
weights = softmax(mask(scores))
context = weights V
```

缩放除以 `sqrt(Dh)` 而不是 `sqrt(D)`，因为每个头的点积发生在单头特征维度 `Dh` 上。
如果维度较大而不缩放，点积方差会增大，softmax 容易饱和并产生过小梯度。

shape 推导：

```text
Q                [B,H,Q,Dh]
K/V              [B,H,K,Dh]
Q @ K^T          [B,H,Q,K]
weights @ V      [B,H,Q,Dh]
合并多头          [B,Q,D]
```

多头拆分的是特征维度，不是序列维度，因此每个头仍然看到完整序列。

## 6. 自注意力与交叉注意力

Encoder self-attention：

```text
Q = K = V = Encoder 当前状态
```

Decoder masked self-attention：

```text
Q = K = V = Decoder 当前状态
并使用 causal mask
```

Decoder cross-attention：

```text
Q   = Decoder 当前状态       [B,H,T,Dh]
K/V = Encoder memory         [B,H,S,Dh]
权重                           [B,H,T,S]
输出                           [B,T,D]
```

Query 决定输出位置数，因此 cross-attention 输出长度是目标长度 `T`；源长度 `S` 在对
Value 的加权求和中被消去。

## 7. FFN、残差与 LayerNorm

逐位置前馈网络：

```text
FFN(x) = W2(ReLU(W1x + b1)) + b2
[B,L,D] → [B,L,Dff] → [B,L,D]
```

FFN 对每个 token 独立应用同一组参数；token 之间的信息交换来自 Attention。输出必须回到
`D`，才能与残差分支逐元素相加。

本项目采用教学版 Post-LN：

```text
x = Norm(x + Dropout(SelfAttention(x)))
x = Norm(x + Dropout(FFN(x)))
```

Decoder 多一个 cross-attention 子层，因此有三组独立的 LayerNorm。不同子层不能共享同一个
LayerNorm，因为每组 LayerNorm 都有独立可学习的缩放和平移参数。

## 8. 完整模型

```text
src_tokens [B,S]
→ source embedding × sqrt(D)
→ positional encoding
→ EncoderLayer × N
→ memory [B,S,D]

tgt_input [B,T]
→ target embedding × sqrt(D)
→ positional encoding
→ DecoderLayer × N（读取 memory）
→ hidden [B,T,D]
→ Linear(D, V_tgt)
→ logits [B,T,V_tgt]
```

源端和目标端词表不同，因此使用独立 Embedding。`nn.ModuleList` 用于注册堆叠层的参数；
普通 Python list 不会自动完成模块注册。

## 9. Loss、accuracy 与训练步骤

`cross_entropy` 接收原始 logits，内部已经执行数值稳定的 `log_softmax` 和负对数似然，
因此不能提前手动 softmax。

```text
logits  [B,T,V] → [B*T,V]
targets [B,T]   → [B*T]
```

PAD 通过 `ignore_index=pad_id` 排除。epoch 指标按有效 token 聚合：

```text
epoch loss     = sum(batch_mean_loss × batch_valid_tokens) / total_valid_tokens
epoch accuracy = total_correct_tokens / total_valid_tokens
```

不能直接平均 batch accuracy，因为不同 batch 的有效 token 数可能不同。

一次训练步骤的顺序：

```text
model.train()
→ optimizer.zero_grad(set_to_none=True)
→ forward
→ loss
→ backward
→ clip_grad_norm_
→ optimizer.step()
```

`model.eval()` 控制 Dropout 等模块行为；`torch.no_grad()` 禁止构建自动求导图；
`optimizer.zero_grad()` 才负责清除旧梯度。这三个概念不能混淆。

## 10. Checkpoint 与贪心解码

可靠 checkpoint 同时保存：

```text
model_config
model_state_dict
src_vocab_tokens
tgt_vocab_tokens
training_args
```

`state_dict` 只有参数 Tensor，不能说明层数、宽度、头数和词表大小。加载时必须先用配置重建
同构模型，再加载权重；配置缺失或权重 shape 不匹配都应明确失败。

贪心解码过程：

```text
Encoder 只运行一次得到 memory
generated 从 <sos> 开始
循环：Decoder(generated) → 取最后位置 logits → argmax → 追加 token
遇到 <eos> 后停止
```

只使用最后位置 hidden state，是因为 causal self-attention 已经让它包含整个可见前缀的信息；
前面位置的 logits 对应已经完成的旧预测。

Notebook 修改模块后若出现“磁盘代码已有方法，但运行时 AttributeError”，通常是内核仍缓存
旧模块。重启内核最可靠；使用 `importlib.reload` 时必须按依赖顺序重新加载。

## 11. 测试覆盖的关键不变量

- 词表构建与编码 token 粒度一致，越界 ID 明确失败
- 空语料、空数据集和空样本明确失败
- 位置编码支持奇数宽度并拒绝超长输入
- Mask shape/dtype 受控，不允许意外广播
- 全屏蔽 attention 不产生 NaN
- 多头注意力支持 self/cross attention 并能反向传播
- Encoder 忽略源 PAD，Decoder 隔离未来目标 token
- 完整模型输出 shape 正确且梯度有限
- loss/accuracy 忽略 PAD，全 PAD 目标明确失败
- 单 batch 能够过拟合，证明训练链路可学习
- checkpoint 往返保持 logits 和词表映射
- 缺字段或配置/权重不匹配的 checkpoint 明确失败
- 贪心解码遇到 EOS 提前停止、Encoder 只执行一次并恢复原训练模式

## 12. 衔接：Decoder-only GPT 与 KV Cache

在 Encoder–Decoder 基础上，项目已经继续完成：

```text
Decoder-only GPT
→ 移除 Encoder 与 cross-attention
→ 单序列 next-token prediction
→ Pre-LN causal self-attention blocks
→ 普通自回归生成
→ 分层缓存历史 K/V
→ 验证有无 KV Cache 的 logits 与生成结果一致
```

KV Cache 不改变模型语义，而是避免生成时反复投影和处理全部历史 token。下一份总结会从
Encoder–Decoder 已有知识出发，循序解释 GPT 训练、普通生成、prefill 与单 token decode：

[`gpt_kv_cache_summary.md`](gpt_kv_cache_summary.md)

