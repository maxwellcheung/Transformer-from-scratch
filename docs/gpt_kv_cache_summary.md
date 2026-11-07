# Decoder-only GPT 与 KV Cache：从“会写”到“真正理解”

这份文档不是 API 清单，而是一条可以反复走的学习路线。第一次阅读时，可以从头顺着故事走；
以后复习时，可以直接跳到 shape 总表、生成时间线和排错清单。

项目采用的学习方法是：

```text
Notebook 探索 → 模块化重写 → 组件级测试 → 端到端验证
```

你已经亲手完成了数据、GPT Block、完整模型、训练、普通生成和 KV Cache。现在要做的，是把
这些看似分散的代码重新串成一幅完整地图。

## 1. 先看全局：我们究竟做了什么

Encoder–Decoder Transformer 面向“输入一段内容，再生成另一段内容”的任务。源句经过
Encoder 变成 memory，Decoder 一边读取目标前缀，一边通过 cross-attention 读取 memory。

Decoder-only GPT 更干脆：它只有一条 token 序列，只做一件事——根据左侧上下文预测下一个
token。

```text
token IDs [B,T]
    ↓
Token Embedding + Position Embedding
    ↓
Pre-LN GPTBlock × N
    ↓
Final LayerNorm
    ↓
Linear(d_model, vocab_size)
    ↓
logits [B,T,V]
```

这里没有 Encoder，也没有 cross-attention。每个 GPT Block 只有 causal self-attention 和
逐位置 FeedForward。

对应代码地图：

- [`gpt/data.py`](../gpt/data.py)：把文本变成 next-token 训练样本
- [`gpt/layers.py`](../gpt/layers.py)：Pre-LN GPT Block
- [`gpt/model.py`](../gpt/model.py)：Embedding、Block 堆叠、lm_head 与分层 Cache
- [`gpt/training.py`](../gpt/training.py)：一次完整参数更新
- [`gpt/inference.py`](../gpt/inference.py)：普通生成与 KV Cache 生成
- [`seq2seq/attention.py`](../seq2seq/attention.py)：共享的多头注意力和缓存注意力
- [`tests/test_kv_cache.py`](../tests/test_kv_cache.py)：Cache 正确性的分层证据

## 2. 从一句话开始：next-token prediction

假设训练文本是：

```text
i love cats
```

词表编码时先加入 `<sos>`，末尾加入 `<eos>`：

```text
[<sos>, i, love, cats, <eos>]
```

然后错开一位：

```text
input_ids  [<sos>, i,    love, cats]
targets    [i,     love, cats, <eos>]
```

两者长度相同，不是巧合。模型在每个输入位置都要交出一个答案：

```text
看到 <sos>          应预测 i
看到 <sos> i        应预测 love
看到 <sos> i love   应预测 cats
看到完整句子前缀     应预测 <eos>
```

这就是 [`LanguageModelDataset`](../gpt/data.py) 中的核心：

```python
input_ids = tokens[:-1]
targets = tokens[1:]
```

不同句子长度不同时，collate 函数用 `<pad>` 补齐。`<pad>` 只是为了组成规则 Tensor，并不是
需要学习的语言内容，所以 loss 和 accuracy 都必须忽略它。

请牢牢记住第一条主线：

```text
训练数据的错位，决定了模型每个位置应该预测谁。
```

## 3. 为什么训练时能并行，生成时却要逐个来

训练时，整句文本已经在数据集中。模型可以一次产生全部位置的 logits，但 causal mask 会
遮住未来位置，所以位置 t 只能读取位置 0..t。

```text
位置 0：看 0
位置 1：看 0,1
位置 2：看 0,1,2
位置 3：看 0,1,2,3
```

因此训练同时满足两点：

1. 计算可以并行；
2. 信息不能从未来泄漏到过去。

推理时不同。下一个 token 还不存在，必须先生成它，才能把它作为后续上下文。因此自回归生成
天然是串行过程。

## 4. Embedding：token 身份与位置缺一不可

`input_ids` 的 shape 是 `[B,T]`，里面只是整数索引。Token Embedding 把每个 ID 查表为
`d_model` 维向量：

```text
token IDs             [B,T]
token vectors         [B,T,D]
```

Attention 本身不知道先后顺序，所以还要加入位置向量。本项目的 GPT 使用可学习位置 Embedding：

```text
position IDs            [T]
position vectors        [T,D]
```

`[T,D]` 与 `[B,T,D]` 相加时，PyTorch 把前者视为 `[1,T,D]`，沿 batch 维广播：

```text
x = token_vectors + position_vectors    [B,T,D]
```

普通前向的位置编号从 0 开始。KV Cache 增量解码时却不能每轮都从 0 开始；新 token 的位置必须
接在历史之后。这一点后面会再次出现。

## 5. GPT Block：先归一化，再走子层

本项目使用 Pre-LN：

```text
x = x + SelfAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

拆开看第一行：

1. `norm1(x)` 让输入分布更稳定；
2. 归一化结果同时作为 Q、K、V，这是 self-attention；
3. attention 输出经过 dropout；
4. 与进入子层前的 x 做残差相加。

第二行同理，只是子层换成 FeedForward。

为什么 FeedForward 不改变 shape？因为残差连接要求两边能逐元素相加：

```text
[B,T,D] → Linear(D,Dff) → 激活 → Linear(Dff,D) → [B,T,D]
```

FeedForward 在每个 token 位置独立执行；token 之间的信息交流来自 attention。

和 Encoder–Decoder 阶段的 Post-LN 对照：

```text
Post-LN   x = LayerNorm(x + Sublayer(x))
Pre-LN    x = x + Sublayer(LayerNorm(x))
```

两种结构不要混写。尤其在 Cache 路径中，K/V 必须由 `LayerNorm(x)` 计算，才能与普通路径保持
完全一致。

## 6. Causal self-attention 的 shape 故事

设：

```text
B   batch size
T   sequence length
D   d_model
H   num_heads
Dh  head_dimension = D / H
```

线性投影后仍是 `[B,T,D]`，拆头后变成：

```text
Q/K/V                 [B,H,T,Dh]
Q @ K.transpose       [B,H,T,T]
attention weights     [B,H,T,T]
weights @ V           [B,H,T,Dh]
合并多头               [B,T,D]
```

多头拆分的是特征维度 D，不是序列长度 T。每个头仍然看到完整序列，只是在不同特征子空间中
学习关注关系。

缩放点积注意力除以 `sqrt(Dh)`：

```text
scores = QKᵀ / sqrt(Dh)
```

因为每个头的点积发生在 Dh 维空间，而不是完整 D 维空间。缩放可以避免维度增大后点积过大、
softmax 过度饱和。

## 7. Mask：True 的含义一定要先说清楚

本项目约定：

```text
mask == True  表示禁止关注
```

训练时组合两类 Mask：

```text
causal mask          [1,1,T,T]
padding mask         [B,1,1,T]
combined mask        [B,1,T,T]
```

两者用逻辑或合并。一个位置只要属于未来，或者是 `<pad>`，就应该被屏蔽。

如果一整行都被屏蔽，直接对全 `-inf` 做 softmax 会产生 NaN。共享 attention 实现先识别这种
情况，让相应 weights 和 context 为零。这是小模型也值得认真处理的数值边界。

## 8. 完整 GPT：为什么输出是 [B,T,V]

经过多个 GPT Block 后，hidden state 仍是 `[B,T,D]`。最终 lm_head 是：

```python
nn.Linear(d_model, vocab_size)
```

它独立地把每个位置的 D 维表示映射到 V 个词表候选，因此：

```text
hidden    [B,T,D]
logits    [B,T,V]
```

`logits[b,t,v]` 表示第 b 条样本在位置 t 对词表 token v 的未归一化评分。

这里不要提前手动 softmax。训练时 `cross_entropy` 会完成数值稳定的 log-softmax；生成时
argmax 直接比较 logits 大小也不需要 softmax。

## 9. 一次训练步骤究竟发生了什么

GPT 的 batch 是：

```text
(input_ids, targets)
```

它与 Seq2Seq 的 `(src_tokens, tgt_input, tgt_output)` 不同，所以需要独立 `train_step`；但
token loss 和 token accuracy 只关心 `[B,T,V]` 与 `[B,T]`，可以复用。

标准顺序是：

```text
model.train()
→ optimizer.zero_grad(set_to_none=True)
→ logits = model(input_ids)
→ loss
→ loss.backward()
→ clip_grad_norm_
→ optimizer.step()
```

`model.eval()`、`torch.no_grad()` 和 `optimizer.zero_grad()` 各管一件事：

- `eval()` 改变 Dropout 等模块的行为；
- `no_grad()` 不构建求导图；
- `zero_grad()` 清除旧梯度。

三者不能互相替代。

单句 `i love cats` 的教学实验最终得到：

```text
loss              0.0004
token accuracy    1.0000
predictions       [i, love, cats, <eos>]
```

这证明训练链路能学习，不证明模型具备语言泛化能力。一个 batch 都无法过拟合通常说明实现有
问题；能够过拟合只是必要条件，不是最终目标。

## 10. 普通自回归生成：正确，但重复

无 Cache 贪心生成从 prompt 开始，每轮执行：

```text
完整 generated → GPT → 取 logits[:, -1, :] → argmax → 拼接
```

示例：

```text
[<sos>]                    → i
[<sos>, i]                 → love
[<sos>, i, love]           → cats
[<sos>, i, love, cats]     → <eos>
```

只取最后位置，是因为它已经通过 causal attention 汇总了当前可见前缀。前面位置的 logits 是
以前那些预测位置的答案。

`max_new_tokens` 防止模型永不结束；预测到 `<eos>` 时则可以提前停止。生成函数会暂时调用
`model.eval()`，并在 finally 中恢复调用前的训练/评估状态。

问题在于：第 4 轮又计算了一遍 `<sos>`、`i`、`love` 和 `cats`。模型越大、上下文越长，
这种重复越昂贵。

## 11. KV Cache：先理解“为什么只缓存 K 和 V”

生成下一个 token 时，我们只关心新位置的输出。因此只需要新位置的 Query：

```text
Q_new 查询所有历史 Key
得到权重后加权所有历史 Value
```

旧 Query 对新的输出已经没有用途，所以不需要保存。历史 K/V 仍会被新 Query 使用，因此保存
它们。

假设已经缓存 3 个 token，现在来了 1 个新 token：

```text
past_key      [B,H,3,Dh]
new_key       [B,H,1,Dh]
all_key       [B,H,4,Dh]
new_query     [B,H,1,Dh]
scores        [B,H,1,4]
context       [B,H,1,Dh]
```

K/V 沿 `dim=2` 拼接，因为拆头后的第 2 维是序列长度。沿 `dim=3` 拼接会错误地改变单头特征
宽度 Dh。

FeedForward 不需要 Cache，因为它对每个 token 独立计算；历史 token 的 FFN 输出不会参与
当前 token 的 FFN。

## 12. 为什么每一层都要自己的 Cache

第 0 层和第 1 层有不同的输入隐藏状态，也有不同的 `k_proj`、`v_proj` 参数，所以它们得到的
K/V 不可能共用。

完整模型的 Cache 是：

```text
past_key_values = [
    (layer_0_key, layer_0_value),
    (layer_1_key, layer_1_value),
    ...
]
```

每层 Cache 的 shape 是 `[B,H,T,Dh]`，列表长度等于 GPT Block 数量。

这也解释了项目为何分三级验证 Cache：

1. 先验证 MultiHeadAttention；
2. 再验证包含 LayerNorm、残差和 FFN 的 GPTBlock；
3. 最后验证多层 GPTModel。

逐层验证可以迅速定位错误，而不是等最终文本错了才猜测问题在哪。

## 13. Prefill 与 Decode：Cache 生成的两幕

### 第一幕：Prefill

模型一次接收完整 prompt：

```text
prompt [B,P] → logits [B,P,V] + 每层初始 K/V
```

因为 prompt 中有多个位置同时计算，仍然必须使用 causal mask。prompt 最后位置的 logits 已经
能够预测第一个新 token。

### 第二幕：Decode

生成第一个新 token 后，只把这个新 token `[B,1]` 输入模型：

```text
new token + past_key_values
→ 当前 token logits [B,1,V]
→ 更新后的 present_key_values
```

此时 K/V 中只存在过去和当前，没有未来位置，所以单 token Query 不需要 causal mask。若 batch
中有 padding，padding mask 仍需要另外处理；当前教学版缓存路径面向无 padding prompt。

最容易写错的一行是：

```python
position_ids = torch.arange(past_length, total_length)
```

增量 token 的位置必须接在历史后面。如果每轮又从 0 开始，token 身份虽然正确，位置身份却错
了，缓存 logits 就无法和完整计算一致。

## 14. 一条完整的 Cache 时间线

以 `<sos>` 生成 `i love cats <eos>` 为例：

```text
Prefill:
输入 <sos>
Cache = [<sos> 的 K/V]
最后 logits 预测 i

Decode 1:
只输入 i
Cache = [<sos>, i] 的 K/V
新 logits 预测 love

Decode 2:
只输入 love
Cache = [<sos>, i, love] 的 K/V
新 logits 预测 cats

Decode 3:
只输入 cats
Cache = [<sos>, i, love, cats] 的 K/V
新 logits 预测 <eos>

停止:
<eos> 已经生成，不再需要把它输入模型预测后续 token
```

注意：把整个 `generated` 再传给带历史 Cache 的模型是错误的。那会把旧 token 的 K/V 重复追加。

## 15. Cache 节省了什么，又付出了什么

设 prompt 长度为 P，实际生成 N 个 token。无 Cache 时，每轮重新输入完整前缀：

```text
P + (P+1) + ... + (P+N-1)
= N×P + N×(N-1)/2
```

有 Cache 时，prefill 处理 P 个 token，此后每轮通常只处理 1 个新 token。若最后生成 EOS 就
停止，处理数量约为：

```text
P + N - 1
```

教学例子中 P=1、N=4：

```text
无 Cache    1+2+3+4 = 10
有 Cache    1+1+1+1 = 4
```

代价是保存 K/V。所有层的 Cache 元素数量约为：

```text
2 × num_layers × B × T × d_model
```

前面的 2 来自 Key 和 Value。再乘 dtype 的字节数，才是实际内存占用。所以 KV Cache 的本质
是：

```text
用显存换取更少的重复计算。
```

当前实现用 `torch.cat` 逐步扩展 Cache，适合教学。真实高性能推理常预分配 Cache 空间，避免
反复分配和拷贝。

## 16. 我们怎样证明 Cache 没有改变模型

“能生成文本”不是充分证据，因为错误实现也可能碰巧生成相同的短句。本项目检查更强的不变量：

```text
完整 attention 最后位置 output
≈
prefill + cached decode 的 output
```

验证分四层：

1. MultiHeadAttention 最后位置一致；
2. GPTBlock 最后位置一致；
3. GPTModel 最后位置 logits 一致；
4. 普通贪心生成与 Cache 贪心生成 token 完全相同。

最终结果：

```text
uncached    [<sos>, i, love, cats, <eos>]
cached      [<sos>, i, love, cats, <eos>]
完全相同    True
```

完整回归测试为 `70 passed`。近似比较 logits 使用容差，是因为浮点运算次序变化可能产生极小
误差；最终 token 序列则要求完全相同。

## 17. 当前教学实现的边界

理解边界比假装代码无所不能更重要。当前版本有意保持简单：

- 缓存 prefill 面向无 padding prompt；
- 有历史 Cache 时，每次只允许输入一个新 token；
- Cache 使用 `torch.cat`，没有预分配；
- 生成采用 greedy argmax，没有 temperature、top-k、top-p；
- GPT 尚未提供独立 checkpoint 和命令行训练入口。

这些不是核心原理缺失，而是清晰的后续工程扩展点。

## 18. 常见错误与排查顺序

### 错误一：`MultiHeadAttention.forward()` 缺少 key 和 value

Self-attention 仍需显式传三次同一个 Tensor：

```python
attention(x, x, x, mask=mask)
```

### 错误二：模型输出是 None

`forward()` 漏写 `return output`。看到 `NoneType has no attribute shape` 时，先从返回值查起。

### 错误三：未来 token 改变了过去输出

检查 causal mask 的方向、布尔含义和 shape。本项目中 True 是屏蔽，上三角应为 True。

### 错误四：Cache shape 越来越奇怪

检查是否沿 `dim=2` 拼接，以及 decode 是否误传了完整 generated。

### 错误五：Cache 与完整 logits 不一致

依次检查：

1. Cache K/V 是否来自 Pre-LN 后的输入；
2. 新位置是否从 `past_length` 开始；
3. 每层是否使用自己的 Cache；
4. prefill 是否使用 causal mask；
5. dropout 是否在比较时关闭。

### 错误六：磁盘代码有新函数，Notebook 却 ImportError

Python 内核缓存了旧模块。可以重启 Kernel，或者：

```python
import importlib
import gpt.inference
importlib.reload(gpt.inference)
```

## 19. Shape 速查表

```text
input_ids                         [B,T]
targets                           [B,T]
token embedding                   [B,T,D]
position embedding                  [T,D]
GPT hidden                        [B,T,D]
训练 logits                       [B,T,V]

拆头 Q/K/V                        [B,H,T,Dh]
训练 attention scores             [B,H,T,T]

cached new Query                  [B,H,1,Dh]
cached all Key/Value              [B,H,T,Dh]
cached attention scores           [B,H,1,T]
单层 cache                        (K, V)
多层 cache                        list[(K, V)]
```

## 20. 建议的复习方法

不要只读文档。按下面顺序，遮住答案，自己说一遍：

1. 用纸写出 `input_ids` 与 `targets` 的一位错开；
2. 从 `[B,T,D]` 推导拆头后的 Q/K/V 和 scores；
3. 解释为什么训练并行、推理串行；
4. 画出 Pre-LN GPT Block 的两条残差；
5. 手推一次 prefill 和两次 decode 的 Cache 长度；
6. 解释为什么缓存 K/V 而不缓存旧 Q；
7. 解释为什么位置编号必须从 `past_length` 开始；
8. 不看代码写出普通生成和缓存生成的伪代码；
9. 最后运行 `python -m pytest -q`，用测试验证记忆。

如果这九步都能独立完成，你掌握的就不再是若干代码片段，而是一套能够迁移到更大 GPT、
采样策略和高性能推理系统中的基本思维框架。

## 21. 最后的六个自测问题

1. 为什么 GPT 不需要 cross-attention？
2. 为什么 targets 与 input_ids 等长，却错开一位？
3. 为什么 causal mask 让训练并行成为可能？
4. 为什么每层都必须有独立 K/V Cache？
5. 为什么 prefill 需要 causal mask，而单 token decode 通常不需要？
6. KV Cache 减少了什么计算，又增加了什么资源开销？
