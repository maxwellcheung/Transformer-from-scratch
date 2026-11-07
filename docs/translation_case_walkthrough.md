# 从“我 爱 你”到 “i love you”：一条句子走完整个 Transformer

这份文档只追踪一件事：一句中文怎样进入 Encoder–Decoder Transformer，经过一连串向量变换，
最后一步步生成英文 `i love you`。

我们不会一上来就堆公式。先抓住完整路线，再把每个模块拆开，最后回到一个更重要的问题：

```text
模型为什么会获得翻译能力？
```

阅读时请始终带着三个问题：

1. 当前 Tensor 的 shape 是什么？
2. 当前模块解决什么问题？
3. 这个模块怎样帮助最终预测下一个英文 token？

---

## 1. 先说明案例与当前项目的关系

当前项目的教学实验曾使用 `i love you → 我 爱 你`。本文为了从中文视角讲清数据流，反过来
使用：

```text
我 爱 你 → i love you
```

Transformer 架构对翻译方向没有偏好。只要交换源语料和目标语料、分别重建源词表和目标词表，
相同模型就可以学习中文到英文。

项目使用 `text.split()` 按空格分词，因此本文写作：

```text
我 爱 你
```

而不是没有空格的 `我爱你`。在真实中文系统中，通常会使用字级 tokenizer、BPE、
SentencePiece 等更成熟的分词方法；这里保留空格，是为了把注意力集中在 Transformer 本身。

---

## 2. 先看完整旅程

训练阶段的一条样本会经历：

```text
源文本：我 爱 你
目标文本：i love you
        ↓ 分词
源 token：我 / 爱 / 你
目标 token：i / love / you
        ↓ 查词表
源 ID：[4, 5, 6]
目标输入：[<sos>, i, love, you]
目标答案：[i, love, you, <eos>]
        ↓ Embedding + 位置编码
源向量 [B,S,D]
目标向量 [B,T,D]
        ↓
Encoder：理解源句各词之间的关系
        ↓
memory [B,S,D]
        ↓
Decoder：读取英文前缀，并通过 cross-attention 查询中文 memory
        ↓
decoder hidden [B,T,D]
        ↓ 输出投影
logits [B,T,V_tgt]
        ↓ Cross Entropy
loss
        ↓ 反向传播与参数更新
模型逐渐学会翻译
```

推理阶段则是：

```text
我 爱 你 → Encoder 只运行一次 → memory

<sos>             → 预测 i
<sos> i           → 预测 love
<sos> i love      → 预测 you
<sos> i love you  → 预测 <eos>
```

训练一次算出所有位置的答案；推理必须逐个生成。这是后文最重要的区别之一。

---

## 3. 固定一个小模型，后面所有 shape 都用它

为了不让字母漂浮在空中，我们固定一个教学配置：

```text
batch size B          = 1
源长度 S               = 3
目标输入长度 T          = 4
模型宽度 D              = 8
注意力头数 H            = 2
单头宽度 Dh             = D/H = 4
FFN 中间宽度 Dff         = 16
Encoder 层数 N          = 2
Decoder 层数 N          = 2
```

对应配置大致是：

```python
TransformerConfig(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_layers=2,
    dropout=0.0,
    src_pad_id=src_vocab.pad_id,
    tgt_pad_id=tgt_vocab.pad_id,
    max_len=32,
)
```

请注意 `num_layers=2` 的真实含义：

```text
2 个 EncoderLayer
+
2 个 DecoderLayer
```

不是 Encoder 和 Decoder 加起来一共 2 层。

---

## 4. 第一步：文本变成 token

源文本：

```text
我 爱 你
```

分词结果：

```python
['我', '爱', '你']
```

目标文本：

```text
i love you
```

分词结果：

```python
['i', 'love', 'you']
```

token 是词表处理的最小单位。模型并不直接看到“爱”这个汉字，也不直接看到字符串 `love`；
它只看到每个 token 对应的整数 ID。

如果构建词表时按字符、编码时按单词，token 粒度不一致，就会产生大量 `<unk>`。因此构建和
编码必须使用相同 tokenizer。

---

## 5. 第二步：token 变成整数 ID

为了讲解方便，假设两个词表如下。真实 ID 由语料频率和首次出现顺序决定，但原理相同。

源词表：

```text
0 <pad>
1 <sos>
2 <eos>
3 <unk>
4 我
5 爱
6 你
```

目标词表：

```text
0 <pad>
1 <sos>
2 <eos>
3 <unk>
4 i
5 love
6 you
```

于是源句编码为：

```text
src = [4, 5, 6]
shape = [3]
dtype = torch.int64
```

整数 ID 只是查表索引。ID 5 并不比 ID 4“更爱”或“语义更大”，数字之间的距离没有语言意义。

Embedding 接收整数索引，因此输入必须是 `torch.long`，不能是浮点数。

---

## 6. 目标为什么要拆成 input 和 output

训练 Decoder 时，目标句被拆成两条错开一位、长度相同的序列：

```text
tgt_input   [<sos>, i,    love, you]
tgt_output  [i,     love, you,  <eos>]
```

对应 ID：

```text
tgt_input   [1, 4, 5, 6]
tgt_output  [4, 5, 6, 2]
```

每个位置的任务是：

```text
输入 <sos>              正确答案 i
输入前缀 <sos> i        正确答案 love
输入前缀 ... love       正确答案 you
输入前缀 ... you        正确答案 <eos>
```

这种训练方式叫 teacher forcing：训练时把正确的历史目标 token 提供给 Decoder，让它学习
预测下一位。

为什么两者要等长？因为 Decoder 每输出一个位置的 logits，就需要一个一一对应的正确答案。

---

## 7. Dataset 与 batch：第一批 Tensor 出现了

`TranslationDataset.__getitem__()` 返回：

```text
src         [S] = [3]
tgt_input   [T] = [4]
tgt_output  [T] = [4]
```

DataLoader 的 collate 函数把多条样本补齐并堆成 batch。本文只有一条样本，所以：

```text
src_tokens   [B,S] = [1,3]
tgt_input    [B,T] = [1,4]
tgt_output   [B,T] = [1,4]
```

如果同一 batch 中另一个源句只有两个 token，它会补成：

```text
[token_a, token_b, <pad>]
```

`<pad>` 只是形状占位符，不应被 attention 当成真实内容，也不应参与 loss 和 accuracy。

---

## 8. Mask：先把不能看的位置标出来

项目约定：

```text
mask == True  表示禁止关注
```

### 8.1 源 padding mask

源序列用于 Encoder self-attention，也用于 Decoder cross-attention。源 padding mask 是：

```text
src_mask [B,1,1,S]
本文 shape = [1,1,1,3]
```

本文没有 padding，所以三个位置都是 False。

### 8.2 目标 padding mask

```text
tgt_padding_mask [B,1,1,T]
本文 shape = [1,1,1,4]
```

### 8.3 目标 causal mask

causal mask 禁止当前位置看到未来目标：

```text
位置 0 允许看 0
位置 1 允许看 0,1
位置 2 允许看 0,1,2
位置 3 允许看 0,1,2,3
```

矩阵中上三角为 True：

```text
False True  True  True
False False True  True
False False False True
False False False False
```

shape：

```text
causal_mask [1,1,T,T] = [1,1,4,4]
```

最终目标 mask：

```text
tgt_mask = tgt_padding_mask | causal_mask
shape = [B,1,T,T] = [1,1,4,4]
```

源端不需要 causal mask，因为 Encoder 理解的是一条已经完整给出的源句，每个源 token 可以读取
整条源句。

---

## 9. ID 怎样真正变成向量

Embedding 可以理解为一张可学习表格。源词表大小如果是 7，`d_model=8`，源 Embedding 参数
矩阵就是：

```text
E_src [7,8]
```

查 ID 4，就取第 4 行；查 ID 5，就取第 5 行。

```text
src IDs             [1,3]
src embedding       [1,3,8]
```

示意上，`我` 的向量可能是：

```text
[0.12, -0.31, 0.08, 0.44, -0.17, 0.09, 0.25, -0.06]
```

这只是示意，不是预先规定的含义。训练开始时 Embedding 近似随机；反向传播会不断调整这些值。

从数学上看，Embedding 查表等价于：

```text
token 的 one-hot 向量 × Embedding 矩阵
```

但 `nn.Embedding` 不需要真的构造巨大 one-hot，直接按 ID 取行，更高效。

源端和目标端为什么使用两个 Embedding？因为中文源词表和英文目标词表不是同一套 token：

```text
src_embedding  学习中文 token 的表示
tgt_embedding  学习英文 token 的表示
```

---

## 10. 为什么 Embedding 还要乘 sqrt(D)

当前实现执行：

```python
x = embedding(tokens)
x = x * sqrt(d_model)
```

本文 `D=8`，所以乘 `sqrt(8)`。这是原始 Transformer 的做法，用于调节 token embedding
与位置编码的数值尺度，让两者相加时更平衡。

这一步不改变 shape：

```text
[1,3,8] → [1,3,8]
```

---

## 11. 位置编码：让模型知道“我、爱、你”的顺序

如果只给 Attention 三个 token 向量，它并不知道谁在前、谁在后。交换 token 顺序时，纯
self-attention 本身没有固定的位置概念。

项目使用正弦位置编码：

```text
偶数维使用 sin
奇数维使用 cos
不同维度使用不同频率
```

位置 0 的编码以：

```text
[0, 1, 0, 1, ...]
```

开头，因为 `sin(0)=0`、`cos(0)=1`。

源向量与对应位置向量相加：

```text
src embedding         [1,3,8]
src positional        [1,3,8]
相加后 x              [1,3,8]
```

目标端也独立加入位置编码：

```text
tgt embedding         [1,4,8]
tgt positional        [1,4,8]
相加后 y              [1,4,8]
```

位置编码告诉模型 token 在哪里，Embedding 告诉模型 token 是谁。

---

## 12. 模型到底有多少模块、多少层

在顶层，`Seq2SeqTransformer` 有七组主要部分：

1. 源 Embedding
2. 源位置编码
3. N 个 EncoderLayer
4. 目标 Embedding
5. 目标位置编码
6. N 个 DecoderLayer
7. 输出投影 Linear(D, V_tgt)

本文 `N=2`，所以堆叠结构是：

```text
EncoderLayer 0
EncoderLayer 1

DecoderLayer 0
DecoderLayer 1
```

一个 EncoderLayer 内部有：

```text
1 个 MultiHeadAttention（self-attention）
1 个 FeedForward
2 个 LayerNorm
2 条残差路径
2 个子层输出 Dropout
```

一个 DecoderLayer 内部有：

```text
1 个 masked self-attention
1 个 cross-attention
1 个 FeedForward
3 个 LayerNorm
3 条残差路径
3 个子层输出 Dropout
```

因此当 `N=2` 时，共有：

```text
MultiHeadAttention：2 个 Encoder self-attention
                  + 2 个 Decoder self-attention
                  + 2 个 Decoder cross-attention
                  = 6 个

FeedForward：2 + 2 = 4 个
LayerNorm：2×2 + 3×2 = 10 个
```

每个 MultiHeadAttention 里还有四个线性层：

```text
q_proj、k_proj、v_proj、out_proj
```

“两个头”不是复制两个完整模块，而是把投影后的 D 维特征拆成 H 份并行计算。

---

## 13. Encoder 的任务：把三个孤立词变成有上下文的 memory

进入第一个 EncoderLayer 前：

```text
x [1,3,8]
```

Encoder self-attention 使用：

```text
Q = x
K = x
V = x
```

所以叫 self-attention：询问者、被查询的索引和被汇总的内容都来自同一条源序列。

### 13.1 Q、K、V 线性投影

三个线性层分别把 x 投影成：

```text
Q [1,3,8]
K [1,3,8]
V [1,3,8]
```

Q 可以理解为“当前 token 想找什么”，K 是“每个 token 可以用什么特征被匹配”，V 是
“匹配后真正取回什么内容”。这是理解工具，不是说三者各自有固定的人类语义。

### 13.2 拆成两个头

`D=8、H=2`，所以 `Dh=4`：

```text
Q/K/V [1,3,8]
→ reshape + transpose
Q/K/V [1,2,3,4]
```

两个头都看到三个 token，但各自在四维特征子空间中学习不同关系。

### 13.3 计算注意力分数

```text
scores = Q @ Kᵀ / sqrt(Dh)
```

shape：

```text
Q                 [1,2,3,4]
Kᵀ                [1,2,4,3]
scores            [1,2,3,3]
```

最后两个 `3×3` 表示每个源位置对三个源位置的关注程度。例如 `爱` 的 Query 可以同时参考
`我`、`爱`、`你`。

除以 `sqrt(Dh)` 而不是 `sqrt(D)`，因为点积发生在每个头自己的 Dh 维空间。

### 13.4 softmax 与加权求和

Mask 后沿最后一维做 softmax：

```text
weights [1,2,3,3]
每一行权重之和 = 1
```

再乘 V：

```text
weights @ V
[1,2,3,3] @ [1,2,3,4]
→ context [1,2,3,4]
```

合并两个头并通过 out_proj：

```text
[1,2,3,4] → [1,3,8]
```

现在每个位置不再只是原始 token 向量，而是融合了整句相关信息的上下文表示。

### 13.5 残差、LayerNorm 和 FFN

当前 Encoder 使用 Post-LN：

```text
x = Norm(x + Dropout(SelfAttention(x)))
x = Norm(x + Dropout(FeedForward(x)))
```

残差让原信息可以直接通过，LayerNorm 稳定特征分布。

FFN：

```text
[1,3,8] → Linear(8,16) → ReLU → Linear(16,8) → [1,3,8]
```

Attention 负责 token 之间交流；FFN 独立加工每个 token 的特征。

经过两个 EncoderLayer 后得到：

```text
memory [1,3,8]
```

memory 仍有三个位置，分别对应源 token 的位置，但每个位置已经带有整句上下文。

---

## 14. Decoder 的任务：一边读英文前缀，一边查询中文 memory

训练时 Decoder 一次接收：

```text
tgt_input = [<sos>, i, love, you]
```

经过目标 Embedding、缩放和位置编码：

```text
y [1,4,8]
```

每个 DecoderLayer 有三步：

1. masked self-attention
2. cross-attention
3. FeedForward

### 14.1 masked self-attention：先理解已经给出的英文前缀

这里仍然：

```text
Q = K = V = y
```

拆头后：

```text
Q/K/V              [1,2,4,4]
self weights       [1,2,4,4]
self output        [1,4,8]
```

但 causal mask 会遮住未来。例如位置 1 对应 token `i`，它不能偷看后面的 `love` 和 `you`。

训练虽然一次传入了完整 `tgt_input`，信息约束却和逐步生成一致。这就是训练能够并行而不泄漏
答案的关键。

### 14.2 cross-attention：翻译真正发生联系的地方

cross-attention 中：

```text
Q   来自 Decoder 当前状态 y
K/V 来自 Encoder memory
```

本文 shape：

```text
Decoder Q          [1,2,4,4]
Encoder K/V        [1,2,3,4]
cross weights      [1,2,4,3]
cross output       [1,4,8]
```

`cross weights` 的最后两个维度是 `4×3`：

```text
4 个目标查询位置
查询
3 个源 token 位置
```

直观上，预测英文 `i` 时，Decoder 可以更关注中文 `我`；预测 `love` 时，可以更关注 `爱`；
预测 `you` 时，可以更关注 `你`。

但不要把注意力权重简单等同于人工词典对齐。模型可能分散关注、组合多个位置，不同头也可能
学习不同关系。cross-attention 提供的是可学习的信息通道。

为什么 cross-attention 输出长度是 4，而不是 3？

```text
输出位置数由 Query 数量决定。
```

这里 Q 来自目标序列，所以输出仍是 `[1,4,8]`。源长度 3 在对 V 加权求和时被消去。

### 14.3 FeedForward：加工融合后的目标表示

cross-attention 把源信息带入目标位置后，FFN 再对每个位置独立变换：

```text
[1,4,8] → [1,4,16] → [1,4,8]
```

经过两个 DecoderLayer 后：

```text
decoder_hidden [1,4,8]
```

---

## 15. 输出投影：从 8 维隐藏状态回到英文词表

Decoder hidden 还是连续向量，不能直接叫作单词。输出层执行：

```python
nn.Linear(d_model, tgt_vocab_size)
```

如果目标词表大小为 `V_tgt=7`：

```text
decoder_hidden       [1,4,8]
logits               [1,4,7]
```

每个目标位置都有 7 个分数。例如第 0 个位置：

```text
<pad>  -1.2
<sos>  -0.8
<eos>  -1.0
<unk>  -0.4
i       3.7
love    0.6
you     0.2
```

这些数值只是示意。分数最大的 token 是 `i`。

logits 不是概率，不要求和为 1。训练时 `cross_entropy` 内部会做数值稳定的 log-softmax；
推理用 argmax 时，只需比较大小，也不必先 softmax。

四个位置分别应预测：

```text
位置 0 → i
位置 1 → love
位置 2 → you
位置 3 → <eos>
```

---

## 16. Loss：模型怎样知道自己哪里错了

目标答案是：

```text
tgt_output [4,5,6,2]
shape [1,4]
```

logits 与 targets 展平：

```text
logits   [1,4,7] → [4,7]
targets  [1,4]   → [4]
```

Cross Entropy 会分别检查四个位置，正确 token 的概率越低，loss 越大。

如果 batch 有 padding，使用：

```python
ignore_index=tgt_pad_id
```

让 `<pad>` 不参与损失。否则模型会花能力学习无意义的补齐位置，accuracy 也会被大量 PAD
人为抬高或扭曲。

---

## 17. 翻译能力是怎样一点点学出来的

模型刚初始化时：

- Embedding 近似随机；
- Q/K/V 投影近似随机；
- FFN 与输出投影近似随机；
- 生成结果通常毫无意义。

一次训练更新发生：

```text
前向传播
→ 得到 logits
→ 与正确 tgt_output 比较得到 loss
→ loss.backward() 计算所有参数梯度
→ 梯度裁剪
→ optimizer.step() 更新参数
```

梯度会沿完整路径反向流动：

```text
输出投影
← Decoder FFN
← cross-attention
← Decoder self-attention
← 目标 Embedding
← Encoder memory
← Encoder FFN/self-attention
← 源 Embedding
```

经过大量中英句对和多轮更新，模型逐渐形成几类能力：

### 17.1 词与子词表示

Embedding 学到：哪些 token 在相似上下文中出现，它们应有怎样的连续表示。

### 17.2 源语言上下文理解

Encoder self-attention 学到：一个中文 token 的含义要结合周围 token 判断。

### 17.3 源目标对应关系

Decoder cross-attention 学到：为了预测某个英文 token，应从哪些中文位置提取信息。

### 17.4 目标语言规律

Decoder masked self-attention 学到：已经生成的英文前缀怎样约束下一个英文 token。

### 17.5 词表决策

输出投影学到：某种 Decoder hidden pattern 应映射到哪个目标 token。

因此翻译能力不是某一个模块单独提供的。它来自：

```text
数据中的对应关系
+
可微分的模型结构
+
loss 给出的错误信号
+
反向传播与大量参数更新
```

小数据实验达到 accuracy 1.0，可能只是记住训练句。真正的翻译能力需要大量、多样、覆盖丰富
语言现象的平行语料，并需要在未见过的验证集和测试集上评估。

---

## 18. 训练时为什么可以一次预测四个位置

训练时正确目标句已知，可以构造完整：

```text
tgt_input = [<sos>, i, love, you]
```

模型一次输出：

```text
logits [1,4,V_tgt]
```

causal mask 确保：

```text
预测 i 时只能看到 <sos>
预测 love 时只能看到 <sos>, i
预测 you 时只能看到 <sos>, i, love
预测 <eos> 时可以看到完整目标前缀
```

所以计算是并行的，信息条件仍然与推理一致。

这不意味着训练和推理完全相同。训练看到的历史 token 是正确答案；推理看到的是模型自己之前
生成的 token。如果早期生成错了，错误可能继续影响后续，这也是自回归模型的重要特点。

---

## 19. 推理阶段：模型怎样真的生成 i love you

输入中文源句后，`greedy_decode()` 首先执行一次：

```text
memory, src_mask = model.encode(src_tokens)
```

Encoder 只运行一次，因为源句不会改变。

然后初始化：

```text
generated = [<sos>]
```

### 第 1 步

```text
Decoder 输入        [<sos>]
Decoder hidden      [1,1,8]
取最后位置          [1,8]
输出投影            [1,V_tgt]
argmax              i
generated           [<sos>, i]
```

### 第 2 步

```text
Decoder 输入        [<sos>, i]
最后位置预测        love
generated           [<sos>, i, love]
```

### 第 3 步

```text
Decoder 输入        [<sos>, i, love]
最后位置预测        you
generated           [<sos>, i, love, you]
```

### 第 4 步

```text
Decoder 输入        [<sos>, i, love, you]
最后位置预测        <eos>
停止生成
```

词表 decode 会隐藏 `<sos>`、`<pad>`，遇到 `<eos>` 停止，最终得到：

```text
i love you
```

贪心生成每次选择当前最大 logits。它简单确定，但不一定得到全局概率最高的整句；更复杂方法包括
beam search。GPT 开放式生成还常用 temperature、top-k 和 top-p 采样。

---

## 20. 一张表看完全部数据变化

使用本文配置：

```text
阶段                              shape
------------------------------------------------
源 token IDs                      [1,3]
目标输入 IDs                      [1,4]
目标输出 IDs                      [1,4]

源 Embedding                      [1,3,8]
源位置编码后                       [1,3,8]
源 padding mask                   [1,1,1,3]

Encoder Q/K/V（未拆头）            [1,3,8]
Encoder Q/K/V（拆头）              [1,2,3,4]
Encoder attention weights         [1,2,3,3]
Encoder memory                    [1,3,8]

目标 Embedding                    [1,4,8]
目标位置编码后                     [1,4,8]
目标 padding mask                 [1,1,1,4]
目标 causal mask                  [1,1,4,4]
目标组合 mask                     [1,1,4,4]

Decoder self Q/K/V                [1,2,4,4]
Decoder self weights              [1,2,4,4]

Decoder cross Q                   [1,2,4,4]
Decoder cross K/V                 [1,2,3,4]
Decoder cross weights             [1,2,4,3]

Decoder hidden                    [1,4,8]
最终 logits                       [1,4,V_tgt]
训练 targets                      [1,4]
```

只要这张表能独立推导出来，Transformer 的数据骨架就已经非常清楚。

---

## 21. 各模块一句话职责表

```text
Tokenizer
把字符串切成模型处理的 token。

Vocabulary
建立 token 与整数 ID 的双向映射。

Dataset
构造 src、tgt_input、tgt_output。

Collate
对不同长度样本 padding，组成 batch。

Embedding
把离散 ID 变成可学习连续向量。

Positional Encoding
把顺序信息加入 token 表示。

Encoder self-attention
让每个源 token 汇总整条源句。

Encoder FFN
独立加工每个源位置的特征。

Decoder masked self-attention
建模已知目标前缀，同时禁止未来泄漏。

Decoder cross-attention
让每个目标位置按需读取源 memory。

Decoder FFN
独立加工融合源信息后的目标特征。

Residual
保留原信息并改善深层网络优化。

LayerNorm
稳定每个位置的特征尺度。

Output Projection
把隐藏向量映射为目标词表 logits。

Cross Entropy
衡量预测与正确 token 的差距。

Backpropagation
把错误信号传给所有可学习参数。

Greedy Decode
从 <sos> 开始逐个选择下一个 token。
```

---

## 22. 参数到底存在哪里

可学习参数包括：

- 源 Embedding 表；
- 目标 Embedding 表；
- 每层 attention 的 q/k/v/out 投影；
- 每层 FFN 的两个 Linear；
- 每个 LayerNorm 的缩放与平移；
- 最终输出投影。

正弦位置编码不是可学习参数，它通过 `register_buffer` 保存：

- 会随模型移动到 CPU/GPU；
- 会进入 state_dict；
- 不会被 optimizer 更新。

模型配置决定参数矩阵 shape，`state_dict` 保存具体参数数值。只保存权重而不保存配置，加载时
就不知道应重建多少层、多大宽度和多少个头。

---

## 23. Transformer 不是什么

为了真正理解，也要知道不该怎样描述它。

### 它不是人工词典

模型没有硬编码 `爱 = love`。对应关系来自训练数据和参数更新。

### Attention 权重不等于唯一解释

一个头的高权重可以提供观察线索，但模型决策还经过多头合并、残差、FFN 和多层堆叠。

### token ID 不是语义坐标

ID 4 和 ID 5 的数字差不代表词义差。连续语义主要存在 Embedding 和隐藏向量中。

### 训练集准确率高不等于会翻译新句子

小数据很容易被记忆。泛化必须在未参与训练的数据上验证。

### Transformer 不是一次性“理解整句后直接吐出答案”

Encoder 一次处理源句；Decoder 在推理时仍然逐 token 生成，每一步都以已有目标前缀和源 memory
为条件。

---

## 24. 常见错误应怎样定位

### 大量 `<unk>`

先查 tokenizer：构建词表和编码是否使用相同分词方式。

### Embedding 报 dtype 错误

token IDs 必须是 `torch.long`。

### `src` 或 `tgt` shape 不对

区分单样本 `[L]` 与 batch `[B,L]`；collate 后才出现 batch 维。

### 过去位置被未来目标影响

检查 causal mask 的上三角方向，以及 True 是否表示屏蔽。

### cross-attention 输出长度理解错误

输出长度由 Query 决定。Decoder Query 长度是 T，所以输出 `[B,T,D]`。

### loss 不下降

依次检查：

1. input/output 是否正确错位；
2. PAD 是否忽略；
3. 是否执行 `loss.backward()`；
4. 是否执行 `optimizer.step()`；
5. 梯度是否有限；
6. 单 batch 能否过拟合。

### 推理一直不停止

检查模型是否学到 `<eos>`，并保留 `max_new_tokens` 安全上限。

---

## 25. 代码阅读路线

建议按数据流而不是文件名随机阅读：

1. [`seq2seq/vocab.py`](../seq2seq/vocab.py)：token 与 ID
2. [`seq2seq/data.py`](../seq2seq/data.py)：三条训练 Tensor
3. [`seq2seq/masks.py`](../seq2seq/masks.py)：padding 与 causal 约束
4. [`seq2seq/position.py`](../seq2seq/position.py)：顺序信息
5. [`seq2seq/attention.py`](../seq2seq/attention.py)：Q/K/V 与多头
6. [`seq2seq/layers.py`](../seq2seq/layers.py)：EncoderLayer、DecoderLayer
7. [`seq2seq/model.py`](../seq2seq/model.py)：完整数据流
8. [`seq2seq/training.py`](../seq2seq/training.py)：loss 与参数更新
9. [`seq2seq/inference.py`](../seq2seq/inference.py)：逐 token 翻译
10. [`tests/`](../tests/)：用不变量检查理解

---

## 26. 最后，用自己的话回答这些问题

1. 为什么源端和目标端要使用两个 Embedding？
2. 为什么目标需要 `tgt_input` 和 `tgt_output` 两条错位序列？
3. 为什么 Encoder 不需要 causal mask？
4. `d_model=8、num_heads=2` 时，拆头后的单头宽度是多少？
5. 为什么 cross-attention 的 Q 来自 Decoder，而 K/V 来自 Encoder？
6. 为什么 cross-attention 输出长度等于目标长度？
7. Attention 和 FFN 分别负责哪一种信息处理？
8. 为什么 logits 不需要先 softmax 再交给 cross entropy？
9. 为什么训练能并行预测所有目标位置，推理却必须逐个生成？
10. 模型的翻译能力来自哪四个共同因素？

如果你能不看文档，完整讲出：

```text
文本
→ token
→ ID
→ Embedding + 位置
→ Encoder memory
→ Decoder self-attention
→ cross-attention
→ FFN
→ logits
→ loss 与反向传播
→ 自回归生成
```

并且能说清每一步的 shape 和职责，那么你对 Transformer 的理解已经从“知道名词”走到了
“能够沿数据流解释架构”。这正是继续学习更大模型、Decoder-only GPT、KV Cache 和高性能
推理的可靠起点。
