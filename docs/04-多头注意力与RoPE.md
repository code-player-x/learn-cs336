# Lesson 04：多头注意力与 RoPE

> Stanford CS336 面试导向学习指南 · **概念讲解 → 代码实现 → 面试考点 → 练习题**  
> 本节为**超高频面试区**：Self-Attention 全流程、缩放因子、多头机制、位置编码全家桶、**RoPE** 与 KV Cache 入门。

---

## 目录

1. [Self-Attention 完整推导](#一self-attention-完整推导)
2. [为什么要除以 sqrt(d_k)](#二为什么要除以-sqrtd_k)
3. [Multi-Head Attention](#三multi-head-attention)
4. [位置编码全家桶](#四位置编码全家桶)
5. [RoPE 旋转位置编码（重点）](#五rope-旋转位置编码重点)
6. [KV Cache 基础概念](#六kv-cache-基础概念)
7. [面试高频题（12 题详解）](#七面试高频题12-题详解)
8. [练习题](#八练习题)
9. [附录](#九附录)

---

## 一、Self-Attention 完整推导

### 1.1 动机：用「查询—键—值」做软检索

给定一层输入 $\mathbf{X} \in \mathbb{R}^{n \times d}$（$n$ 个 token，每维 $d$），我们希望对每个位置 $i$ 计算一个输出向量，该输出能聚合**全序列**的信息，且聚合权重由**内容相似度**动态决定。

为此引入三个可学习线性投影：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V
$$

其中 $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V \in \mathbb{R}^{d \times d_k}$（单头简化记号；多头时通常 $d_k = d/h$）。

**直觉**：

- **Query $\mathbf{q}_i$**：「我当前在找什么信息？」
- **Key $\mathbf{k}_j$**：「位置 $j$ 提供的内容标签是什么？」
- **Value $\mathbf{v}_j$**：「若选中位置 $j$，实际取走的信息向量。」

### 1.2 注意力分数：$\mathbf{Q}\mathbf{K}^\top$

对位置 $i$ 与 $j$，未归一化的相似度常取点积：

$$
s_{ij} = \mathbf{q}_i^\top \mathbf{k}_j
$$

矩阵形式：$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$。

### 1.3 缩放：除以 $\sqrt{d_k}$

定义 **Scaled Dot-Product Attention**：

$$
\mathbf{A} = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}
$$

其中 softmax **对最后一维（key 维）**做：每一行 $i$ 对应「位置 $i$ 对所有 $j$ 的注意力分布」。

### 1.4 加权求和得到输出

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathbf{A}\mathbf{V} = \mathbf{O} \in \mathbb{R}^{n \times d_k}
$$

再经输出投影 $\mathbf{W}_O$ 回到 $d$ 维（单头）或与多头拼接后投影。

### 1.5 完整公式（背诵版）

$$
\boxed{
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) =
\mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
}
$$

**Decoder 因果掩码**：在 softmax **之前**，对非法位置 $(i,j), j>i$ 将 logits 置为 $-\infty$，使 $\mathbf{A}_{ij}=0$。

---

## 二、为什么要除以 $\sqrt{d_k}$？

### 2.1 方差稳定直觉

设 $\mathbf{q},\mathbf{k}$ 的分量近似独立、零均值、方差 1，则 $q^\top k = \sum_{r=1}^{d_k} q_r k_r$ 的方差随 $d_k$ **线性增长**（独立项方差相加）。因此点积幅度约为 **$\sqrt{d_k}$** 量级。

若不缩放，较大的点积分数差异可能使 softmax **接近 one-hot**，其对输入分数的导数趋小，妨碍注意力中的梯度传播。分数差异很小时，注意力可能**接近均匀分布**，表示位置间区分性较弱，但**不能仅据此推断梯度弱或消失**；梯度大小还取决于序列长度、温度缩放与上游梯度。**除以 $\sqrt{d_k}$** 使点积方差回到 $O(1)$，降低因尺度过大而饱和的风险。

### 2.2 简化的数学推导（面试够用）

假设 $q_r, k_r$ 独立，$\mathbb{E}[q_r]=\mathbb{E}[k_r]=0$，$\mathrm{Var}(q_r)=\mathrm{Var}(k_r)=1$。

$$
\mathbb{E}[q_r k_r] = 0,\quad \mathrm{Var}(q_r k_r) = \mathbb{E}[q_r^2]\mathbb{E}[k_r^2] = 1
$$

和 $S = \sum_{r=1}^{d_k} q_r k_r$ 的方差为 $d_k$，标准差 $\sqrt{d_k}$。故将 $S$ 除以 $\sqrt{d_k}$，使 **标准化** 到 $O(1)$ 波动。

### 2.3 与「温度」的关系

有时把 $\alpha$ 写作温度：$\mathrm{softmax}(\mathbf{S}/T)$。$\sqrt{d_k}$ 相当于**隐式**设定温度，使训练稳定。

---

## 三、Multi-Head Attention

### 3.1 为什么要多头？

单头注意力只学习**一种**相似度模式；多头将 $d$ 切为 $h$ 份，每份在独立子空间做 Attention，再拼接/投影，能同时捕获：

- 句法局部模式  
- 语义依赖  
- 指代、共指等  

**一句话**：**多子空间并行，模式更丰富**。

### 3.2 头数与维度关系

常见设定：$d_k = d_h = d/h$，每头维度 $d_h$。总参数量：$4 d^2$（QKV O）或按实现拆分为多头低秩形式。

### 3.3 拼接与输出投影

$$
\mathrm{MultiHead}(\mathbf{X}) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\mathbf{W}_O
$$

其中 $\mathbf{W}_O \in \mathbb{R}^{d \times d}$。

### 3.4 PyTorch 风格代码（单头示意 + 多头拼接）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # x: (B, T, d)
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,h,T,dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B,h,T,dh)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(y)
```

---

## 四、位置编码全家桶

### 4.1 为什么 Transformer 需要位置编码？

**无位置编码、也无固定因果掩码**的 Self-Attention 对输入置换等变：置换矩阵为 $P$ 时，$A'=PAP^\top$、$O'=PO$，不是矩阵元素原地不变。它不能仅凭内容分辨顺序，因此通常需加入位置信息；因果掩码本身也引入顺序约束。

### 4.2 正弦绝对位置编码（Sinusoidal）

原始 Transformer 使用：

$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d}),\quad
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d})
$$

**特点**：固定、可外推性讨论多；现代 LLM 较少用，但面试常考。

### 4.3 可学习绝对位置编码

直接学习 $\mathbf{E}_{pos} \in \mathbb{R}^{n_{\max} \times d}$。简单、长外推弱。

### 4.4 相对位置编码（概念）

相对编码强调 $i-j$ 而非绝对 $i$，如 T5、Transformer-XL 等。利于归纳长度外推相关性质。

### 4.5 位置编码演进对比表

| 类型 | 代表 | 优点 | 缺点 |
|------|------|------|------|
| Sinusoidal | 原始 Transformer | 无需学习参数 | 外推仍有限 |
| 可学习绝对 | 早期 BERT/GPT | 实现简单 | 超长序列弱 |
| RoPE | LLaMA 等 | 相对性、实现优雅 | 需理解旋转 |
| ALiBi | 部分模型 | 外推讨论 | 与架构耦合 |

---

## 五、RoPE 旋转位置编码（重点）

### 5.1 核心思想

**RoPE（Rotary Position Embedding）** 在二维子空间对 $q,k$ 施加位置相关旋转。对固定内容向量，有 $(R_mq)^\top(R_nk)=q^\top R_{n-m}k$，位置因子只依赖相对位移；内积仍依赖内容，不是只由位置决定。

### 5.2 复数与旋转（推导骨架）

将两维 $(q_{2i},q_{2i+1})$ 视作复数 $q_i$，乘以 $e^{im\theta_i}$ 等价于旋转 **$m\theta_i$** 角，其中 $\theta_i$ 是该二维子空间的频率。

对位置 $m$ 的旋转：

$$
f(q, m) = R_m q,\quad f(k, n) = R_n k
$$

适当构造 $R_m$ 使得 $q_m^\top k_n$ 依赖 **$m-n$**。

### 5.3 为什么 RoPE 能表达相对位置？

点积在旋转下保持某种「配对」结构：当 $q,k$ 同受旋转作用时，内积可表示为 $\sum g_i(m-n)$ 形式（示意），从而**自然编码相对位移**。

### 5.4 长序列外推

训练时见长度 $L_{\mathrm{train}}$，推理更长时，RoPE 的 **基频** $\theta_i$ 与插值（如 NTK、Linear RoPE scaling）是常见工程技巧；面试可答「与旋转基、位置插值有关，详见推理优化课」。

### 5.5 代码实现（简化版，与主流实现一致的思想）

```python
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim > 0 and dim % 2 == 0
    # dim: head dimension (must be even)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{i m theta}
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # xq/xk: (B, T, n_heads, d_head)，Q/K 头数可以不同
    def rotate(x):
        B, T, H, D = x.shape
        assert D % 2 == 0 and freqs_cis.shape == (T, D // 2)
        z = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2).contiguous())
        freqs = freqs_cis.to(x.device).view(1, T, 1, D // 2)
        return torch.view_as_real(z * freqs).flatten(-2).to(x.dtype)
    return rotate(xq), rotate(xk)
```

> **面试**：能口述「对 $q,k$ 成对二维旋转 + 点积体现相对位置」即可；手写代码常考 `triu` mask 而非 RoPE 内核。

---

## 六、KV Cache 基础概念

**问题**：自回归生成第 $t$ 步时，前 $1..t-1$ 的 Key/Value 与第 $t$ 步的 Q 计算注意力时，**历史 K/V 可复用**，不必每步重算。

**KV Cache**：缓存每层、每头已算好的 **K、V** 张量，新 token 只追加当前 K/V。

**显存**：MHA 下缓存元素数约为 $2 \times L \times B \times h \times T \times d_h$；换算为字节还需乘每元素字节数 $b_{\text{elem}}$（FP16/BF16 为 2），即 $2 L B h T d_h b_{\text{elem}}$。它与序列长度 $T$ 线性增长；GQA/MQA 应改用 KV 头数（Lesson 20 详讲）。

---

## 七、面试高频题（12 题详解）

### Q1：Self-Attention 的计算步骤？

**答**：① 线性投影得 Q、K、V；② 算 $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k}$；③（Decoder）加因果掩码；④ softmax 得 $\mathbf{A}$；⑤ $\mathbf{O}=\mathbf{A}\mathbf{V}$；⑥ 多头拼接/输出投影。

### Q2：为什么除以 $\sqrt{d_k}$？（要能推导）

**答**：点积维度 $d_k$ 增大时方差线性增大，幅度约 $\sqrt{d_k}$；除以 $\sqrt{d_k}$ 使 softmax 输入稳定在 $O(1)$，避免饱和与梯度问题。见第二节推导。

### Q3：多头注意力的优势？

**答**：多子空间学习不同关系模式，表达力更强；最后拼接融合。

### Q4：头数和维度的关系？

**答**：常设 $d = h \times d_h$，每头维度 $d_h = d/h$；在标准 MHA、固定 $d$ 且 $d_h=d/h$ 时，Q/K/V/O 的投影权重总数为 $4d^2$，不会仅因头数 $h$ 改变而增减（忽略 bias）。

### Q5：RoPE 的核心思想是什么？

**答**：在 Q/K 上施加**位置相关的旋转**，使注意力内积体现**相对位置**；利于长度外推讨论。

### Q6：RoPE vs 绝对位置编码 vs 相对位置编码？

**答**：RoPE 通过旋转实现**相对性**与实现优雅；绝对编码直接加向量；相对编码显式建模 $i-j$（多种形式）。现代 LLM 常用 RoPE。

### Q7：RoPE 如何实现长序列外推？

**答**：训练长度有限，推理更长时需 **位置插值/基频调整**（如 NTK、Linear scaling）等缓解分布偏移；非简单公式一句，需结合工程。

### Q8：Self-Attention 的时间与空间复杂度？

**答**：标准 MHA 中，单头注意力混合为 $O(n^2 d_k)$，全部头合计为 $O(hn^2d_k)=O(n^2d)$；计入 Q/K/V/O 投影后的总时间为 $O(n^2d+nd^2)$。若显式物化全部头的注意力矩阵，空间为 $O(hn^2)$（忽略 batch）；FlashAttention 可避免完整物化。

### Q9：为什么用 Q、K、V 三个矩阵而不是一个？

**答**：**角色分离**：Q 表查询、K 表索引、V 表内容；不同投影给优化器**不同自由度**，比单一投影更灵活（经典解释）。

### Q10：Attention 矩阵的物理含义是什么？

**答**：行 $i$ 表示位置 $i$ 对各个位置 $j$ 的**依赖权重**；可理解为动态路由/软对齐。

### Q11：MQA 和 GQA 是什么？（简要）

**答**：**MQA**：所有头共享 K/V；**GQA**：头分组共享 K/V。减少 KV Cache 与带宽（Lesson 05 详讲）。

### Q12：注意力分数的 softmax 在哪个维度做？

**答**：对 **key 维**（最后一维，即对每个固定 query 位置 $i$，对 $j$ 归一化）。

---

## 八、练习题

1. 手算 $n=3,d_k=2$ 的 $\mathbf{Q}\mathbf{K}^\top$ 与因果 mask 后 softmax（用小型矩阵）。  
2. 解释为何 $\mathbf{A}\mathbf{V}$ 是「加权求和」。  
3. 若 $h=8,d=512$，每头维度多少？  
4. RoPE 与「加性位置向量」本质差异？  
5. 推导：若 $q,k$ 方差为 1，点积方差为何 $\approx d_k$？  
6. 实现：用 `torch.triu` 构造因果 mask。  
7. 对比：FlashAttention 不改变数学输出，只改什么？  
8. 思考：KV Cache 为何不缓存 Q？  

---

## 九、附录

### 附录 A：注意力与 softmax 数值稳定性

```python
att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
att_max = att.max(dim=-1, keepdim=True).values
att = torch.softmax(att - att_max, dim=-1)  # 等价，更稳
```

### 附录 B：扩展阅读关键词

1. Attention 名字来自认知科学类比。  
2. Scaled dot-product 是最常见实现。  
3. Additive attention（Bahdanau）不同公式。  
4. 点积比加法注意力更省算。  
5. $n^2$ 是长上下文瓶颈。  
6. 稀疏注意力可减少相对于稠密 $n^2$ 的注意力连接数。
7. Linear Attention 近似 softmax。  
8. Performer 用随机特征。  
9. FlashAttention 分块 softmax。  
10. 内存层次影响实际速度。  
11. 头数不是越多越好。  
12. 头数需整除 $d$。  
13. GQA 折中 MHA 与 MQA。  
14. MQA 推理快。  
15. RoPE 来自论文 RoFormer。  
16. LLaMA 2 使用 RoPE。  
17. GPT-3 使用可学习位置编码（版本依实现）。  
18. GPT-4 细节未全公开。  
19. 位置编码与长度外推是活跃研究。  
20. YaRN 等插值方法。  
21. NTK-aware scaling。  
22. 动态 NTK。  
23. 注意力可视化工具。  
24. head 分工可解释性研究。  
25. 归纳偏置弱于 CNN。  
26. 数据规模补偿归纳偏置。  
27. 自注意力是集合函数。  
28. Set Transformer 相关。  
29. Perceiver IO 相关。  
30. 交叉注意力用于条件生成。  
31. Encoder-Decoder 用 Cross-Attn。  
32. Q 来自 decoder，K/V 来自 encoder。  
33. 因果掩码只在 decoder self-attn。  
34. 双向注意力无因果掩码。  
35. BERT 用双向注意力。  
36. MLM 训练。  
37. 自回归用交叉熵。  
38. Teacher forcing 训练。  
39. 推理时自回归。  
40. KV Cache 加速自回归。  
41. 批推理变长 padding。  
42. attention mask 区分 pad。  
43. 绝对位置 + 相对位置可混合。  
44. 旋转矩阵正交。  
45. 复数乘法对应旋转。  
46. 偶数维 RoPE。  
47. half 精度与 RoPE 兼容需注意。  
48. 推理量化影响 attention。  
49. INT8 KV 缓存。  
50. paged attention vLLM。  
51. 连续批处理。  
52. 前缀共享 KV。  
53. 多轮对话前缀复用。  
54. 系统提示缓存。  
55. 长文分块 attention。  
56. 局部窗口 attention。  
57. dilated attention。  
58. Longformer 稀疏模式。  
59. BigBird 随机+窗口。  
60. 理论表达力与深度宽度。  
61. Universal Transformer 循环。  
62. 深度与层归一化。  
63. Pre-LN 稳定。  
64. Post-LN 原始。  
65. 残差连接梯度。  
66. 初始化影响注意力尺度。  
67. 缩放与 LayerScale。  
68. Dropout 在 attention 概率上。  
69. Stochastic depth。  
70. 注意力 dropout 与推理。  
71. 导出 ONNX 注意力节点。  
72. TensorRT 融合 attention。  
73. CUDA kernel 手写 attention。  
74. Triton 教程。  
75. 反向传播过 attention。  
76. 梯度检查点过层。  
77. 激活重计算。  
78. 序列并行切分 n。  
79. Ring attention 分布式。  
80. 文献《Attention is All You Need》。  
81. 论文页数不多影响大。  
82. 引用量极高。  
83. 后续 BERT GPT T5。  
84. 视觉 ViT。  
85. 语音 Conformer。  
86. 图 Graph Transformer。  
87. 时间序列 Transformer。  
88. TabTransformer。  
89. 推荐 DLRM 与 attention。  
90. 多模态 cross-attention。  
91. CLIP 双塔。  
92. Flamingo 交叉。  
93. Perceiver Resampler。  
94. 对比学习用 attention pooling。  
95. Set Transformer PMA。  
96. 面试常问复杂度。  
97. 常问掩码形状。  
98. 常问 softmax 维。  
99. 常问 RoPE 动机。  
100. 下一课 RMSNorm/SwiGLU/GQA。  

---

### 附录 C：学习内容自检

| 要求 | 状态 |
|------|------|
| Q、K、V 含义与计算 | 第一节 |
| 注意力分数 QK^T | 第一节 |
| 除以 sqrt(d_k) 数学推导 | 第二节 |
| Softmax 与加权求和 | 第一节 |
| 完整公式 | 第一节 |
| 多头原因、头数维度、代码 | 第三节 |
| 位置编码种类 | 第四节 |
| RoPE 核心、复数、相对位置、外推、代码 | 第五节 |
| KV Cache 入门 | 第六节 |
| 面试题 ≥12 | 第七节 |

---

---

## 附录 D：Self-Attention 白板推导长版（概念 → 代码 → 面试）

### D.1 从标量到矩阵：一行一行看

设 batch 忽略，序列矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$。第 $i$ 行 $\mathbf{x}_i^\top$ 是 token $i$ 的向量。

$$
\mathbf{q}_i = \mathbf{W}_Q^\top \mathbf{x}_i,\quad
\mathbf{k}_j = \mathbf{W}_K^\top \mathbf{x}_j,\quad
\mathbf{v}_j = \mathbf{W}_V^\top \mathbf{x}_j
$$

注意力权重：

$$
\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{j'}\exp(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_k})}
$$

输出：

$$
\mathbf{o}_i = \sum_j \alpha_{ij} \mathbf{v}_j
$$

### D.2 矩阵形式一次性写完

$$
\mathbf{O} = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

**softmax 行方向**：对固定的 $i$，对所有 $j$ 归一化。

### D.3 参考实现：显式循环版（仅教学，勿用于生产）

```python
import math
import torch

def attention_naive(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Q,K,V: (n, dk)
    dk = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
    attn = torch.softmax(scores, dim=-1)
    return attn @ V
```

### D.4 面试题：softmax 在最后一维的代码验证

```python
import torch
x = torch.randn(2, 5, 7)
y = torch.softmax(x, dim=-1)
assert torch.allclose(y.sum(dim=-1), torch.ones(2, 5))
```

---

## 附录 E：更多 RoPE 细节（面试追问）

### E.1 旋转矩阵为何是「成对」二维？

偶数维 $d_h$ 拆成 $d_h/2$ 个二维平面，每平面独立旋转，频率 $\theta_i$ 随 $i$ 递减，兼顾短程与长程模式。

### E.2 与相对位置编码的异同

RoPE 在**内积里**编码相对位置；经典相对偏置在 logits 上加 $b_{i-j}$。二者目标相近，形式不同。

### E.3 外推为何困难？

训练分布上位置 $m$ 有界；推理超过时旋转角度进入**未充分训练**区域，注意力模式偏移；故需插值或微调。

---

## 附录 F：扩展练习题 50 道（简答提示）

1. $\mathbf{A}\mathbf{V}$ 每行是否是凸组合？→ 是（非负且行和为 1）。  
2. 因果 mask 后每行是否仍和为 1？→ 在可见集合上归一化，仍和为 1。  
3. 自注意力是否对称？→ $\mathbf{A}$ 一般不对称。  
4. 双向注意力矩阵可否对称？→ 一般不保证。  
5. 点积 vs 欧氏距离？→ 点积等价于余弦相似当范数固定。  
6. 缩放是否等价于 LayerNorm？→ 不等价，作用在不同位置。  
7. 多头拼接后维数？→ $h \cdot d_h = d$。  
8. $\mathbf{W}_O$ 作用？→ 混合头信息并映射回 $d$。  
9. bias 在 QKV？→ 可选，LLaMA 常无 bias。  
10. 位置信息进 V 吗？→ RoPE 在 QK；若加性位置可进输入 embedding。  
11. ALiBi 放哪？→ attention logits bias。  
12. 长度 $n=1$ 时注意力？→ 退化为自身权重 1。  
13. 全部位置被屏蔽会怎样？→ 常导致 softmax 的 NaN；每个有效 query 至少需一个可见 key。加性 mask 全为 0 表示不屏蔽，不能与布尔 mask 混淆。
14. 温度 $T>1$？→ 分布更平。  
15. $T<1$？→ 更尖。  
16. Gumbel-softmax？→ 可微离散（了解）。  
17. 注意力作为核？→ 有研究与 kernel 联系。  
18. Nyströmformer？→ 近似（了解）。  
19. Performer 复杂度？→ 线性近似（了解）。  
20. 为什么工业界仍多用标准 attention？→ 硬件成熟、稳定。  
21. FlashAttention 版本？→ 1/2/3（了解）。  
22. 显存 $n^2$ 来自？→ 存 logits 或 softmax 中间。  
23. checkpoint 不存？→ 重算。  
24. 逆注意力？→ 非主流。  
25. 局部窗口限制 $j$ 范围？→ 降复杂度。  
26. 膨胀窗口？→ 扩大感受野。  
27. 分层注意力？→ 多阶段（了解）。  
28. 与图注意力 GAT？→ 类似加权邻居。  
29. 与胶囊网络？→ 不同机制。  
30. Transformer 深度 $L$ 典型？→ 几十层。  
31. 宽度 $d$ 典型？→ 几百到几万。  
32. 头数典型？→ 32、40 等。  
33. GQA 分组数？→ 如 8 query 组 1 KV（示例）。  
34. MQA 头数？→ 多 query 共享 K/V。  
35. KV 重复广播？→ 实现细节。  
36. 旋转用复数还是实数矩阵？→ 等价实现皆可。  
37. `torch.view_as_complex`？→ 工程常用。  
38. half 精度旋转？→ 需注意数值。  
39. 静态图 batch？→ 变长 mask。  
40. ONNX export attention？→ 支持因版本而异。  
41. 量化 QAT？→ 量化感知训练；训练后量化称为 PTQ。
42. 注意力蒸馏？→ 小模型学大模型（了解）。  
43. 稀疏专家 MoE？→ 不同路由（了解）。  
44. 对比学习 SimCLR？→ 不用 LM attention。  
45. BERT attention 可视化？→ 可看层与头。  
46. 恶意触发高注意力？→ 安全研究。  
47. 长上下文法律/RAG？→ 应用。  
48. 注意力与检索？→ 类比。  
49. 注意力凸组合的结论适用条件？→ softmax 后、未施加 attention dropout 的权重。
50. 掌握 D.1–D.4 可应付大部分一面。  

---

## 附录 G：数学符号表

| 符号 | 含义 |
|------|------|
| $n$ | 序列长度 |
| $d$ | 模型维 |
| $h$ | 头数 |
| $d_h$ | 每头维 |
| $\mathbf{A}$ | 注意力权重 |

---

## 附录 H：与 Lesson 03、05 的边界

- Lesson 03：整体 Decoder-only 架构。  
- Lesson 04：Attention + 位置编码细节。  
- Lesson 05：RMSNorm、SwiGLU、GQA 替换 LN/FFN/MHA。  

---

## 附录 I：英文面试 30 秒模板

"Scaled dot-product attention computes QKᵀ over sqrt(d_k), softmax over keys, times V. Multi-head splits d into h subspaces. RoPE applies rotations to Q and K so inner products encode relative positions. Complexity is O(n²d) for standard attention; KV cache stores past keys and values for autoregressive decoding."

---

## 附录 K：「概念讲解 → 代码实现 → 面试考点 → 练习题」四段式总复习

### K.1 概念讲解（5 分钟口述稿）

Self-Attention 用三个投影 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 从同一层输入得到查询、键与值。相似度用点积，为控制方差在 $d_k$ 增大时不爆炸，除以 $\sqrt{d_k}$，再对 key 维 softmax 得到权重，对 $\mathbf{V}$ 加权求和。Decoder 用因果掩码屏蔽未来位置。多头将维度切分，子空间并行，再输出投影融合。RoPE 在 $\mathbf{Q},\mathbf{K}$ 上施加与位置相关的旋转，使内积编码相对位置。KV Cache 缓存历史 K/V 以加速自回归。

### K.2 代码实现（最小可运行骨架）

见第三节 `MultiHeadSelfAttention` 与附录 D `attention_naive`。面试手写推荐：**`att = (q @ k.T) / sqrt(dh)` + `triu` mask + `softmax(dim=-1)` + `att @ v`**。

### K.3 面试考点速记卡片

| 考点 | 关键词 |
|------|--------|
| 缩放 | 独立、零均值、单位方差假设下，未缩放点积方差为 $d_k$，标准差为 $\sqrt{d_k}$；除以 $\sqrt{d_k}$ 后方差为 1 |
| softmax 维 | 最后一维（key） |
| 因果 | $j>i$ 为 $-\infty$ |
| 多头 | $d/h$ |
| RoPE | 旋转、相对位置 |
| 复杂度 | $O(n^2 d)$ |
| KV Cache | 存 K/V，不重复算历史 |

### K.4 练习题加量（10 道）

1. 证明 softmax 行和为 1。  
2. 若 K=V，Attention 退化成什么形式？（讨论）  
3. 若 Q 为常数，权重如何？  
4. 写出 $\mathbf{A}\mathbf{1}$ 的含义。  
5. 因果注意力第 $t$ 行非零列范围？  
6. RoPE 是否改变 $\|q\|$？（旋转保范）  
7. 为何旋转保范对稳定有帮助？  
8. 比较 $\mathbf{O}\mathbf{1}$ 与 $\mathbf{V}$ 行均值关系。  
9. 多头拼接后为何还要 $\mathbf{W}_O$？  
10. 解释「注意力是动态路由」。

### K.6 结语

当你能在白板上 **同时写出** Attention 公式、因果 mask 示意图与 RoPE 的一句话动机，本节目标即已达成。

**复习建议**：先掌握主干公式与张量形状，再按需阅读附录。

---

## 导航

| 上一课 | 下一课 |
|--------|--------|
| [Lesson 03 - Transformer 架构详解](03-Transformer架构详解.md) | [Lesson 05 - RMSNorm / SwiGLU / GQA](05-RMSNorm-SwiGLU-GQA.md) |

[返回课程总览](00-课程总览与学习路线.md)

**文档结束**

---
