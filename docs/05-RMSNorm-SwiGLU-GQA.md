# Lesson 05：RMSNorm、SwiGLU 与 GQA

> Stanford CS336 面试导向学习指南 · **概念讲解 → 代码实现 → 面试考点 → 练习题**  
> 本节聚焦 LLaMA 系常见的 RMSNorm、SwiGLU 与 RoPE，以及较新模型采用的 GQA。并非所有 LLaMA 版本/规模都使用 GQA，例如初代采用 MHA。

---

## 目录

1. [为什么现代 LLM 替换原始组件](#一为什么现代-llm-替换原始组件)
2. [RMSNorm](#二rmsnorm)
3. [SwiGLU 激活与 FFN](#三swiglu-激活与-ffn)
4. [GQA：分组查询注意力](#四gqagrouped-query-attention)
5. [现代 LLM「四件套」总结表](#五现代-llm四件套总结表)
6. [面试高频题（10 题详解）](#六面试高频题10-题详解)
7. [练习题](#七练习题)
8. [附录](#八附录)

---

## 一、为什么现代 LLM 替换原始组件？

**原始 Transformer（Vaswani 2017）** 使用：

- **LayerNorm** + 残差  
- **Post-LN**（论文图示）与后续实现变体  
- **ReLU FFN** 或后续常用 **GELU**  
- **Multi-Head Attention（MHA）**：每头独立 K/V  

**规模化训练**后发现：

1. **LayerNorm** 的均值分支与偏置在部分设定下可简化，**RMSNorm** 更省算且稳定足够好。  
2. **FFN** 用 **门控 GLU 变体（SwiGLU）** 提升效果；代价是参数与算力需重新配比（常调整中间宽）。  
3. **推理**时 **KV Cache** 随层数与序列长度线性增长，**GQA/MQA** 通过共享 K/V 降显存与带宽。

**一句话**：在**效果、训练稳定性、推理效率**三角中寻找更优折中，而非死守 2017 论文的每一处细节。

---

## 二、RMSNorm

### 2.1 LayerNorm 回顾

对 $\mathbf{x} \in \mathbb{R}^d$（单 token），LayerNorm：

$$
\mathrm{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu = \frac{1}{d}\sum_i x_i$，$\sigma^2 = \frac{1}{d}\sum_i (x_i-\mu)^2$；这里采用 [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 的写法，$\epsilon$ 加在方差上、位于平方根内。

### 2.2 RMSNorm 公式

**RMSNorm（Root Mean Square Layer Normalization）** 去掉**中心化**（不减均值），通常也**去掉 $\beta$**，只保留缩放：

$$
\mathrm{RMSNorm}(\mathbf{x}) = \alpha \odot \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})},\quad
\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
$$

$\alpha \in \mathbb{R}^d$ 为可学习缩放（类比 $\gamma$）。

### 2.3 与 LayerNorm 对比

| 项目 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减均值 | 是 | 否 |
| 归一化分母 | 标准差（含稳定项） | 均方根（含稳定项） |
| 偏置 $\beta$ | 常有 | 常无 |
| 计算量 | 略高 | 略低 |

### 2.4 为什么 RMSNorm「足够好」？

经验上，深层网络中**尺度稳定**是关键；RMS 已能抑制向量范数爆炸/消失；**去均值**带来的额外归纳偏置在部分设定下收益有限，反而增加计算。

### 2.5 Pre-Norm vs Post-Norm（再述）

- **Pre-Norm**：$\mathbf{x} \leftarrow \mathbf{x} + \mathrm{Sublayer}(\mathrm{Norm}(\mathbf{x}))$  
- **Post-Norm**：$\mathbf{x} \leftarrow \mathrm{Norm}(\mathbf{x} + \mathrm{Sublayer}(\mathbf{x}))$  

现代大模型 **Pre-Norm + RMSNorm** 极常见，训练更稳、易加深。

### 2.6 PyTorch 风格实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        dtype = x.dtype
        x_float = x.float()  # 半精度输入用 FP32 计算平方和
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_float * inv_rms) * self.weight.float()).to(dtype)
```

---

## 三、SwiGLU 激活与 FFN

### 3.1 从 ReLU 到 GELU 到 SwiGLU

- **ReLU**：$\max(0,x)$，简单但不平滑。  
- **GELU**：平滑，Transformer 常用。  
- **GLU（Gated Linear Unit）**：把一路当门控。

### 3.2 SwiGLU 公式

**Swish** 激活：$\sigma(x) = x \cdot \mathrm{sigmoid}(\beta x)$，常取 $\beta=1$。

**SwiGLU** FFN 常写作：

$$
\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = \big(\mathrm{Swish}(\mathbf{x}\mathbf{W}_1) \odot \mathbf{x}\mathbf{V}\big)\mathbf{W}_2
$$

其中 $\odot$ 为逐元素乘，$\mathbf{W}_1,\mathbf{V}$ 将 $\mathbf{x}$ 映到中间维（常取 $2/3$ 或按实现调整相对 $4d$ 的配比），$\mathbf{W}_2$ 映回 $d$。

**直觉**：门控 $\sigma(\mathbf{x}\mathbf{W}_1)$ 控制 $\mathbf{x}\mathbf{V}$ 哪些通道通过，**表达能力**强于单路 MLP。

### 3.3 对隐层维度的影响（$8d/3$ 梗）

若经典 FFN 用两层 $d \to 4d \to d$，参数量约 $2 \cdot 4d^2 = 8d^2$。

SwiGLU 有 **三路线性**（$\mathbf{W}_1,\mathbf{V},\mathbf{W}_2$），若仍将「总参数」控制在相近量级，常把中间维从 $4d$ 调到约 **$\frac{8}{3}d$**，使总参数量级与 $8d^2$ 可比（**面试说法**：「为补偿三矩阵，缩中间宽」；精确常数依实现）。

### 3.4 代码实现（示意）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.v = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v(x))
```

`F.silu` 即 Swish（$\beta=1$）。

---

## 四、GQA（Grouped Query Attention）

### 4.1 MHA → MQA → GQA 演进

- **MHA（Multi-Head Attention）**：每头有独立 **Q、K、V**。  
- **MQA（Multi-Query Attention）**：**所有头共享一组 K、V**；推理 KV Cache 最小，但表达力可能下降。  
- **GQA（Grouped Query Attention）**：头分为若干组，**组内共享 K/V**，介于 MHA 与 MQA 之间。

### 4.2 为什么 GQA 成为「实用默认」？

在 **长上下文推理** 下，KV Cache 与内存带宽常是瓶颈；**减少 K/V 头数**直接降低缓存大小与读取量；GQA 在 **精度与效率** 间折中优于极端 MQA。

### 4.3 KV Cache 显存直觉

每层缓存 $\mathbf{K},\mathbf{V}$，形状 roughly $(B, n_{\mathrm{kv}}, T, d_h)$。  
从 MHA 到 GQA：$n_{\mathrm{kv}} = h$ 降为 $n_{\mathrm{groups}}$ 或等价更小的 KV 头数 → **线性**减少 KV 张量大小。

### 4.4 代码骨架（概念）

```python
import torch
import torch.nn as nn

class GQAProjection(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads > 0 and n_kv_heads > 0
        assert d_model % n_q_heads == 0 and n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, (d_model // n_q_heads) * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, (d_model // n_q_heads) * n_kv_heads, bias=False)

    def forward(self, x: torch.Tensor):
        # 返回 q, k, v 后需在 head 维 repeat_interleave k/v 以对齐 q 头数
        ...
```

> **面试**：能口述「K/V 头少，Q 头多，K/V 在组内广播/repeat」即可。

---

## 五、现代 LLM「四件套」总结表

| 组件 | 经典 Transformer | 现代 LLM（如 LLaMA） |
|------|------------------|----------------------|
| 归一化 | LayerNorm | **RMSNorm** |
| 位置编码 | Sinusoidal / 可学习 | **RoPE**（Lesson 04） |
| FFN | ReLU/GELU MLP | **SwiGLU** |
| 注意力 | MHA | **GQA**（或 MQA） |

另：**Pre-Norm**、**无 bias**（依实现）、**权重共享** 等亦常见。

---

## 六、面试高频题（10 题详解）

### Q1：RMSNorm 和 LayerNorm 的区别？

**答**：RMSNorm **不减均值**，常无偏置；用 RMS 缩放；计算更省；实践中对大规模 LM 效果与稳定性表现良好。

### Q2：Pre-Norm 为什么比 Post-Norm 更稳定？

**答**：Pre-Norm 让归一化进入子层输入，**梯度路径更平滑**，深层更易优化；Post-Norm 在残差后归一，训练深层时更敏感。

### Q3：SwiGLU 的优势是什么？

**答**：**门控**机制提升非线性表达能力；Swish 平滑；在同等参数预算下常优于单路 FFN（经验）。

### Q4：GQA 的核心思想和优势？

**答**：**分组共享 K/V**，减少 KV 头数；降低 **KV Cache** 与内存带宽压力；相对 MQA 保留更多表达能力。

### Q5：MHA / MQA / GQA 的参数量与 KV 对比？

**答**：固定 $d$、$h_q$ 且 $d_h=d/h_q$ 时，Q/O 共 $2d^2$ 个权重，K/V 共 $2d h_{kv}d_h$；MHA 取 $h_{kv}=h_q$，总计 $4d^2$，GQA/MQA 减少 K/V 参数。KV Cache 与 $h_{kv}$ 成正比（同 $d_h,T,L$ 下）。

### Q6：为什么 LLaMA 选择这些组件？

**答**：RMSNorm+Pre-Norm 用于稳定优化，SwiGLU 提供门控非线性，RoPE 编码位置；部分较新 LLaMA 型号采用 GQA 以减少 KV 缓存。不能把这些组件或权重共享归为所有 LLaMA 型号一致的配置。

### Q7：RMSNorm 的计算复杂度？

**答**：相对 LN **略低**（少均值）；主项仍是 $O(d)$ 每 token；相对整体 $O(n^2 d)$ attention 常可忽略。

### Q8：SwiGLU 对 FFN 隐层维度有什么影响？

**答**：三矩阵结构下为控制总参数，常把中间维从经典 $4d$ 调整为约 **$8d/3$** 量级（经验值，依实现）。

### Q9：GQA 如何减少 KV Cache 显存？

**答**：缓存的 K/V **头数减少**；推理每步读取的 KV 体积下降，带宽压力下降。

### Q10：现代 LLM 还有哪些改进？（Tie Embedding 等）

**答**：**Tied input/output embeddings**；**无 bias**；**RoPE**；有时 **MQA**；**滑动窗口/稀疏注意力**（部分模型）；量化与 KV cache 压缩（推理侧）。

---

## 七、练习题

1. 手算 $\mathbf{x}=(3,4)$ 的 RMS（加 $\epsilon=0$）与 RMSNorm（$\alpha=(1,1)$）。  
2. 对比 LN 与 RMSNorm 的 Python 行数差异。  
3. 为什么 SwiGLU 用 `silu` 而非 `relu`？  
4. 若 $n_q=32, n_{kv}=8$，每组重复几次 K/V？  
5. 解释「门控」与 LSTM 门控异同（口头）。  
6. 为什么推理比训练更在意 KV？  
7. 写一行：RMS 的 `torch` 实现。  
8. 若去掉 RMSNorm 的 `weight` 会怎样？  

---

## 八、附录

### 附录 A：RMSNorm 数值示例

```python
import torch
from typing import Optional

def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_float * rms
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)
```

### 附录 B：LayerNorm 对照实现（复习）

```python
import torch

def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta
```

---

---

## 附录 C：概念讲解 → 代码实现 → 面试考点 → 练习题（四段强化）

### C.1 RMSNorm 概念深化

RMSNorm 保留 **按特征维缩放** 的核心作用，使每 token 向量范数稳定；去掉减均值等价于假设已存在**其他机制**（如残差、权重初始化）处理均值漂移，或在大规模训练中经验上不必要。

### C.2 SwiGLU 概念深化

忽略偏置时，GLU 为 $\mathrm{GLU}(x)=(xW)\odot\mathrm{sigmoid}(xV)$：两路线性投影分别提供内容与门控。SwiGLU 将 sigmoid 门替换为 Swish/SiLU，不必通过把输入张量直接切成两半实现。

### C.3 GQA 概念深化

Query 头保持 **细粒度查询模式**；KV 头在组内共享，使 **Key/Value 子空间** 更粗；Attention 仍对每组内 **repeat** 后的 K/V 做点积，数学上等价于「少套 K/V，多套 Q」。

### C.4 代码：带 repeat 的 GQA 注意力（示意）

```python
import torch
import torch.nn.functional as F
import math

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """x: (B, n_kv, T, dh) -> (B, n_kv*n_rep, T, dh) 通过 repeat interleave"""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)

def gqa_attention(q, k, v, n_rep: int, causal: bool = False):
    # q: (B, n_q, T, dh); k,v: (B, n_kv, T, dh)
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        assert q.size(-2) == k.size(-2)  # 全序列 self-attention；KV 增量 mask 另处理
        mask = torch.triu(torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return att @ v
```

### C.5 面试考点卡片

| 主题 | 答法关键词 |
|------|------------|
| RMS vs LN | 无去均值、无偏置、RMS 缩放 |
| SwiGLU | 三矩阵、门控、中间宽 $8d/3$ |
| GQA | 少 KV 头、repeat KV、Cache↓ |

### C.6 练习题加量（15 道）

1. RMSNorm 是否等价于 LN 当 $\mu=0$？讨论。  
2. 写出 RMS 与 L2 范数关系。  
3. SwiGLU 参数量相对两层 MLP 如何估算？  
4. 为何推理吞吐用「tokens/s」衡量？  
5. MQA 何时可能掉点？  
6. GQA 分组数如何选择？  
7. 无 bias 线性层利弊？  
8. RMSNorm 放在 Attention 前还是 FFN 前？（LLaMA：子层前）  
9. 与 LN 的 LayerScale 区别？  
10. 组合 Pre-Norm + RMSNorm + SwiGLU + GQA 的 Block 画出来。  
11. ZeRO 与 GQA 都省显存，层次是否相同？  
12. KV int8 与 GQA 可否叠加？  
13. 长文本下谁更瓶颈：Attention 还是 KV？  
14. 解释带宽瓶颈。  
15. 本课与 A1 作业对齐点？  

---

## 附录 E：SwiGLU 参数量推导草稿

设模型维 $d$，中间维 $d_{\mathrm{ff}}$。SwiGLU 三矩阵：$\mathbf{W}_1,\mathbf{V} \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$，$\mathbf{W}_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$。

参数量近似：$2 d d_{\mathrm{ff}} + d_{\mathrm{ff}} d = d_{\mathrm{ff}} (2d + d) = 3 d d_{\mathrm{ff}}$（忽略 bias）。

令与经典 $8d^2$（$d_{\mathrm{ff}}=4d$ 时 $2\cdot d \cdot 4d=8d^2$）可比，解 $d_{\mathrm{ff}}$ 得约 $\frac{8}{3}d$ 量级（**示意**，常数依是否含 bias、是否融合而定）。

---

## 附录 F：RMSNorm 反向传播直觉（了解）

RMSNorm 对 $\mathbf{x}$ 的梯度涉及 RMS 分母；实现需数值稳定；PyTorch `autograd` 已处理。

---

## 附录 G：与 Lesson 04 的衔接

- Lesson 04：Attention + RoPE。  
- Lesson 05：Norm + FFN + GQA，拼成 **LLaMA 风格 Block**。

---

## 附录 H：英文面试段落

“LLaMA-family models use RMSNorm, RoPE and SwiGLU. GQA in newer models reduces KV heads and cache traffic; the exact attention type depends on model generation and size.”

---

## 附录 J：学习内容自检

| 学习内容 | 章节 |
|----------|------|
| 为何替换原始组件 | 第一节 |
| RMSNorm 公式与对比 | 第二节 |
| Pre/Post-Norm | 2.5 |
| SwiGLU 公式与门控直觉 | 第三节 |
| 隐层维度 8d/3 | 3.3、附录 E |
| GQA 演进与 KV | 第四节 |
| 四件套表 | 第五节 |
| 面试 10 题 | 第六节 |
| 练习题 | 第七、附录 C |

---

### 附录 K：结语

**RMSNorm + SwiGLU + GQA（+ RoPE）** 已成为「开源 LLM 标准答案」的常见配方；面试时把 **动机（稳、省、强）** 说清楚，把 **公式与形状** 写对，即可与面试官同频。

---

## 附录 L：面试追问 20 条（极简答案）

1. RMSNorm 有 bias 吗？→ 通常无。  
2. RMSNorm 可学习参数？→ `weight` $\alpha$。  
3. SwiGLU 几个 Linear？→ 常三个。  
4. SiLU 与 Swish？→ 常等价（$\beta=1$）。  
5. GQA 论文？→ Ainslie et al.（2023）；并非所有 LLaMA 型号都使用 GQA。
6. MQA 论文？→ Shazeer 等。  
7. 为何不叫 MHA-GQA？→ 命名习惯。  
8. KV 头数能任意吗？→ 需整除 Q 头数。  
9. $d_h$ 会变吗？→ 通常 $d/h$ 固定。  
10. SwiGLU 中间宽谁定？→ 架构搜索/经验。  
11. 还能用 GELU FFN 吗？→ 可以，效果权衡。  
12. RMSNorm 用于输出？→ 视实现，常在子层前。  
13. Final norm？→ LLaMA 有 `norm` 在 lm head 前。  
14. Weight tying？→ 独立话题。  
15. 偏置在 Attention？→ LLaMA 常无。  
16. 偏置在 SwiGLU？→ 常无。  
17. 激活函数还有 GeGLU？→ 同类门控变体。  
18. ReGLU？→ ReLU 门控变体。  
19. 选 SwiGLU 原因？→ 实验效果好。  
20. 本节与原始 Transformer 最大区别？→ **Norm+FFN+Attn 三处**均可能不同。  

---

## 附录 M：对比表扩展（经典 Block vs LLaMA 风格 Block）

| 子模块 | 经典 Decoder Block | LLaMA 风格 |
|--------|---------------------|------------|
| Norm 位置与类型 | 原始 Transformer 为 Post-LN + LayerNorm | **Pre-Norm + RMSNorm** |
| Self-Attn | MHA | **GQA**（较新/部分型号）+ **RoPE** |
| FFN | 原始 Transformer 为 ReLU MLP，后续也常用 GELU | **SwiGLU** |
| 偏置 | 常有 | **常无** |

---

## 附录 N：扩展阅读论文 / 关键词

- RMSNorm: *Root Mean Square Layer Normalization*  
- GLU Variants: *GLU Variants Improve Neural Network*（Shazeer）  
- GQA: *Grouped-Query Attention*（Ainslie et al.）  
- LLaMA: *LLaMA: Open and Efficient Foundation Language Models*  

---

## 导航

| 上一课 | 下一课 |
|--------|--------|
| [Lesson 04 - 多头注意力与 RoPE](04-多头注意力与RoPE.md) | [Lesson 06 - AdamW 优化器实现](06-AdamW优化器实现.md)（若已创建） |

[返回课程总览](00-课程总览与学习路线.md)

---
