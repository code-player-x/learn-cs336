# Lesson 10：FlashAttention 原理与 Triton

> **Stanford CS336**：Language Modeling from Scratch — 面试导向学习指南（第 10 节）

**先修**：[Lesson 09：GPU 架构与内存层级](./09-GPU架构与内存层级.md)（HBM / SRAM、算术强度）、[Lesson 04：多头注意力与 RoPE](./04-多头注意力与RoPE.md)。

**面试热度**：★★★★★（系统岗 / 大模型工程岗极高频）

---

## 导读

本节从 **标准 Scaled Dot-Product Attention** 的 **\(O(N^2)\) 峰值显存** 与 **HBM 访存瓶颈** 出发，讲清 **FlashAttention** 的 **IO 感知（IO-aware）** 设计：**分块（tiling）**、**在线 Softmax（online softmax）** 与 **融合内核**；给出 **显存复杂度 \(O(N)\)**、**IO 复杂度 \(O(N^2 d / M)\)**（\(M\) 为片上 SRAM / tile 相关规模）、**FlashAttention-2** 的工程改进；介绍 **Triton** 块级编程模型与 **PyTorch `F.scaled_dot_product_attention`**；并对接 **CS336 Assignment 2（Systems：Triton 实现 FlashAttention-2 等）**。

---

## 一、概念讲解（由浅入深）

### 1.1 标准 Attention 在算什么？

对单头、省略 batch 与 head 下标，设 \(\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N\times d}\)，缩放因子 \(s = 1/\sqrt{d}\)。标准前向为：

\[
\mathbf{S} = s \cdot \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N\times N}, \quad
\mathbf{P} = \mathrm{softmax}_{\mathrm{row}}(\mathbf{S}), \quad
\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N\times d}.
\]

**直觉**：每个 query 位置与 **所有** key 位置做点积得到一行 logits，经 softmax 得到权重，再对 value 行加权求和。复杂度上，主导项常写作 **\(O(N^2 d)\)** 次乘加（QK\(^\top\) 与 PV），但 **实现方式** 决定了你是 **算得快** 还是 **卡在搬运数据上**。

---

### 1.2 瓶颈一：\(O(N^2)\) 显存（物化注意力矩阵）

若 **显式物化** \(\mathbf{S}\)（未 softmax 的 logits）或 \(\mathbf{P}\)（概率矩阵），需要 **\(N \times N\)** 个浮点数。

- 当 \(N\) 从 2K 增到 32K、128K 时，**\(N^2\)** 项主导峰值显存，往往比 **\(N \times d\)** 的 Q/K/V 存储更「致命」。
- **反向传播** 若需长期保留完整 \(\mathbf{P}\) 或等价大张量，压力进一步放大。

**面试一句话**：朴素实现把 **完整 \(N \times N\) 注意力矩阵** 放在 **HBM**，峰值显存 **至少 \(\Theta(N^2)\)** 量级（常数与精度、是否存多份中间结果有关）。

---

### 1.3 瓶颈二：频繁 HBM 访问与 Memory-bound

GPU 存储层次（详见 Lesson 09）可粗分为：

| 层级 | 典型特征 |
|------|----------|
| **HBM** | 容量大（数十 GB），带宽相对算力 **仍常成为瓶颈** |
| **片上 SRAM（如 Shared Memory）** | 容量小，记教学分析中的 **\(M\)**（与 **tile 大小** 同阶），**有效带宽远高于 HBM** |

标准「多阶段」实现典型路径是：

1. 算 \(\mathbf{Q}\mathbf{K}^\top\) → **写 \(\mathbf{S}\) 到 HBM**；
2. 读 \(\mathbf{S}\) → softmax → **写 \(\mathbf{P}\) 到 HBM**；
3. 读 \(\mathbf{P}\)、\(\mathbf{V}\) → 算 \(\mathbf{P}\mathbf{V}\)。

每一步都把 **大块中间张量** 在 **HBM 与计算单元之间来回搬运**。当 **算术强度（FLOPs / Byte）** 偏低时，算子表现为 **访存受限（memory-bound）**：算力没跑满，时间花在 **等数据**。

**直觉**：Transformer 里 Attention 常常是 **「算得动但搬不动」** —— 优化目标之一是 **减少 HBM 访问量**，并 **避免把 \(N \times N\) 整块长期驻留**。

---

### 1.4 FlashAttention 核心思想：IO 感知（IO-aware）算法设计

**IO-aware** 的含义：不仅看 **渐近 FLOPs**，还把 **内存层级** 纳入设计 —— **尽量减少对慢速存储（HBM）的读写次数与数据量**，让 **热数据** 尽量留在 **片上 SRAM**，在 **一次或少数几次 kernel 启动** 内完成 **QK\(^\top\) → 稳定 softmax → PV** 的 **融合**。

三条主线：

1. **Tiling / Blocking**：把 \(\mathbf{Q},\mathbf{K},\mathbf{V}\) 沿序列维切成 **小块**，使每一步的中间结果 **能放进 SRAM 可容纳的 tile**。
2. **Online Softmax**：对每一行 softmax **不一次性看到整行 logits**，用 **递推统计量** 合并各子块贡献，从而 **无需完整 \(N \times N\) 的 \(\mathbf{S}\) 或 \(\mathbf{P}\)**。
3. **Kernel 融合**：在 **融合内核** 内完成矩阵乘与 softmax 规约，显著减少 **全局内存往返**。

**与「近似注意力」的区别**：FlashAttention 是 **精确（exact）** 的 softmax attention（在同一实数运算模型下与朴素实现 **数学等价**），不是稀疏近似或低秩近似。

---

### 1.5 分块计算（Tiling）：把 Q、K、V 切成能放进 SRAM 的块

**目标**：任何时刻 **不在 HBM 上持有完整 \(N \times N\) 矩阵**；在 **片上** 只保留 **当前 query 块 × 当前 key/value 块** 相关的中间量。

沿序列维分块。为便于理解，先看 **单个 query 行** \(\mathbf{q} \in \mathbb{R}^{d}\)。第 \(j\) 个 K/V 块（块长 \(B_K\)）：

\[
\mathbf{K}_j \in \mathbb{R}^{B_K \times d}, \quad \mathbf{V}_j \in \mathbb{R}^{B_K \times d}.
\]

该 query 与第 \(j\) 块对应的 **局部 logits**（列向量）为：

\[
\mathbf{s}_j = s \cdot \mathbf{K}_j \mathbf{q} \in \mathbb{R}^{B_K}.
\]

**真实 GPU kernel** 通常以 **query 块 × key 块** 的矩阵乘组织（便于走 **Tensor Core** 的 `tl.dot`），与上面 **行向量形式** 数学等价。

**处理顺序**：

- 对 **外层** query 块：固定一块 \(\mathbf{Q}\) 在片上（或寄存器/共享内存能覆盖的范围）；
- **内层** 沿 K/V 序列维 **依次** 扫过各个 key 块：每步计算局部 \(\mathbf{s}_j\)，用 **在线 softmax** 更新全局统计，并累积对 \(\mathbf{V}_j\) 的加权贡献；
- **绝不** 分配形状为 \((N,N)\) 的完整注意力矩阵。

---

### 1.6 Online Softmax：为什么不能「一块一块 softmax」再拼起来？

**问题**：标准 softmax 对一行 \(\mathbf{s} \in \mathbb{R}^{N}\) 需要 **全局最大值** 与 **全局求和**：

\[
p_i = \frac{e^{s_i - m}}{\sum_{k=1}^{N} e^{s_k - m}}, \quad m = \max_k s_k.
\]

若对每个块 **单独** 做 softmax 再拼接，**分母与基准最大值都错了** ——  softmax 是 **全局归一化**，不是各块独立归一化。

**解决**：维护 **递推的** 运行统计量，使 **每纳入一个新块** 时，等价于在 **「当前全局基准最大值」** 下 **重标度** 旧累积与新块贡献。

---

### 1.7 三个运行统计量：\(m\)（max）、\(\ell\)（exp-sum）、\(\mathbf{o}\)（输出累加器）

对 **单个 query 行**（向量记法）：

| 符号 | 含义 |
|------|------|
| \(m\) / \(M\) | **截至目前** 该行 logits 的 **全局最大值**（running max） |
| \(\ell\) / \(L\) | 在 **当前 \(m\) 为参考基准** 时，\(\sum_i \exp(s_i - m)\)（running sum of exponentials） |
| \(\mathbf{o}\) / \(\mathbf{O}\) | **未归一化** 的加权输出；处理完所有块后 **\(\mathbf{o} / \ell\)** 为最终 attention 输出 |

处理第 \(j\) 个 K/V 块前，记 **旧状态** 为 \(M_{\mathrm{old}}, L_{\mathrm{old}}, \mathbf{O}_{\mathrm{old}}\)；本块算出局部 \(m_j, \ell_j\) 及与 \(\mathbf{V}_j\) 的结合项。

---

### 1.8 块更新公式与重标度

对第 \(j\) 块，设：

\[
\mathbf{s}_j = s \cdot \mathbf{K}_j \mathbf{q}, \quad
m_j = \max(\mathbf{s}_j), \quad
\ell_j = \sum_{i \in \mathrm{block}_j} \exp(s_i - m_j).
\]

**合并后的新最大值**：

\[
M_{\mathrm{new}} = \max(M_{\mathrm{old}}, m_j).
\]

**重标度因子**（把「旧基准」和「新块局部基准」统一到 \(M_{\mathrm{new}}\)）：

\[
\mathrm{exp\_old} = \exp(M_{\mathrm{old}} - M_{\mathrm{new}}), \quad
\mathrm{exp\_new} = \exp(m_j - M_{\mathrm{new}}).
\]

**更新指数和（全局，基准为 \(M_{\mathrm{new}}\)）**：

\[
L_{\mathrm{new}} = L_{\mathrm{old}} \cdot \mathrm{exp\_old} + \ell_j \cdot \mathrm{exp\_new}.
\]

**更新未归一化输出**。记对块 \(j\) 在基准 \(M_{\mathrm{new}}\) 下的权重与 \(\mathbf{V}_j\) 的乘积为一块贡献，则：

\[
\mathbf{O}_{\mathrm{new}} =
\mathbf{O}_{\mathrm{old}} \cdot \mathrm{exp\_old}
+
(\mathrm{weights}_j \mathbf{V}_j),
\]

其中 \(\mathrm{weights}_j = \exp(\mathbf{s}_j - M_{\mathrm{new}})\)（逐元素），**工程上通常不先物化完整 \(\mathbf{P}\)**。

**该行全部块处理完毕后**：

\[
\mathbf{o}_{\mathrm{final}} = \mathbf{O}_{\mathrm{final}} \;/\; L_{\mathrm{final}}.
\]

---

### 1.9 数学正确性证明（为何与标准 softmax 一致）

设整行 logits 为 \(\mathbf{s}\)，全局最大值 \(M^\star = \max_i s_i\)。标准 softmax 权重 \(p_i = \exp(s_i - M^\star) / \sum_k \exp(s_k - M^\star)\)。

**归纳思路**：假设处理完前若干个块后，\((M_{\mathrm{old}}, L_{\mathrm{old}}, \mathbf{O}_{\mathrm{old}})\) 满足：

- \(M_{\mathrm{old}}\) 等于 **已覆盖下标集合** 上的最大值；
- \(L_{\mathrm{old}} = \sum_{i \in \mathrm{已覆盖}} \exp(s_i - M_{\mathrm{old}})\)；
- \(\mathbf{O}_{\mathrm{old}} = \sum_{i \in \mathrm{已覆盖}} \exp(s_i - M_{\mathrm{old}}) \mathbf{V}[i]\)（行向量形式对应 value 行）。

**并入新块** 时，令 \(M_{\mathrm{new}} = \max(M_{\mathrm{old}}, m_j)\)。

- 对 **旧下标** \(i\)：\(s_i\) 不变，但基准从 \(M_{\mathrm{old}}\) 变为 \(M_{\mathrm{new}}\)，故每个指数项乘以 \(\exp(M_{\mathrm{old}} - M_{\mathrm{new}}) = \mathrm{exp\_old}\)。因此旧贡献整体乘以 \(\mathrm{exp\_old}\)，与更新式一致。
- 对 **新块下标**：在基准 \(M_{\mathrm{new}}\) 下直接累加 \(\exp(s - M_{\mathrm{new}})\)，相当于 \(\ell_j \cdot \exp(m_j - M_{\mathrm{new}})\) 与 \(\mathbf{V}_j\) 的乘积形式（块内先以 \(m_j\) 为局部基准算出 \(\ell_j\)，再乘 \(\mathrm{exp\_new}\) 统一到 \(M_{\mathrm{new}}\)），与 **先全局 max 再 exp** 的定义 **一致**。

**结论**：在 **实数精确运算** 下，online softmax 与 **一次性全局 softmax** 等价。**浮点实现** 中因 **舍入顺序**、**非结合律**，可能与朴素实现有 **极小数值差异**。

---

### 1.10 FlashAttention 前向算法伪代码（概念级）

下面用 **单 query 行** 展示逻辑；完整 kernel 会对 **query 块** 向量化，并对 **batch / head** 分 grid。

```
输入: q, K, V, scale = 1/sqrt(d)
初始化: m = -inf, L = 0, O = 0 (长度为 d 的向量)

将 K, V 按列（序列维）分成块 K_1,...,K_J 与 V_1,...,V_J

for j = 1 to J:
    S_j = scale * (K_j @ q)           # 局部 logits，长度 = 块宽
    m_j = max(S_j)
    # 数值稳定：块内可先减 m_j 再 exp
    ell_j = sum(exp(S_j - m_j))

    m_new = max(m, m_j)
    exp_old = exp(m - m_new)
    exp_new = exp(m_j - m_new)

    P_partial = exp(S_j - m_new)      # 与全局基准一致
    contrib = P_partial @ V_j         # 向量，形状 (d,)

    L = L * exp_old + ell_j * exp_new
    O = O * exp_old + contrib
    m = m_new

return O / L
```

**要点**：循环结束后 **\(m\)** 即为该行 **全局 max**；**\(L\)** 为 **\(\sum_i \exp(s_i - m)\)**；**\(O/L\)** 即 \(\sum_i p_i \mathbf{V}[i]\)。

---

### 1.11 显存复杂度：由 \(O(N^2)\) 到 \(O(N)\)

| 实现 | 主导峰值 |
|------|----------|
| 物化 \(\mathbf{S}\) 或 \(\mathbf{P}\) | **\(\Theta(N^2)\)** |
| FlashAttention（不显式存完整矩阵） | **\(O(N)\)** 量级的 **每行统计**（\(m,\ell\)）与 **\(\mathbf{o} \in \mathbb{R}^d\)**，外加 **小块缓冲**（与 tile 尺寸有关） |

**表述建议**：峰值由 **「必须常驻的 \(N \times N\) 矩阵」** 变为 **「行级标量 + 长度 \(d\) 向量 + 片上 tile」**；对长序列 **极其关键**。

---

### 1.12 IO 复杂度：\(O(N^2 d / M)\)

在 **IO 模型** 中（关注 **HBM \(\leftrightarrow\) 片上** 搬运量与 **片上可重用容量 \(M\)**），FlashAttention 类方法对 Attention 的 HBM 访问量可概括为：

\[
\mathrm{IO}_{\mathrm{HBM}} \approx O\left(\frac{N^2 d}{M}\right).
\]

**直觉**：分子与注意力 **标量工作量** 同阶；分母 **\(M\)** 越大，单次 tile 能 **复用** 更多数据，**HBM 往返次数** 相对越少。

**注意**：这是 **量级分析**；真实性能还受 **occupancy、mask、序列是否整除块长、bank conflict、Tensor Core 利用率** 等影响。

---

### 1.13 FlashAttention-2 相对 FlashAttention-1 的改进

1. **更好的并行与工作划分**：减少 **warp 空泡**，提高 **SM 占用率**；对 **不同 warps / blocks** 的任务切分更精细。
2. **更少的非 matmul FLOPs**：归一化与重标度路径更贴近 **Tensor Core 友好** 的实现。
3. **反向传播更快**：整体训练吞吐提升；**前向数学** 与 FA-1 **一致**。

**一句话**：在 **相同数学** 下追求 **更高硬件利用率** 与 **更少零碎开销**。

---

### 1.14 Triton 编程模型

#### 什么是 Triton？

**Triton** 是 **OpenAI** 开源的 **GPU 编程语言与编译器栈**，用 **Python 风格** 编写 **融合算子**（Attention、LayerNorm 等），抽象层级 **高于 CUDA C/C++**，适合 **快速迭代** 自定义 kernel。

#### 块级编程（区别于 CUDA 的线程级）

| 维度 | CUDA | Triton |
|------|------|--------|
| 抽象单位 | **线程 / warp / block** 显式 | **program（块程序）** + **张量块运算** |
| 内存 | shared / register **手工** 为主 | **编译器** 辅助调度，接口偏 **tile** |
| 开发效率 | 样板多、细节多 | **融合 kernel** 迭代快 |
| 极限优化 | 专家可压榨极致 | 多数场景 **足够快** |

#### 常用 API（教学级）

- **`@triton.jit`**：标记 **JIT 编译** 的 kernel 函数。
- **`tl.program_id(axis)`**：当前 program 在 **launch grid** 中的坐标，用于 **划分数据块**（如 batch、head、query tile）。
- **`tl.load` / `tl.store`**：按 **块** 读写全局内存，支持 **`mask`** 处理尾部或不规则形状。
- **`tl.dot`**：块矩阵乘（对接 Tensor Core）。
- **`tl.where`**：按条件选择元素（如 **causal mask**：上三角置 \(-\infty\)）。
- **`tl.max`**：沿指定轴求最大值（online softmax 中求块内 max）。

---

### 1.15 FlashAttention 的 Triton 实现（简化轮廓）

典型结构：

1. **Grid**：`program_id` 映射到 **batch、head、query 序列块** 等维度。
2. **对每个 query tile**：沿 **K/V 序列维** 外层循环；内层维护 **\(m,\ell,\mathbf{acc}\)**（与论文中 \(\mathbf{O}\) 对应）。
3. **内层**：`tl.dot` 得 **QK\(^\top\)** 块 → **`tl.where` 应用 causal / padding mask** → **online softmax 更新** → `tl.dot` 累积 **PV**。
4. **写回**：**\(\mathbf{acc} / \ell\)** 写入输出。

**CS336 Assignment 2（Systems）** 通常要求：使用 **Triton** 实现 **FlashAttention-2** 风格内核，与 **PyTorch 参考实现** 做 **数值对齐（allclose）**、覆盖 **causal / 变长 / 多 head 维** 等，并提交 **基准测试（吞吐、与基线对比）**。细则以 **当年官方 README / PDF** 为准；常见仓库结构含 `assignment2-systems` 或类似命名，需实现 **`flash_attention_triton.py`** 等文件并通过 `pytest`。

---

### 1.16 PyTorch 原生：`F.scaled_dot_product_attention()`

PyTorch 提供统一入口 **`torch.nn.functional.scaled_dot_product_attention`**，内部根据 **硬件、dtype、形状、mask 类型** 选择 **FlashAttention 后端**、**memory-efficient** 路径或 **math** 兜底实现。

---

### 1.17 性能与适用场景

- **经验加速**：长序列、memory-bound 配置下，相对 **未融合朴素实现**，Attention 子模块常见 **约 2～4×** 加速（**非保证**，依 GPU、驱动、PyTorch 版本而变）。
- **显存**：避免 **\(N^2\)** 张量，**长上下文** 训练/推理 **更省显存**。

**局限**：极短序列时 kernel 启动与 tile **固定开销** 占比高；**复杂稀疏 mask** 可能无法走最快路径；低精度下可能有 **微小数值差**。

---

### 1.18 推理 vs 训练中的 FlashAttention

| 场景 | 常见优化点 |
|------|------------|
| **训练** | 前向 + 反向；常配合 **重计算（recomputation）** 降低激活显存；FA-2 强调 **反向吞吐**；与 **梯度检查点** 可叠加 |
| **推理** | 与 **KV Cache** 结合：只对 **新 token** 的 query 与 **缓存的 K/V** 做 attention；**PagedAttention** 等解决 **显存碎片**；FlashAttention 仍减少 **中间 \(N^2\)** 物化 |

面试可答：**训练** 侧重 **吞吐 + 反向显存**；**推理** 侧重 **延迟 + KV 缓存 + 批处理** 下的 **访存与内核选择**。

---

## 二、代码示例

### 2.1 教学用：单 query 行的 Online Softmax + 加权 V（NumPy）

```python
import numpy as np

def online_softmax_attention_row(q, K, V, scale):
    """
    q: (d,)
    K: (N, d)
    V: (N, d)
    返回与 softmax(scale * Q K^T) @ V 的一行等价的结果 (d,)
    """
    N, d = K.shape
    m_old = -np.inf
    L_old = 0.0
    O_old = np.zeros(d, dtype=np.float64)

    block = 128
    for j in range(0, N, block):
        Kj = K[j : j + block]
        Vj = V[j : j + block]
        sj = scale * (Kj @ q)  # (block,)

        mj = float(np.max(sj))
        lj = float(np.sum(np.exp(sj - mj)))

        M_new = max(m_old, mj)
        exp_old = np.exp(m_old - M_new)
        exp_new = np.exp(mj - M_new)

        weights = np.exp(sj - M_new)
        PV = weights @ Vj

        L_new = L_old * exp_old + lj * exp_new
        O_new = O_old * exp_old + PV

        m_old, L_old, O_old = M_new, L_new, O_new

    return O_old / L_old
```

---

### 2.2 极简 Triton：向量加法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

**要点**：`BLOCK_SIZE: tl.constexpr` 为 **编译期常量**；`mask` 处理 **非整除** 尾部。

---

### 2.3 教学用 Triton：沿最后一维做 Softmax（示意）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr,
    stride_row, n_cols,
    BLOCK_COL: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_row
    cols = tl.arange(0, BLOCK_COL)
    mask = cols < n_cols
    x = tl.load(inp_ptr + row_start + cols, mask=mask, other=-float("inf"))
    m = tl.max(x)
    x_shifted = x - m
    num = tl.exp(x_shifted)
    den = tl.sum(num)
    out = num / den
    tl.store(out_ptr + row_start + cols, out, mask=mask)
```

**说明**：真实场景需处理 **`n_cols` 大于 `BLOCK_COL` 的分块规约**；此处展示 **`tl.max`、`tl.exp`、`tl.sum`** 的用法。FlashAttention 的 online softmax 是在 **K 维循环** 中 **增量** 更新 **\(m,\ell\)**，而不是对整个行一次性 `max/sum`。

---

### 2.4 FlashAttention 风格 Triton 内核（骨架，非完整可运行）

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    # ... K/V/Out 的 stride, scale, seqlen, causal 等
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # start_m = tl.program_id(0) * BLOCK_M + ...
    # q = tl.load(...)  # [BLOCK_M, HEAD_DIM]

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # for start_n in range(0, seqlen_k, BLOCK_N):
    #     k = tl.load(...)
    #     s = tl.dot(q, tl.trans(k)) * scale
    #     s = tl.where(causal_mask, s, float("-inf"))
    #     m_ij = tl.max(s, 1)
    #     m_new = tl.maximum(m_i, m_ij)
    #     alpha = tl.exp(m_i - m_new)
    #     p = tl.exp(s - m_new[:, None])
    #     l_i = l_i * alpha + tl.sum(p, 1)
    #     acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    #     m_i = m_new

    # o = acc / l_i[:, None]
    # tl.store(Out + ..., o)
    pass
```

Assignment 2 实现时需补全：**指针算术、mask、dtype、边界、反向或对照测试**。

---

### 2.5 PyTorch：`scaled_dot_product_attention`

```python
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
B, H, T, D = 2, 8, 4096, 64
q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

out = F.scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
)
```

较新 PyTorch 可通过环境变量或上下文配置 **SDPA 后端**；具体 API 以 [官方文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 为准。

---

## 三、面试要点（速记清单）

1. **瓶颈**：物化 **\(N \times N\)** \(\Rightarrow\) **\(O(N^2)\)** 显存；Attention 常 **memory-bound**，HBM 往返多。
2. **核心**：**IO-aware** = **分块 tiling** + **online softmax** + **融合 kernel**；**精确注意力**，非稀疏近似。
3. **三路量**：**\(m,\ell,\mathbf{O}\)**；**\(M_{\mathrm{new}}, \mathrm{exp\_old}, \mathrm{exp\_new}\)** 做 **基准统一**。
4. **归一化**：行处理完后 **\(\mathbf{O} / L\)**。
5. **显存**：不显式存完整注意力矩阵时，峰值 **约 \(O(N)\)**（相对 \(N^2\)）。
6. **IO**：**\(O(N^2 d / M)\)**，\(M\) 为片上 tile / SRAM 相关规模。
7. **FA-2**：**更好并行**、**更少非 matmul FLOPs**、**更快反向**。
8. **Triton**：**块级** GPU 语言；**`tl.load/store/dot/where/max`** + **`@triton.jit`**。
9. **正确性**：数学 **等价**；浮点 **可能有微小差**。
10. **PyTorch**：**`F.scaled_dot_product_attention`**；长序列收益大。
11. **推理/训练**：训练关注 **反向与重计算**；推理关注 **KV cache 与延迟**。
12. **A2**：Triton 实现 FA-2 风格 + **数值测试** + **benchmark**。

---

## 四、面试高频题详解（12+）

### Q1：FlashAttention 的核心思想是什么？

**答**：在 **不近似注意力定义** 的前提下，通过 **分块 tiling** 使计算主要在 **片上 SRAM** 完成，并用 **在线 Softmax** 维护每行的 **全局最大值 \(m\)**、**指数和 \(\ell\)** 与 **未归一化输出累加 \(\mathbf{O}\)**，从而 **不显式构造完整 \(N \times N\) 注意力矩阵**，显著 **降低 HBM 访问量** 与 **峰值显存**。本质是 **IO 感知（IO-aware）的精确注意力 + 融合内核**。

---

### Q2：为什么标准 Attention 是访存瓶颈？

**答**：标准多阶段实现会 **物化** 大尺寸中间张量（如 \(\mathbf{S}\)、\(\mathbf{P}\)），在 **HBM 与计算单元之间多次读写**；Attention 的 **算术强度** 往往不足以 **吃满算力**，表现为 **memory-bound**。此外 **\(N^2\)** 规模的张量 **占用带宽与显存**，进一步放大瓶颈。

---

### Q3：FlashAttention 如何实现分块 Softmax（online softmax 算法）？

**答**：将一行 logits 按 **K/V 块** 顺序处理。每块计算 **局部最大值 \(m_j\)** 与 **局部指数和 \(\ell_j\)**（块内可先减 \(m_j\) 稳定 exp）。用 **\(M_{\mathrm{new}} = \max(M_{\mathrm{old}}, m_j)\)** 更新全局最大值，用 **\(\exp(M_{\mathrm{old}}-M_{\mathrm{new}})\)** 与 **\(\exp(m_j-M_{\mathrm{new}})\)** 把 **旧累积** 与 **新块** 统一到 **同一基准最大值**，更新 **\(L\)** 与 **\(\mathbf{O}\)**；全程 **无需存储整行 logits**。最后 **\(\mathbf{O}/L\)** 即 softmax 加权后的输出。

---

### Q4：FlashAttention 的内存复杂度是多少？

**答**：**不显式物化** 完整 \(N \times N\) 矩阵时，额外需要 **每行** 的标量 **\(m,\ell\)** 与 **长度 \(d\) 的向量 \(\mathbf{O}\)**，以及 **小块临时缓冲**，峰值 **\(O(N)\)** 量级（与 batch、head 数线性；常数依赖 tile）。对比朴素 **\(\Theta(N^2)\)** 存储 **占主导** 的中间矩阵，长序列下差异巨大。

---

### Q5：FlashAttention 的 IO 复杂度是多少？

**答**：在常用 **IO 模型** 下，HBM 访问量可记为 **\(O(N^2 d / M)\)** 量级，其中 **\(M\)** 表示 **片上可重用数据量 / tile 规模**（与 SRAM、分块策略相关）。含义：**注意力计算规模** 与 **\(N^2 d\)** 同阶；**更大的 \(M\)** 提高数据复用，**减少 HBM 往返**。

---

### Q6：FlashAttention-2 相比 FlashAttention-1 有什么改进？

**答**：**数学定义不变**。改进主要在 **并行度与工作划分**（减少空泡、提高占用）、**减少非矩阵乘的零碎 FLOPs**、**更快的反向传播**，从而提升 **训练端到端吞吐**。

---

### Q7：Triton 和 CUDA 的区别？

**答**：**CUDA** 以 **线程** 为基本编程单位，需显式管理 **shared memory、同步、occupancy** 等；**Triton** 以 **program + 张量块** 组织计算，编译器承担更多 **调度与优化**，写 **融合算子** 更快。**极限手工优化** 仍可能用 CUDA；**快速实现 Attention 类融合内核** 时常选 Triton。

---

### Q8：什么是 IO 感知（IO-aware）算法设计？

**答**：在设计算法时 **不仅考虑 FLOPs**，还显式考虑 **内存层级**（尤其是 **HBM 与片上 SRAM 的速度差**），通过 **分块、融合、减少中间结果物化** 等手段 **降低对慢速存储的访问量**。FlashAttention 是典型：**减少 HBM 读写** 比 **减少乘法次数** 更关键。

---

### Q9：FlashAttention 是否会损失精度？为什么？

**答**：算法是 **数学上精确的 softmax attention**（非近似方法）。**实现层面** 使用与标准 softmax 相同的 **减最大值** 技巧保证稳定；与 **朴素实现** 相比，因 **浮点运算顺序不同**、**FP16/BF16**、**融合求和** 等，可能有 **极小数值差异**，通常 **可接受**。若需与参考严格对齐，应在作业中以 **`torch.allclose`** 等容差验证。

---

### Q10：FlashAttention 在推理和训练中分别有什么优化？

**答**：**训练** 中关注 **前向省显存**、**反向高效**（FA-2 优化反向）、常与 **重计算**、**梯度检查点** 配合降低激活。**推理** 中与 **KV Cache** 结合（只对新 query 与缓存 K/V 计算），并常与 **PagedAttention、连续批处理** 等配合；FlashAttention 仍避免 **大张量物化**，降低 **显存与带宽** 压力。

---

### Q11：如何在 PyTorch 中使用 FlashAttention？

**答**：优先使用 **`torch.nn.functional.scaled_dot_product_attention`**，保证 **CUDA**、**支持的 dtype（如 fp16/bf16）**、**张量布局（如 `(B, H, T, D)`）**；因果注意力设 **`is_causal=True`** 或传入兼容的 **`attn_mask`**。亦可安装 **`flash-attn`** 等库使用独立 API。是否走 Flash 后端取决于 **PyTorch 版本与硬件**；可用 profiler 或日志确认实际路径。

---

### Q12：分块计算（Tiling）的基本原理？

**答**：当 **工作集大于片上快速存储** 时，把大矩阵/张量沿某维切成 **小块（tile）**，使 **当前计算所需数据** 能放入 **SRAM/寄存器**，算完一块再载入下一块，通过 **提高数据局部性** 减少对 **HBM** 的重复访问。FlashAttention 将 **Q、K、V** 分块，并在块上维护 **online softmax** 统计量，**从不组装完整 \(N \times N\) 矩阵**。

---

### Q13（补充）：反向传播为什么也能省显存？

**答**：朴素实现常在 **前向保存大张量** 供反向使用。FlashAttention 常用 **重计算**：反向需要时在 **分块结构下重新计算** 部分中间量，以 **额外计算换存储**，避免 **\(O(N^2)\)** 激活常驻 HBM，从而降低 **反向峰值显存**。

---

## 五、练习

### 练习 1：复杂度对比

给定 \(N=65536, d=128\)，比较 **物化 \(\mathbf{P}\in\mathbb{R}^{N\times N}\)** 与 **仅每行存 \(\ell\) 与 \(\mathbf{o}\in\mathbb{R}^d\)** 的显存主导项阶别。

**提示**：\(N^2\) vs \(Nd\)；估算 float16 下字节数。

---

### 练习 2：手推块更新

给定 \(M_{\mathrm{old}}, L_{\mathrm{old}}\) 与新块 \(m_j, \ell_j\)，写出 \(M_{\mathrm{new}}, \mathrm{exp\_old}, \mathrm{exp\_new}, L_{\mathrm{new}}\)，并说明 \(L_{\mathrm{new}}\) 是在基准 \(M_{\mathrm{new}}\) 下的全行指数和。

---

### 练习 3：PyTorch 实测

用 **朴素 attention**（物化 \(\mathbf{P}\)）与 **`scaled_dot_product_attention`** 在 **fp16**、\(T=2048\) 下对比 **`torch.cuda.max_memory_allocated`** 与 **耗时**。

---

### 练习 4：Triton 阅读

阅读 Triton 官方教程中的 **vector add** 与 **matmul**，说明 **`program_id`** 如何映射到 **输出 tile**。

---

### 练习 5：CS336 Assignment 2 自检清单

对照当年官方说明，逐项勾选：

- [ ] **数值对齐**：与 `torch` 参考实现 **`allclose`**（给定 atol/rtol）。
- [ ] **Causal mask** 与 **padding / 变长**（若作业要求）。
- [ ] **多种 head 维 \(d\)**、**序列长度**（含非整除 block）。
- [ ] **性能基准**：相对基线或目标吞吐、记录 GPU 型号与 PyTorch 版本。
- [ ] **代码风格与提交格式**（如 `pytest` 全绿）。

---

## 六、导航

| 文档 | 说明 |
|------|------|
| [00-课程总览与学习路线](00-课程总览与学习路线.md) | CS336 全局路线图 |
| [09-GPU架构与内存层级](09-GPU架构与内存层级.md) | HBM/SRAM、算术强度（本节先修） |
| [04-多头注意力与RoPE](04-多头注意力与RoPE.md) | Attention 数学基础 |
| [08-Assignment1实战指南](08-Assignment1实战指南.md) | Assignment 1 整合 |
| [12-Assignment2系统优化实战](12-Assignment2系统优化实战.md) | Assignment 2 实战（若已收录） |

**下一课建议**：分布式训练（DDP / 通信）与 **Lesson 12** Assignment 2 实战文档衔接。

---

## 附录：符号表与延伸阅读

| 符号 | 含义 |
|------|------|
| \(N\) | 序列长度 |
| \(d\) | head 维度 |
| \(s\) | \(1/\sqrt{d}\) |
| \(M_{\mathrm{old}}, M_{\mathrm{new}}\) | 行 softmax 参考最大值 |
| \(L_{\mathrm{old}}, L_{\mathrm{new}}\) | 与当前最大值一致的指数和 |
| \(\mathbf{O}\) | 未归一化输出累加 |
| \(M\) | IO 分析中的片上 / tile 规模 |

**延伸阅读**

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（NeurIPS 2022）.
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
- [Triton 语言与教程](https://triton-lang.org/).
- PyTorch [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

---

*文档面向 CS336 与面试复习；公式与复杂度为教学表述，工程实现以具体 GPU、驱动、PyTorch 与作业当年说明为准。*
