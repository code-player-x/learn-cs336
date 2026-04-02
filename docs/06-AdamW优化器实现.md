# Lesson 06：AdamW 优化器实现 — 从梯度下降到解耦权重衰减

> **Stanford CS336**：Language Modeling from Scratch — 面向面试的体系化笔记（第 06 节）

**本节定位**：把 **梯度下降 → SGD → Momentum → Adam → AdamW** 串成一条线，讲清 **一阶/二阶矩、偏差修正、解耦权重衰减**，并覆盖 **学习率调度（warmup、余弦衰减、线性衰减、阶梯衰减）**、**梯度裁剪（按范数 / 按值）**、**混合精度训练（FP16 / BF16 / FP32、loss scaling）** 与 **CS336 Assignment 1** 对手写 AdamW 的典型要求。

**先修**：自动求导、参数张量、训练循环概念（Lesson 01）；Transformer 前向与反向（Lesson 03～05）。

**面试热度**：★★★★☆（中高频：Adam vs SGD、AdamW vs Adam、调度与裁剪、混合精度、显存开销）

---

## 概念详解（面向初学者）

### 优化器在训练里做什么？

语言模型训练本质是：给定损失 \(L(\theta)\)，在极高维参数空间 \(\theta\) 上**迭代减小损失**。每一步典型流程为：

1. **前向**计算 loss；  
2. **反向**得到梯度 \(\mathbf{g}_t = \nabla_\theta L(\theta_{t-1})\)；  
3. **优化器**根据 \(\mathbf{g}_t\) 与历史统计，更新 \(\theta_t\)。

**AdamW** 是当前 **Decoder-only LLM 预训练** 的主流选择之一（常与 **cosine + warmup**、**全局梯度范数裁剪**、**混合精度** 一起出现）。理解它，等于理解「现代 LM 训练脚本里一半的超参」。

---

### 1. 梯度下降（Gradient Descent, GD）

**全批量梯度下降**使用**整个数据集**上的平均梯度：

\[
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta L_i(\theta_t)
\]

- \(\eta\)：**学习率（learning rate）**，控制每步沿负梯度方向走多远。  
- **优点**：梯度方向是「真实」的期望方向，更新稳定。  
- **缺点**：\(N\) 很大时每一步都扫全数据，**太慢**；且只有大 batch 才近似稳定。

**直觉**：站在损失曲面上，每一步朝「最陡下坡」的方向走一小步。

---

### 2. 随机梯度下降（SGD）

对每个 **mini-batch**（大小 \(B \ll N\)）用样本梯度近似：

\[
\theta_{t+1} = \theta_t - \eta \cdot \mathbf{g}_t
\]

其中 \(\mathbf{g}_t\) 是当前 batch 的梯度（是总体梯度的无偏估计，但**方差大**）。

- **优点**：每步计算量小，可在海量数据上迭代。  
- **缺点**：噪声大，损失曲面抖动；在狭长「峡谷」里易**之字形**震荡，收敛慢。  
- **工程**：常配合 **Momentum**、**学习率调度**、**weight decay**。

**与 GD 对比**：SGD 用噪声换速度；batch 越大，梯度估计越稳，越接近 GD。

---

### 3. 带动量的 SGD（Momentum）

引入 **速度** \(\mathbf{v}_t\)，把历史梯度做指数滑动平均，阻尼震荡、加速一致方向：

\[
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_t
\]
\[
\theta_{t+1} = \theta_t - \eta \cdot \mathbf{v}_t
\]

常见变体也会写成 \(\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\mathbf{g}_t\)，与 Adam 中一阶矩形式统一，本质是 **对梯度的 EMA（指数滑动平均）**。

- **直觉**：以前几步的「惯性」冲过平坦区、减少直角弯折。  
- \(\beta\)：典型 **0.9**。  
- **局限**：仍只有**一个全局学习率**缩放整向量；各参数维度的梯度尺度差异大时，不如自适应方法省心。

---

### 4. Adam：一阶矩、二阶矩、偏差修正与更新式

**Adam**（Adaptive Moment Estimation）同时维护：

- **一阶矩** \(\mathbf{m}_t\)：梯度的指数滑动平均（可理解为梯度方向的「平滑估计」，与 momentum 思想相通）；  
- **二阶矩** \(\mathbf{v}_t\)：梯度**逐元素平方**的指数滑动平均（刻画各维梯度的**尺度**，用于自适应归一化）。

**递推（与 PyTorch / 论文常见写法一致）**：

\[
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t
\]
\[
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2
\]

其中 \(\mathbf{g}_t^2\) 表示 **Hadamard 逐元素平方**（每个参数位置独立）。

**偏差修正（bias correction）**：初始化 \(\mathbf{m}_0=\mathbf{0}\)、\(\mathbf{v}_0=\mathbf{0}\)，导致初期 \(\mathbf{m}_t,\mathbf{v}_t\) **系统性偏小**（尤其 \(t\) 很小时）。定义：

\[
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
\]

**参数更新（Adam 核心步）**：

\[
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\]

- **\(\epsilon\)**：数值稳定项（典型 **1e-8**），加在 \(\sqrt{\hat{\mathbf{v}}_t}\) 上防止分母过小。  
- **直觉**：\(\hat{\mathbf{m}}_t\) 定更新方向；\(\sqrt{\hat{\mathbf{v}}_t}\) 做逐维缩放——历史梯度幅度大的维，有效步长自动被压低（**自适应**）。

---

### 5. AdamW：解耦权重衰减（Decoupled Weight Decay）

**经典「Adam + L2」** 常在实现上把 L2 项并入梯度（\(\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\theta_t\)），该项再进入 \(\mathbf{m}_t,\mathbf{v}_t\) 的 EMA，与 **自适应分母 \(\sqrt{\hat{\mathbf{v}}}\)** 耦合，**权重衰减的有效行为**与「对 \(\theta\) 显式 \(\ell_2\) 惩罚」不一致。

**AdamW**（Loshchilov & Hutter, 2019）将 **weight decay** 与 Adam 的自适应更新**解耦**。常见等价写法之一：

\[
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta \lambda \theta_t
\]

即：先做标准 Adam 步，再对权重做 **与二阶矩无关** 的衰减（实现上常写作在自适应步之后执行 `θ ← θ - η·λ·θ`）。

| 维度 | L2 并入梯度（与 Adam 耦合） | AdamW（解耦） |
|------|---------------------------|---------------|
| 正则如何作用 | 进入 \(\mathbf{m},\mathbf{v}\)，与 \(\sqrt{\hat{\mathbf{v}}}\) 纠缠 | **不**进入矩估计，单独衰减 \(\theta\) |
| 与自适应关系 | 衰减强度受二阶缩放影响 | 衰减与自适应步长 **独立** |
| LLM 实践 | 较少作为主配置 | **预训练常用** |

**面试一句话**：在 Adam 里把 L2 当梯度加进去会改变自适应行为；**AdamW 把 weight decay 当作参数空间上的显式收缩**，更符合「解耦 decay」的工程语义。

---

### 6. 超参数备忘

| 符号 | 含义 | 典型值 / 说明 |
|------|------|----------------|
| \(\eta\) / `lr` | 学习率 | 与调度器配合；base 常见 \(10^{-4}\sim 3\times 10^{-4}\)（视模型与 batch 而定） |
| \(\beta_1\) | 一阶矩 EMA 衰减 | **0.9** |
| \(\beta_2\) | 二阶矩 EMA 衰减 | **0.999**（通用）；部分 **LLM** 用 **0.95** 等更短窗口 |
| \(\epsilon\) | 分母稳定项 | **1e-8**（FP32）；半精度下有时需略调 |
| `weight_decay` | \(\lambda\) | **0.01** 量级常见于 Transformer（需任务验证） |

---

### 7. 学习率调度：warmup、余弦、线性衰减、阶梯衰减

**Warmup**：训练前若干 step 将 \(\eta\) 从 0（或很小）**爬升**到目标 base lr。动机：初期 \(\mathbf{m},\mathbf{v}\) 估计不稳定；大 batch / 大 lr 下易数值爆炸；**Transformer** 类模型尤其依赖 warmup。

**Cosine decay**：在 warmup 结束后，学习率按余弦从 \(\eta_{\max}\) 平滑降到 \(\eta_{\min}\)（接近 0 或某一 floor）。典型形式（示意）：

\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\frac{\pi (t - T_{\mathrm{warm}})}{T_{\mathrm{decay}}}\right)
\]

**Linear decay**：从某步起 \(\eta\) **线性**降至 \(\eta_{\min}\)，形式简单，部分基线与课程作业采用。

**Step decay（阶梯衰减）**：每隔固定 epoch 或 step 将 \(\eta\) 乘以常数 \(\gamma\in(0,1)\)（如每 30 epoch ×0.1）。边界处 lr **突变**，可能带来 loss 抖动，但在 CV 传统任务中很常见。

**常见组合**：**linear warmup + cosine decay** 是 LLM 预训练标配之一。

---

### 8. 梯度裁剪：按范数与按值

**动机**：长序列、大模型或半精度下可能出现 **梯度爆炸**，单步更新过大导致 loss 发散。

**按全局范数裁剪（clip by global norm）**：先算所有参数梯度的整体 \(\ell_2\) 范数 \(G = \|\text{concat}(\mathbf{g}_i)\|_2\)。若 \(G > c\)，则所有梯度乘以 \(c/G\)，**方向不变、只缩小模长**。

**按值裁剪（clip by value）**：逐元素将 \(g_{ij}\) 限制在 \([-c, c]\)，**会改变方向**。

**LLM 训练更常用 global norm**：保留梯度方向，抑制极端大更新。

---

### 9. 混合精度训练（FP32、FP16、BF16、Loss Scaling）

**目标**：前向与反向用 **FP16 或 bfloat16** 加速、省显存；**主权重（master weights）** 常用 **FP32** 存储并在优化器里累加更新，减轻舍入误差。

| 格式 | 指数位 | 尾数位 | 特点 |
|------|--------|--------|------|
| FP32 | 8 | 23 | 全精度基准 |
| FP16 | 5 | 10 | 动态范围小，易 underflow/overflow |
| BF16 | 8 | 7 | 与 FP32 同指数宽度，**动态范围大**，尾数少 |

- **Loss scaling（FP16 常用）**：前向得到 loss 后先乘以较大因子 \(S\)（如 \(2^{16}\)），反传梯度同比例放大，避免梯度过小在 FP16 中下溢为 0；`optimizer.step()` 前再 **unscale**，最后用 `scaler.update()` 根据是否出现 inf 调整 \(S\)。  
- **BF16**：许多场景下**不需要** loss scaling（梯度不易下溢），实现更简单。  
- **Master weights**：半精度前向，FP32 存 \(\theta\) 做 `step`，再写回半精度权重（具体 API 因 `autocast` / FSDP 等而异）。

**为何大模型训练常用 BF16 而非 FP16？** 核心原因是 **动态范围**：BF16 与 FP32 **相同的 8 位指数**，对大激活/大梯度更宽容；FP16 指数位少，即便有 loss scaling，仍可能在某些层或长训练中出现 **数值不稳定**。在 A100/H100 等硬件上 BF16 吞吐高，已成为 LLM 预训练默认选项之一。但 **FP16 + loss scaling** 在成熟框架下同样广泛使用，最终以硬件支持与实测为准。

---

### 10. CS336 Assignment 1：优化器相关要求（与官方说明对齐）

Assignment 1（Basics）通常要求 **BPE + Decoder-only Transformer LM + 手写优化器 + 训练循环**。与优化器直接相关的要点（**以当年官方 README/PDF 为准**）：

1. **手写 AdamW，且不直接 `import` 使用 `torch.optim.AdamW`**：需实现 **bias correction** 与 **decoupled weight decay**。  
2. **公式对齐**：\(\theta \leftarrow \theta - \eta \left( \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) + \lambda \theta \right)\)（与先 Adam 步再 decay 等价）。  
3. **`param_groups`**：支持不同组不同 `lr` / `weight_decay`（例如 **bias 不衰减**）。  
4. **`state`**：每个参数存 `step`、`exp_avg`、`exp_avg_sq`，且在 **`model.to(device)` 之后** 创建，保证与参数 **同设备**。  
5. **训练循环**：`zero_grad` → `forward` → `loss` → `backward` →（可选）`clip` → `optimizer.step()` → `scheduler.step()`。  
6. **可复现**：固定种子、必要时 CUDA 确定性设置。

---

## 代码（PyTorch 从零实现）

### 1. 完整 AdamW：不依赖 `torch.optim.AdamW`

下面实现 **仅依赖 `torch`**，**不** `import torch.optim` 中的 AdamW，便于与 CS336 Assignment 1「禁止直接调用 `torch.optim.AdamW`」的要求一致。接口提供 `param_groups`，便于 bias 单独一组关闭 `weight_decay`。

```python
import math
from typing import Iterable, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn


Number = Union[float, int]
ParamGroup = List[Dict[str, Any]]


class AdamW:
    """
    纯 PyTorch 张量实现 AdamW，不依赖 torch.optim.AdamW。
    更新: θ ← θ - η * ( m_hat / (sqrt(v_hat) + ε) + λ * θ )
    等价于先 Adam 步，再 θ ← θ - η*λ*θ（decoupled weight decay）。
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], ParamGroup],
        lr: Number = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            self.param_groups = []
            for g in params:
                self.param_groups.append(
                    {
                        "params": list(g["params"]),
                        "lr": float(g.get("lr", lr)),
                        "betas": tuple(g.get("betas", betas)),
                        "eps": float(g.get("eps", eps)),
                        "weight_decay": float(g.get("weight_decay", weight_decay)),
                    }
                )
        else:
            self.param_groups = [
                {
                    "params": list(params),
                    "lr": float(lr),
                    "betas": (float(betas[0]), float(betas[1])),
                    "eps": float(eps),
                    "weight_decay": float(weight_decay),
                }
            ]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def _init_state(self, p: torch.nn.Parameter) -> Dict[str, Any]:
        return {
            "step": 0,
            "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format),
        }

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if p not in self.state:
                    self.state[p] = self._init_state(p)
                st = self.state[p]

                m, v = st["exp_avg"], st["exp_avg_sq"]
                st["step"] += 1
                t = st["step"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)

                denom = v_hat.sqrt().add_(eps)
                p.add_(m_hat.div_(denom), alpha=-lr)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def build_optimizer(model: nn.Module, lr: float = 3e-4, wd: float = 0.1) -> AdamW:
    """二维及以上参数（多为 weight）使用 weight_decay；一维（多为 bias/LayerNorm）不衰减。"""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return AdamW(
        [
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    )
```

> **说明**：`param_groups` 为 **字典列表** 时，每组可单独指定 `lr` / `weight_decay`；`build_optimizer` 将 **bias 等一维参数** 与 **权重矩阵** 分组，符合 Assignment 1 常见写法。

---

### 2. 继承 `Optimizer` 的精简版（若作业仅禁止 `AdamW` 类本身）

```python
import torch
from torch.optim import Optimizer


class AdamWRef(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(
                    p,
                    {
                        "step": 0,
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    },
                )
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                if wd:
                    p.add_(p, alpha=-lr * wd)
        return loss
```

---

### 3. 学习率调度：Linear Warmup + Cosine Decay

```python
def get_lr_linear_warmup_cosine(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine
```

每步将返回值写入 `for g in optimizer.param_groups: g["lr"] = lr`，再调用 `optimizer.step()`。**`step` 与 `scheduler` 的先后次序**须与参考实现一致（常见：`optimizer.step()` 后再更新下一步的 lr，或反之，二选一贯穿全程）。

---

### 4. 学习率调度：线性衰减（无余弦）

```python
def get_lr_linear_decay(
    step: int, base_lr: float, decay_start: int, total_steps: int, floor: float = 0.0
) -> float:
    if step < decay_start:
        return base_lr
    span = max(1, total_steps - decay_start)
    frac = min(1.0, (step - decay_start) / span)
    return base_lr + (floor - base_lr) * frac
```

---

### 5. 学习率调度：阶梯衰减（Step Decay）

```python
def get_lr_step_decay(
    step: int,
    base_lr: float,
    decay_steps: list,
    gamma: float = 0.1,
) -> float:
    """decay_steps 为升序列表，如 [10000, 20000]；每到一步乘以 gamma。"""
    lr = base_lr
    for s in decay_steps:
        if step >= s:
            lr *= gamma
    return lr
```

---

### 6. 梯度裁剪：按全局范数与按值

```python
import torch


def clip_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    device = params[0].device
    total_norm_sq = torch.stack([p.grad.detach().float().pow(2).sum() for p in params]).sum()
    total_norm = total_norm_sq.sqrt().clamp_min(1e-6)
    clip_coef = (max_norm / total_norm).clamp(max=1.0)
    for p in params:
        p.grad.detach().mul_(clip_coef.to(p.grad.dtype))
    return total_norm.to(device)


def clip_grad_value_(parameters, clip_value: float) -> None:
    for p in parameters:
        if p.grad is not None:
            p.grad.detach().clamp_(-clip_value, clip_value)
```

训练循环中可与 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` 对照使用。

---

### 7. 混合精度训练（FP16 + GradScaler 示意）

```python
# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()
# for batch in loader:
#     optimizer.zero_grad(set_to_none=True)
#     with autocast(dtype=torch.float16):
#         loss = model(...)
#     scaler.scale(loss).backward()
#     scaler.unscale_(optimizer)
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     scaler.step(optimizer)
#     scaler.update()
```

BF16 路径常用 `autocast(dtype=torch.bfloat16)`，许多硬件上**可不使用** `GradScaler`。

---

## 面试要点（速记清单）

| 主题 | 要点 |
|------|------|
| GD → SGD | 全批量 vs mini-batch；方差与速度权衡 |
| Momentum | 梯度 EMA；减震荡；仍非逐维自适应 |
| Adam | \(\mathbf{m}_t,\mathbf{v}_t\)；\(\hat{\mathbf{m}}_t,\hat{\mathbf{v}}_t\)；\(\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}\) |
| AdamW | weight decay **不**进矩估计；\(-\eta\lambda\theta\) |
| 超参 | \(\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}\)；`weight_decay` 与任务相关 |
| 调度 | warmup；cosine / linear / step decay；LLM 常见 warmup+cosine |
| 梯度裁剪 | global norm 保方向；clip by value 改方向 |
| 混合精度 | FP16+scaling；BF16 范围大；FP32 master weights |
| 显存 | 优化器状态常为 **2× 参数量**（\(m,v\)），dtype 依实现多为 FP32 |

---

## 面试高频题（10+ 道详解）

### Q1：Adam 和 SGD 的区别？

**答**：（1）**SGD**（含 Momentum）对每个参数使用**相同的全局学习率**缩放整个梯度向量；**噪声**来自 mini-batch，方向靠动量平滑。（2）**Adam** 额外维护 **梯度平方的 EMA**（二阶矩），用 \(\sqrt{\hat{\mathbf{v}}_t}\) **逐元素**归一化更新量，相当于 **自适应学习率**，不同参数维度步长比例可不同。（3）**代价**：Adam 需存 **\(m,v\)** 两份状态，**显存与计算**高于 SGD；部分任务上 SGD 泛化讨论较多，但 **LLM 预训练默认**多为 Adam 系（尤其 AdamW）。

---

### Q2：AdamW 和 Adam 的区别？为什么要解耦权重衰减？

**答**：**Adam** 若把 **L2 正则**实现为「往梯度里加 \(\lambda\theta\)」，该项会进入 \(\mathbf{m}_t,\mathbf{v}_t\) 的更新，与 **自适应分母 \(\sqrt{\hat{\mathbf{v}}}\)** **耦合**，导致「权重衰减」的强度随自适应缩放变化，**不再等价**于对 \(\theta\) 的显式 \(\ell_2\) 惩罚。**AdamW** 把 **weight decay** 从梯度/矩估计中分离，在自适应更新之后执行 \(\theta \leftarrow \theta - \eta\lambda\theta\)（与 \(\sqrt{\hat{\mathbf{v}}}\) 无关），称为 **解耦权重衰减（decoupled weight decay）**。大模型与 Transformer 上 **经验与可解释性**更好，故预训练常用 AdamW。

---

### Q3：学习率预热（warmup）的作用？

**答**：训练初期 **\(\mathbf{m},\mathbf{v}\) 从 0 初始化**，\(\hat{\mathbf{m}}/\sqrt{\hat{\mathbf{v}}}\) 在 **前几步可能异常大**；若 **base lr 较大** 或 **batch 很大**，易出现 **loss 尖峰、梯度爆炸、NaN**，混合精度下更明显。Warmup 在若干 step 内将 lr **从低到高**缓慢升到目标值，让矩估计与梯度统计 **稳定下来**。Transformer 类模型几乎**标配** warmup。

---

### Q4：为什么要做偏差校正（bias correction）？

**答**：\(\mathbf{m}_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}\mathbf{g}_i\) 在 **\(t\) 较小**时，由于 \(\mathbf{m}_0=\mathbf{0}\)，\(\mathbf{m}_t\) 的期望 **系统性地小于** 真实梯度的一阶矩估计；\(\mathbf{v}_t\) 同理。除以 \(1-\beta_1^t\) 与 \(1-\beta_2^t\) 相当于把 EMA 的 **尺度拉回** 与「真实矩」可比的无偏尺度。**\(t\) 大之后** \(1-\beta^t \to 1\)，修正量可忽略。

---

### Q5：\(\beta_1\) 和 \(\beta_2\) 的物理含义？

**答**：二者都是 **指数滑动平均的衰减系数**。\(\beta_1\) 控制 **一阶矩（梯度方向）** 的历史窗口：越接近 1，方向越平滑、惯性越大，典型 **0.9**。\(\beta_2\) 控制 **二阶矩（梯度平方、逐维幅度）** 的平滑：越接近 1，\(\hat{\mathbf{v}}\) 变化越慢、分母越稳定，典型 **0.999**；若 \(\beta_2\) **偏小**，\(\hat{\mathbf{v}}\) **波动更大**，有效步长更不稳定。部分长训 LLM 会尝试 **略小的 \(\beta_2\)**（如 0.95）以更快适应训练动态。

---

### Q6：梯度裁剪的两种方式？

**答**：（1）**按全局 \(\ell_2\) 范数（clip by global norm）**：把所有参数的梯度拼成一向量，若范数大于阈值 \(c\)，则整体乘以 \(c/\|\mathbf{g}\|\)，**方向不变**。（2）**按值裁剪（clip by value）**：对每个梯度元素截断到 \([-c,c]\)，**会改变方向**。LLM 训练 **普遍用 global norm**（如 1.0），与 AdamW、大序列更搭。

---

### Q7：混合精度训练的原理？FP16 vs BF16？

**答**：**原理**：用 **半精度**（FP16 或 BF16）做前向/反向矩阵运算以 **提速、省显存**；**权重更新**常在 **FP32 master weights** 上累加以减少舍入误差。**FP16**：1-5-10 布局，**动态范围小**，梯度易 underflow，常配合 **loss scaling**。**BF16**：1-8-7 布局，**指数与 FP32 同宽**，动态范围大，多数情况下 **更不易溢出/下溢**，常 **无需** loss scaling。二者尾数位均少于 FP32，**数值精度**均低于 FP32，需依赖 master weights 等技巧。

---

### Q8：为什么大模型训练常用 BF16 而不是 FP16？

**答**：首要原因是 **表示范围**：BF16 **指数位与 FP32 相同**，对大激活值、大梯度更鲁棒；FP16 **指数位少**，即使使用 loss scaling，在极深网络或长序列下仍可能 **不稳定**。其次，在 **A100/H100** 等 GPU 上 BF16 **Tensor Core** 支持成熟，吞吐高。FP16 仍在许多场景广泛使用（配合 GradScaler），**并非被淘汰**；选择常由 **硬件、框架与稳定性实验**共同决定。

---

### Q9：学习率调度策略有哪些？

**答**：**常数**（无调度）；**线性 warmup**；**余弦衰减**、**线性衰减**、**指数衰减**；**阶梯衰减（step decay）**；**多项式衰减**；**循环/重启（如 SGDR）** 等。LLM 预训练最常见组合是 **linear warmup + cosine decay**；CV 里 **step decay** 也很经典。需与 **总步数、batch、是否微调** 一起设计。

---

### Q10：AdamW 的参数量开销是多少？

**答**：对每个可训练参数，AdamW 通常维护 **一阶矩 \(\mathbf{m}\)** 与 **二阶矩 \(\mathbf{v}\)**，形状与参数相同，故 **状态张量元素个数约为模型可训练参数数量的 2 倍**。若 \(m,v\) 以 **FP32** 存储（常见），优化器状态显存约为 **\(2 \times 4 \times |\theta|\) 字节**（不计对齐与框架开销）；若与参数同 dtype 则随精度变化。**面试可答**：约为 **2 倍模型参数量的额外状态**（一阶 + 二阶矩）。

---

### Q11（加餐）：手写 AdamW 时最容易踩的坑？

**答**：（1）**步数 \(t\)** 未按参数一致更新，导致 bias correction 错误；（2）**把 weight decay 写进梯度**（耦合 L2）；（3）**在 `model.cuda()` 之前创建优化器**，`state` 留在 CPU；（4）**bias 与 weight 共用同一 `weight_decay`**，与论文不一致；（5）**scheduler 与 `optimizer.step` 顺序**与参考实现不一致导致 lr 曲线错位；（6）**原地运算**误改 \(\mathbf{m}_t\) 后再算 \(\hat{\mathbf{m}}_t\) 的依赖关系（需谨慎）。

---

## 练习

1. 默写 \(\mathbf{m}_t,\mathbf{v}_t\)、\(\hat{\mathbf{m}}_t,\hat{\mathbf{v}}_t\) 及 Adam 更新式；说明 \(t=1\) 时偏差修正因子分别为多少。  
2. 用两种表述对比 **AdamW** 与 **Adam + L2 并入梯度**。  
3. 手写 **linear warmup**：输入 `step`、`warmup_steps`、`base_lr`，返回当前 `lr`。  
4. 若 global norm 为 \(5\)、阈值为 \(1\)，裁剪后梯度范数是多少？  
5. 画表对比 FP32 / FP16 / BF16 的指数位、尾数位与动态范围直觉。  
6. \(\beta_2\) 从 \(0.999\) 改为 \(0.99\) 时，\(\hat{\mathbf{v}}_t\) 波动更大还是更小？对有效步长有何影响？  
7. 阅读 PyTorch `AdamW` 文档：`amsgrad` 选项与 weight decay 是否独立？  
8. 设计实验：同一小 MLP，固定种子，对比 SGD 与 AdamW 的 loss 曲线，预期现象是什么？  
9. 实现 **step decay** 调度，并在三个 `decay_steps` 上打印 lr 是否符合乘以 \(\gamma\) 的预期。  
10. 解释为何 FP16 训练常需要 `GradScaler`，而 BF16 常不需要。

---

## 导航

- **上一课**：[Lesson 05：RMSNorm、SwiGLU 与 GQA](./05-RMSNorm-SwiGLU-GQA.md)  
- **下一课**：[Lesson 07：训练循环与损失函数](./07-训练循环与损失函数.md)  
- **关联**：[Lesson 08：Assignment 1 实战指南](./08-Assignment1实战指南.md)  
- **总览**：[Lesson 00：课程总览与学习路线](./00-课程总览与学习路线.md)

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
| \(\eta\) | 学习率 |
| \(\mathbf{g}_t\) | 第 \(t\) 步梯度 |
| \(\mathbf{m}_t,\mathbf{v}_t\) | 一阶、二阶矩（未修正） |
| \(\hat{\mathbf{m}}_t,\hat{\mathbf{v}}_t\) | 偏差修正后的矩 |
| \(\beta_1,\beta_2\) | 一阶、二阶衰减率 |
| \(\lambda\) | weight decay 系数 |
| \(\epsilon\) | 数值稳定小常数 |

---

*文档版本：与 CS336 公开大纲、通用 PyTorch 实现及本仓库 Assignment 1 指南对齐；作业细则以当年官方说明为准。*
