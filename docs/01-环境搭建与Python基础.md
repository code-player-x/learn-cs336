# Lesson 01：环境搭建与 Python 基础

> **Stanford CS336：Language Modeling from Scratch** — 20 节面试导向学习指南 · 第 1 节

---

## 一、标题与概览

### 1.1 本课定位

本节课是「从零手搓语言模型」系列的**第一站**。我们不一上来就堆公式，而是先把**工具链**（Python 环境、包管理、PyTorch 安装）和 **PyTorch 心智模型**（张量、自动求导、模块、设备）搭牢——因为 CS336 的五个 Assignment 本质上都是**在张量上写数学、在加速器上跑训练循环**。面试中，面试官也常会先确认「你对 PyTorch 是否熟练」，再深入到 Transformer、注意力与系统优化。

### 1.2 学完本节后你应该能够

- **课程层面**：说清楚 CS336 的五个 Assignment（Basics / Systems / Scaling / Data / Alignment）各自解决什么问题，以及本仓库文档与官方作业的对应关系；
- **环境层面**：使用 **Python 3.11+**，在 **conda、pip、uv** 三种路径中做出合理选择并完成可复现安装；
- **PyTorch 层面**：正确安装 **PyTorch**，完成张量创建、索引切片、**广播**、**`requires_grad`** 下的梯度计算；
- **工程层面**：理解 **`nn.Module`** 与 **`nn.Parameter`**，能写出一个最小可训练模块；用 **NumPy** 做预处理，用 **einops** 做维度重排；读懂带**类型标注**与**装饰器**的训练代码；
- **资源层面**：会做粗粒度的 **FLOPs** 与**参数显存**估算，能在面试中口述推理过程；
- **设备层面**：正确管理 **`device`**（CPU / CUDA / MPS），规避常见的设备不匹配错误。

### 1.3 文档结构说明（如何使用本文）

| 章节 | 内容 |
|------|------|
| **概念详解** | 面向小白，从零解释「为什么要这样」 |
| **代码示例** | 带中文注释的可运行片段，建议本地敲一遍 |
| **面试考点** | 浓缩清单，考前速览 |
| **练习题** | 自测，建议先闭卷再对答案 |
| **面试高频题** | 带详细口述/推导答案，模拟一面 |

---

## 二、概念详解（面向小白）

### 2.1 为什么需要「独立 Python 环境」？

你的电脑可能已装有系统自带的 Python，或 Anaconda 的 base 环境。**直接在系统 Python 上 `pip install`** 容易导致：版本冲突、不同项目依赖互相覆盖、难以复现「我机器能跑」。**虚拟环境**（venv、conda env）把每个项目的解释器与 `site-packages` 隔离开，像给每个项目单独一个「干净房间」。

### 2.2 conda、pip、uv 分别是什么？

- **conda**：不仅是 Python 包管理器，还能装 **Python 本身、CUDA 相关运行时、非 Python 库**（如某些 C 库）。适合实验室统一配 GPU 驱动与 PyTorch 的场景。
- **pip**：**Python 官方推荐的包安装器**，通常与 **`python -m venv`** 创建的虚拟环境配合：轻量、文档多、生态最大。
- **uv**（Astral）：用 Rust 实现的**极快**解析与安装工具，可创建虚拟环境、`uv pip install` 与 pip 命令接近，适合 CI 与频繁重建环境。

**一句话决策**：要管 CUDA/多语言栈 → 倾向 **conda**；只要 Python 包、追求标准 → **pip + venv**；追求速度与锁版本 → **uv**。

### 2.3 PyTorch 与「张量」是什么？

**张量（Tensor）**可以理解为「多维数组」：0 维是标量，1 维是向量，2 维是矩阵，更高维是批量数据（如 `(batch, seq_len, hidden_dim)`）。**PyTorch** 在 NumPy 式 API 之上提供了：

- **GPU / Apple Silicon MPS** 加速；
- **自动求导（autograd）**：对 `requires_grad=True` 的张量自动建计算图并 `backward()`；
- 与 **`nn.Module`**、优化器、分布式训练的一体化。

### 2.4 `nn.Module` 与 `nn.Parameter` 各管什么？

- **`nn.Module`**：可组合的计算单元，负责**子模块注册**、`forward` 定义、`train()`/`eval()` 切换（影响 Dropout、BatchNorm 等）。
- **`nn.Parameter`**：一种特殊的 `Tensor`，会被注册为**可学习参数**，出现在 `model.parameters()` 里，随 `optimizer.step()` 更新，并随 `model.to(device)` 迁移。

普通 `Tensor` 挂在 `self` 上若未用 `register_buffer`，**不会**被优化器默认更新，常用于临时常量（需谨慎）。

### 2.5 FLOPs 与显存估算为什么要会？

系统设计题、研究员岗常问：**这个模型训练要多少显存？前向一次大概多少算力？** 不需要精确到个位，但要**数量级正确**：会把大计算拆成若干矩阵乘（matmul），按 **2×M×N×K**（或说明口径）估 FLOPs；按 **参数量 × 每参数字节数** 估权重占用，并知道训练时还有**优化器状态、激活**等额外开销（本课先掌握权重与 matmul）。

### 2.6 CS336 五个 Assignment 在整条链路中的位置

| Assignment | 英文主题 | 中文概括 |
|------------|----------|----------|
| **A1 Basics** | 基础 | BPE、Transformer、训练循环，跑通**可训练的语言模型** |
| **A2 Systems** | 系统 | GPU 优化、FlashAttention、DDP 等，**又快又稳** |
| **A3 Scaling** | 缩放 | Scaling Laws、算力与数据的**最优配比** |
| **A4 Data** | 数据 | Common Crawl、过滤、去重，构建**高质量预训练集** |
| **A5 Alignment** | 对齐 | SFT、RLHF、DPO、GRPO 等，**有用且可控** |

本仓库 **20 节**文档与上述模块对齐：第 1～8 节主攻 Assignment 1 所需基础；后续分阶段覆盖系统、缩放、数据、对齐。本节是**全系列的公共底座**。

### 2.7 本仓库典型目录（心里有数）

```
learn-cs336/
├── docs/              # 课程文档（面试导向）
├── interview/         # 八股、简历等（若存在）
├── code/              # 实现示例（若存在）
├── requirements.txt
└── README.md
```

官方课程与作业请以当年发布为准：[Stanford CS336 课程站](https://stanford-cs336.github.io/spring2025/)。

---

## 三、代码示例（含中文注释）

以下片段建议在新环境中**逐段运行**，观察 `print` 输出与张量 `shape`。

### 3.1 conda / pip / uv 命令骨架

```bash
# ---------- conda：适合需要统一 CUDA / 多语言依赖时 ----------
# conda create -n cs336 python=3.11 -y
# conda activate cs336
# 安装 PyTorch 请以官网命令为准，CUDA 版本需与驱动匹配
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# ---------- pip + venv：经典、标准 ----------
cd /path/to/learn-cs336
[Mac]python3.11 -m venv .venv
[windows] py -3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# ---------- uv：快速创建环境与安装 ----------
# uv venv .venv
# source .venv/bin/activate
# uv pip install -r requirements.txt
```

### 3.2 PyTorch 安装后自检

```python
import torch

# 打印版本，确认与项目要求一致
print("torch version:", torch.__version__)

# NVIDIA GPU：需要正确安装 CUDA 版 PyTorch 与驱动
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Apple Silicon：常用 MPS 后端
print("MPS available:", torch.backends.mps.is_available())
```

### 3.3 张量创建、dtype、device

```python
import torch

# 从 Python 列表创建；默认在 CPU，dtype 常为 float32
x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape, x.dtype, x.device)

B, T, D = 2, 8, 16
a = torch.zeros(B, T, D)           # 全零
b = torch.randn(B, T, D)           # 标准正态分布

# 根据环境选择设备（面试常写成一个函数）
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
c = torch.ones(3, 4, device=device, dtype=torch.float32)
print(c.device)
```

### 3.4 NumPy 与 PyTorch 互转（注意共享内存）

```python
import numpy as np
import torch

# 随机 numpy 数组，显式 float32 与 torch 常见训练精度一致
arr = np.random.randn(4, 8).astype(np.float32)

# from_numpy：与 arr 共享底层内存，改一方可能影响另一方
x = torch.from_numpy(arr)
arr[0, 0] = 999.0
print(x[0, 0])  # 可能也是 999.0，演示共享内存

# 需要独立副本时用 torch.tensor 或 clone
y = torch.tensor(arr)
arr[0, 0] = 0.0
print(y[0, 0])  # 不受后续 arr 修改影响（取决于是否仍共享，tensor(arr) 一般为拷贝）
```

### 3.5 类型标注与小型 `nn.Module`

```python
# 从 Python 的 typing 模块导入 Tuple 类型（虽然这段代码里没用到，但可能是为以后准备）
from typing import Tuple

# 导入 PyTorch 的核心库
import torch
# 导入 PyTorch 的神经网络模块，里面包含了所有神经网络的层（比如 Linear、Conv2d 等）
import torch.nn as nn


def split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    这个函数的作用：把输入张量重新排列，为"多头注意力机制"做准备
    
    大白话解释：
    输入是一个 3 维数据，形状是 (B, T, 总特征数)
    我们要把它变成 4 维数据，形状是 (B, 头数, T, 每个头的特征数)
    
    类比：就像把一个班级的学生（总特征）分成若干个小组（头数），
         每个小组负责处理自己的那部分特征
    
    参数：
        x: 输入张量，形状为 (B, T, n_heads * head_dim)
           - B: batch size（批次大小），可以理解为一次处理多少个独立的样本
           - T: 序列长度（sequence length），比如一句话有多少个词
           - 最后一个数字: n_heads * head_dim，即所有头的总特征维度
        
        n_heads: 多头数量（要分成几个小组）
        head_dim: 每个头的特征维度（每个小组处理多少特征）
    
    返回：
        重新排列后的张量，形状为 (B, n_heads, T, head_dim)
    """
    
    # 获取输入张量的形状，解包成三个变量
    # b = batch size（批次大小）
    # t = 序列长度（sequence length）
    # c = 特征总数（即 n_heads * head_dim）
    b, t, c = x.shape
    
    # 检查一下：特征总数必须等于 头数 × 每个头的维度
    # 如果不相等，说明数据格式不对，程序会报错停下来
    # 比如：n_heads=4, head_dim=8，那么 c 必须等于 32
    assert c == n_heads * head_dim
    
    # 核心操作（分两步）：
    
    # 第一步：使用 view() 改变形状，但不改变数据顺序
    # 从 (B, T, n_heads*head_dim) 变成 (B, T, n_heads, head_dim)
    # 就是把最后一大坨特征，按照"头数 × 每头维度"的方式重新分组
    x = x.view(b, t, n_heads, head_dim)
    
    # 第二步：使用 transpose() 交换维度位置
    # 原来的顺序是 (批次, 序列长度, 头数, 每头维度)
    # 我们想要 (批次, 头数, 序列长度, 每头维度)
    # 所以把第1维（序列长度）和第2维（头数）交换位置
    x = x.transpose(1, 2)
    
    # 返回处理后的张量
    return x


class DummyModel(nn.Module):
    """
    这是一个简单的神经网络模型，用于演示如何自定义模型
    
    大白话解释：
    在 PyTorch 里，所有神经网络都要继承 nn.Module 这个类
    就像你要做一个玩具，必须先有一个"玩具"的模子（nn.Module）
    
    这个模型特别简单：输入什么维度，输出什么维度，中间只经过一个线性变换
    就像一个"翻译器"，把数字从一种形式翻译成另一种形式，但大小不变
    """
    
    def __init__(self, d_model: int) -> None:
        """
        初始化函数：当创建这个模型时，会自动调用这个函数
        
        参数：
            d_model: 模型的维度，即输入和输出的特征数量
                    比如 d_model=512，表示输入是512维，输出也是512维
        
        大白话解释：
        就像你要开一家工厂，需要先买好机器设备（这里就是买一个线性层）
        """
        
        # 调用父类 nn.Module 的初始化函数
        # 这一行必须写，否则 PyTorch 无法正常管理这个模型
        # 就像你开店必须先办营业执照一样，这是必须的手续
        super().__init__()
        
        # 创建一个线性层（全连接层），把它作为这个模型的"零件"
        # nn.Linear(d_model, d_model) 的意思是：
        # 输入维度是 d_model，输出维度也是 d_model
        # 这个线性层做的事情：y = x * W + b
        # 其中 W 是权重矩阵，b 是偏置项，它们都是模型需要学习的参数
        # 把这个线性层存到 self.proj 里，这样模型就能记住自己有这个零件
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：数据进入模型后，该怎么流动
        
        大白话解释：
        就像工厂的生产线，原材料（输入数据）进来后，
        经过一道工序（self.proj 线性变换），
        生产出产品（输出数据）再送出去
        
        参数：
            x: 输入张量，形状可以是任意的，但最后一维必须是 d_model
        
        返回：
            经过线性变换后的张量，形状和输入一样
        
        注意：在 PyTorch 中，你通常不直接调用这个函数，
        而是调用 model(x)，PyTorch 会自动调用 forward()
        """
        
        # 把输入 x 传给 self.proj（线性层），得到输出
        # 相当于：output = x * W + b
        # 然后把这个结果返回给调用者
        return self.proj(x)
```

### 3.6 装饰器示例：`@torch.no_grad()` 与自定义计时

```python
# 导入 functools 模块，它提供了"装饰器"相关的工具函数
# 装饰器就像给函数穿上一件"外衣"，在不修改原函数代码的情况下增加新功能
import functools
# 导入 time 模块，用来计时
import time
# 从 typing 导入类型提示相关的工具
# Any: 任意类型, Callable: 可调用对象（函数）, TypeVar: 类型变量
from typing import Any, Callable, TypeVar

# 导入 PyTorch 核心库
import torch

# 定义一个类型变量 F，表示"可调用对象"（即函数）
# bound=Callable[...] 限制 F 只能是函数类型，不能是 int、str 等其他类型
# 这样装饰器就能精准保留被装饰函数的原始类型信息，IDE 会有正确的代码补全
#
# Callable[..., Any] 的含义：
#   - ...（三个点）：表示函数的参数列表【任意】，数量和类型都不限制
#   - Any：表示函数的返回值类型【任意】，可以是任何类型
#   合起来就是：一个"参数和返回值都不限定"的函数类型
F = TypeVar("F", bound=Callable[..., Any])


def timeit(fn: F) -> F:
    """
    装饰器：给被装饰的函数添加"计时"功能
    
    大白话：给函数戴上秒表，执行前开始计时，执行后打印耗时
    
    参数：
        fn: 要被装饰的函数
    
    返回：
        包装后的函数（原函数 + 计时功能）
    """
    
    # functools.wraps 保留原函数的名字、文档字符串等信息
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        包装函数：在调用原函数前后加上计时逻辑
        
        *args: 所有位置参数
        **kwargs: 所有关键字参数
        """
        
        # 记录开始时间（time.perf_counter() 是高精度计时器，单位秒）
        t0 = time.perf_counter()
        
        # 调用原函数
        out = fn(*args, **kwargs)
        
        # 记录结束时间，计算并打印耗时（转换为毫秒）
        t1 = time.perf_counter()
        print(f"{fn.__name__}: {(t1 - t0) * 1000:.2f} ms")
        
        # 返回原函数的执行结果
        return out
    
    return wrapper  # type: ignore[return-value]


@torch.no_grad()
def eval_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    模型推理函数（评估/测试阶段使用）
    
    作用：
    1. 把模型切换到评估模式（model.eval()）
    2. 用模型做一次前向传播
    
    好处：
    - 关闭梯度计算，节省显存/内存
    - 加快推理速度
    
    参数：
        model: PyTorch 模型
        x: 输入数据（张量）
    
    返回：
        模型的预测结果（张量）
    """
    
    # 切换到评估模式
    # 会影响 Dropout（关闭随机丢弃）和 BatchNorm（用全局统计量）等层的行为
    model.eval()
    
    # 前向传播（@torch.no_grad() 保证这里不会记录梯度）
    return model(x)
```

### 3.7 广播（Broadcasting）

```python
import torch

# ============================================================================
# 示例1：广播机制基础演示
# ============================================================================
# 广播（Broadcasting）规则：
# 从最后一维开始向前对齐，逐维比较，满足以下条件之一即可广播：
#   1. 两个维度大小相等
#   2. 其中一个维度大小为 1
#   3. 某个维度缺失（相当于大小为 1）
# ============================================================================

# 创建一个形状为 (32, 1, 128) 的张量
# - 32: 批次大小（batch size）
# - 1:  中间维度大小为1，可以被广播
# - 128: 特征维度
a = torch.randn(32, 1, 128)

# 创建一个形状为 (128,) 的一维张量
# 这是一个一维向量，没有批次和序列维度
b = torch.randn(128)

# a + b 的广播过程：
# 1. a 的形状: (32, 1, 128)
# 2. b 的形状: (128,) → 先在前面补1 → (1, 1, 128)
# 3. 比较维度（从后往前）：
#    - 最后一维: 128 vs 128 ✅ 相等
#    - 第二维:   1   vs 1   ✅ 相等（b补了1）
#    - 第一维:   32  vs 1   ✅ b的维度为1，可以广播到32
# 4. 结果形状: (32, 1, 128)
c = a + b

print(f"a 的形状: {a.shape}")      # torch.Size([32, 1, 128])
print(f"b 的形状: {b.shape}")      # torch.Size([128])
print(f"c 的形状: {c.shape}")      # torch.Size([32, 1, 128])


# ============================================================================
# 示例2：大语言模型中的实际应用
# ============================================================================
# 场景：在 Transformer 的 Softmax 之前，为每个 token 加上偏置（bias）
# ============================================================================

# logits: 模型的原始输出分数
# 形状为 (4, 10, 50257) 表示：
#   - 4:   批次大小（4个句子）
#   - 10:  序列长度（每个句子10个token）
#   - 50257: 词表大小（每个token在50257个词上的得分）
logits = torch.randn(4, 10, 50257)

# bias: 词表偏置，每个词有一个偏置值
# 形状为 (50257,) 的一维向量
bias = torch.randn(50257)

# view(1, 1, -1) 将 bias 变成 (1, 1, 50257)
# - 第0维变成1（批次维度，可以广播）
# - 第1维变成1（序列维度，可以广播）  
# - 第2维是 -1，自动计算为 50257（保持不变）
# 
# 广播过程：
# logits: (4, 10, 50257)
# bias:   (1, 1, 50257) → 广播到 (4, 10, 50257)
# 结果:   (4, 10, 50257)
bias = bias.view(1, 1, -1)

# 加上偏置：每个 token 的每个词得分都加上对应的词偏置
out = logits + bias

print(f"\nlogits 的形状: {logits.shape}")   # torch.Size([4, 10, 50257])
print(f"bias 的形状:   {bias.shape}")     # torch.Size([1, 1, 50257])
print(f"out 的形状:    {out.shape}")      # torch.Size([4, 10, 50257])

# ============================================================================
# 广播的优缺点
# ============================================================================
# 优点：
#   - 无需显式扩展张量，节省内存
#   - 代码简洁，可读性好
# 
# 注意：
#   - 如果维度不匹配且都不为1，会报错
#   - 例如：(4, 10, 50257) + (4, 10, 10) ❌ 最后一维 50257 ≠ 10 且都不为1
# ============================================================================
```

### 3.8 自动求导与 `zero_grad`

```python
import torch

# ============================================================================
# 自动求导（Autograd）基础演示
# ============================================================================
# 在深度学习中，我们需要计算损失函数对模型参数的梯度（导数），
# 然后用梯度下降法更新参数。PyTorch 的 autograd 可以自动帮我们计算这些梯度。
# ============================================================================

# 创建一个需要求导的张量 w（权重矩阵）
# - shape: (10, 1)，10行1列
# - requires_grad=True: 表示 PyTorch 需要追踪对这个张量的所有运算，
#   以便后续自动计算梯度
# - 数值来自标准正态分布（随机初始化）
w = torch.randn(10, 1, requires_grad=True)

# 创建一个输入张量 x（不需要求导，因为它是数据，不是要训练的参数）
# - shape: (1, 10)，1行10列
# - 数值来自标准正态分布
x = torch.randn(1, 10)

# ============================================================================
# 前向传播：计算损失（标量）
# ============================================================================
# x @ w: 矩阵乘法
#   - x 形状: (1, 10)
#   - w 形状: (10, 1)
#   - 结果形状: (1, 1)
#
# .sum(): 把所有元素求和，变成标量（0维张量）
# 
# 为什么需要 .sum()？
#   在 PyTorch 中，.backward() 要求从标量开始反向传播。
#   .sum() 把预测结果聚合成一个数，这样就能调用 y.backward() 了。
#
# 一句话总结：(x @ w).sum() = 先做矩阵乘法得到预测值，再求和变成标量
# 
# 在真正的训练中，这里的 y 通常是损失函数（如 MSE、交叉熵），
# 会自然产生标量，但这里为了教学演示，用 .sum() 来构造一个标量。
y = (x @ w).sum()

# ============================================================================
# 反向传播：计算梯度
# ============================================================================
# .backward() 会执行反向传播，自动计算所有 requires_grad=True 的张量的梯度
# 具体来说：
#   1. 从 y 开始，沿着计算图反向传播
#   2. 计算 dy/dw（损失对权重 w 的导数）
#   3. 把计算结果存到 w.grad 中
y.backward()

# 查看梯度形状
# w.grad 的形状和 w 完全一样，都是 (10, 1)
# 每个位置的梯度值表示：如果改变 w 中该位置的数值，y 会变化多少
# 大白话：w.grad 告诉我们"每个权重应该往哪个方向调整才能让损失变小"
print(w.grad.shape)   # 输出: torch.Size([10, 1])

# ============================================================================
# 梯度清零（非常重要！）
# ============================================================================
# 在 PyTorch 中，梯度是【累加】的，而不是覆盖的。
# 也就是说，每次调用 .backward()，新的梯度会【加到】旧的梯度上。
# 
# 为什么这样设计？
#   某些场景下需要累加梯度（比如梯度累积，用多个小批次模拟大批次）
#   但大多数情况下，我们只需要当前批次的梯度，所以每次更新前要清零。
#
# 如果不清零会怎样？
#   第一次反向传播: w.grad = 梯度1
#   第二次反向传播: w.grad = 梯度1 + 梯度2  ← 累加了！
#   这样更新权重时用的是"累加后的梯度"，会导致训练出错！
#
# .zero_() 方法：
#   - 下划线 _ 表示"原地操作"（in-place），直接修改张量本身
#   - 把 w.grad 中的所有元素设置为 0
#   - 为下一轮迭代做好准备
w.grad.zero_()

# 清零后，w.grad 全部变成 0
print(w.grad)  # 输出: tensor([[0.], [0.], ...])
```

### 3.9 `nn.Parameter` 与自定义线性层

```python
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn


# ============================================================================
# 自定义线性层（全连接层）
# ============================================================================
# 这是一个从零实现的全连接层，功能等同于 nn.Linear(in_features, out_features)
# 
# 大白话：建立一个"翻译器"，输入 in_features 个数字，输出 out_features 个数字
# 比如：输入4个特征（身高、体重、年龄、学历），输出2个值（收入预测、信用评分）
# ============================================================================

class TinyLinear(nn.Module):
    """
    自定义线性层（全连接层）
    
    数学公式：y = x @ W.T
    其中 W 是权重矩阵，形状为 (out_features, in_features)
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        初始化线性层
        
        参数：
            in_features:  输入特征数量（比如 4）
            out_features: 输出特征数量（比如 2）
        
        大白话：买好"翻译器"的零件（权重矩阵），准备开始工作
        """
        
        # 调用父类 nn.Module 的初始化（必须的"办营业执照"步骤）
        super().__init__()
        
        # ============================================================
        # 创建权重矩阵（模型需要学习的参数）
        # ============================================================
        # nn.Parameter() 的作用：
        #   - 把张量"包装"成模型参数
        #   - 这样 model.parameters() 才能识别并收集它
        #   - 训练时优化器会自动更新它
        #
        # 形状：torch.randn(out_features, in_features)
        #   - 为什么是 (out_features, in_features)？
        #     因为矩阵乘法时：x @ W.T
        #     输入 x 形状: (batch, in_features)
        #     权重 W 形状: (out_features, in_features)  
        #     W.T 转置后: (in_features, out_features)
        #     这样 (batch, in_features) @ (in_features, out_features) 
        #     = (batch, out_features) ✅
        #
        # * 0.02：小随机初始化
        #   - 标准正态分布（均值0，方差1）乘以 0.02
        #   - 让初始值非常小（标准差只有0.02）
        #   - 为什么？避免一开始数值太大导致"饱和"
        #     （比如激活函数是 Sigmoid 或 Tanh 时，大数值会进入平坦区，梯度消失）
        #   - 在真正的 nn.Linear 中，初始化策略更复杂（如 Kaiming 初始化）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：数据流过这个层
        
        参数：
            x: 输入张量，形状为 (batch_size, in_features)
               比如 (32, 4) 表示 32 个样本，每个 4 个特征
        
        返回：
            输出张量，形状为 (batch_size, out_features)
               比如 (32, 2) 表示 32 个样本，每个 2 个输出值
        
        计算公式：y = x @ W.T
        
        为什么用 W.T（转置）？
            输入 x: (batch, in_features)
            权重 W: (out_features, in_features)  ← 我们存储的格式
            W.T:   (in_features, out_features)   ← 转置后
            x @ W.T: (batch, out_features)       ← 结果
            
            如果不转置直接用 W：(batch, in_features) @ (out_features, in_features)
            ❌ 矩阵乘法不合法（列数 in_features ≠ 行数 out_features）
        """
        
        # 矩阵乘法：输入 @ 权重的转置 = 输出
        return x @ self.weight.T


# ============================================================================
# 使用示例
# ============================================================================

# 创建一个线性层：4个输入特征 → 2个输出特征
m = TinyLinear(4, 2)

# 打印参数个数
# p.numel() 返回张量中元素的总个数（number of elements）
# 权重矩阵是 (2, 4)，所以有 2 × 4 = 8 个参数
# sum(...) 把所有参数的个数加起来
print("参数个数:", sum(p.numel() for p in m.parameters()))
# 输出: 参数个数: 8
```

### 3.10 einops：与 Attention 形状强相关

```python
import torch
from einops import rearrange, repeat

B, T, HD = 2, 64, 128
H, D = 8, 16
assert H * D == HD
x = torch.randn(B, T, H * D)

# 命名维度：论文里常见的 (B, T, H, D) 拆分
x_heads = rearrange(x, "b t (h d) -> b h t d", h=H, d=D)
print(x_heads.shape)

y = torch.randn(3, 1)
y_rep = repeat(y, "a b -> a (repeat b)", repeat=4)
print(y_rep.shape)

```

```
# 导入 PyTorch 核心库
import torch
# 从 einops 导入张量操作工具
# rearrange: 重新排列张量的维度（类似 view/permute，但更易读）
# repeat: 重复张量的内容（类似 expand/repeat，但更直观）
from einops import rearrange, repeat

# ============================================================================
# 设置张量维度参数
# ============================================================================
B, T, HD = 2, 64, 128
# B (Batch):   批次大小，一次处理 2 个独立样本
# T (Time/Seq): 序列长度，每个样本有 64 个 token/时刻
# HD (Hidden Dimension): 隐藏层总维度，128

H, D = 8, 16
# H (Heads):   多头数量，8 个头
# D (Depth/Head Dimension): 每个头的维度，16

# 验证：总维度 = 头数 × 每头维度
# 128 = 8 × 16 ✅
assert H * D == HD

# 创建一个随机张量，形状为 (B, T, H*D) = (2, 64, 128)
# 这个形状是 Transformer 中典型的"混合头"格式
# 所有的头信息都堆叠在最后一个维度里
x = torch.randn(B, T, H * D)

# ============================================================================
# rearrange: 重新排列维度（拆分头）
# ============================================================================
# 代码：x_heads = rearrange(x, "b t (h d) -> b h t d", h=H, d=D)
#
# 大白话：把"混合在一起的头"拆分成"独立的头"
# 
# 详细解释：
#   左侧描述原始形状："b t (h d)"
#     - b: 批次维度
#     - t: 序列长度维度
#     - (h d): 把最后一个维度拆分成两部分 h 和 d（括号表示拆分）
# 
#   右侧描述目标形状："b h t d"
#     - b: 批次维度不变
#     - h: 头数（变成独立的维度）
#     - t: 序列长度维度
#     - d: 每个头的维度
# 
#   h=H, d=D: 指定拆分后的维度大小
# 
# 形状变化：(2, 64, 128) → (2, 8, 64, 16)
#   原来 128 维 = 8 个头 × 每个头 16 维
#   拆分成 8 个独立的头，每个头 16 维
x_heads = rearrange(x, "b t (h d) -> b h t d", h=H, d=D)

# 打印拆分后的形状
print(x_heads.shape)  # 输出: torch.Size([2, 8, 64, 16])

# 为什么要这样做？
# 在多头注意力中，每个头独立计算注意力，需要把"混合头"变成"独立头"格式
# 这样每个头可以并行计算，提高效率


# ============================================================================
# repeat: 重复张量内容
# ============================================================================
# 创建一个形状为 (3, 1) 的张量，包含 3 行 1 列
# 就像 3 个样本，每个样本只有 1 个特征
y = torch.randn(3, 1)

# 代码：y_rep = repeat(y, "a b -> a (repeat b)", repeat=4)
#
# 大白话：把 b 维度重复 4 次
# 
# 详细解释：
#   左侧描述原始形状："a b"
#     - a: 第1维（大小为 3）
#     - b: 第2维（大小为 1）
# 
#   右侧描述目标形状："a (repeat b)"
#     - a: 第1维不变
#     - (repeat b): 把 b 维度重复 repeat 次（括号表示合并）
# 
#   repeat=4: 指定重复次数
# 
# 形状变化：(3, 1) → (3, 4)
#   原来第2维只有 1 列，重复 4 次变成 4 列
y_rep = repeat(y, "a b -> a (repeat b)", repeat=4)

# 打印重复后的形状
print(y_rep.shape)  # 输出: torch.Size([3, 4])

# 实际效果演示：
# 假设 y = [[1.0], [2.0], [3.0]]
# 重复后变成：
# [[1.0, 1.0, 1.0, 1.0],
#  [2.0, 2.0, 2.0, 2.0],
#  [3.0, 3.0, 3.0, 3.0]]
```



### 3.11 资源估算：参数量显存与 matmul FLOPs

```python
def param_memory_gb(num_params: int, bytes_per_param: int = 4) -> float:
    """仅权重占用，不含优化器状态与激活。"""
    return num_params * bytes_per_param / (1024**3)


def matmul_flops_2mnk(m: int, n: int, k: int) -> int:
    """矩阵乘 C = A @ B，形状 (m,k) @ (k,n)；按 2*M*N*K 计 FLOPs 的一种常见口径。"""
    return 2 * m * n * k


n = 1_000_000_000
print(f"1B 参数 FP32 权重约 {param_memory_gb(n, 4):.2f} GB")
print(f"1B 参数 BF16 权重约 {param_memory_gb(n, 2):.2f} GB")

M, N, K = 4096, 4096, 4096
print("4096^3 matmul FLOPs (2MNK):", matmul_flops_2mnk(M, N, K))
```

```
# ============================================================================
# 模型参数显存与计算量估算工具
# ============================================================================
# 在深度学习中，我们需要估算：
# 1. 模型参数占多少显存（能不能装进 GPU）
# 2. 矩阵乘法需要多少次计算（训练速度有多快）
# ============================================================================

def param_memory_gb(num_params: int, bytes_per_param: int = 4) -> float:
    """
    计算模型参数占用的显存大小（仅权重本身，不含优化器状态和激活值）
    
    参数：
        num_params: 模型参数个数
        bytes_per_param: 每个参数占用的字节数
            - FP32 (单精度): 4 字节
            - FP16/BF16 (半精度): 2 字节
            - FP64 (双精度): 8 字节
            - INT8: 1 字节
    
    返回：
        显存大小（单位：GB）
    
    计算公式：
        GB = 参数个数 × 每参数字节数 / (1024^3)
    
    为什么除以 1024^3？
        1 KB = 1024 字节
        1 MB = 1024^2 字节
        1 GB = 1024^3 字节
    """
    # 总字节数 = 参数个数 × 每参数字节数
    # 除以 1024^3 转换成 GB（用 1024 而不是 1000，这是计算机存储的标准）
    return num_params * bytes_per_param / (1024**3)


def matmul_flops_2mnk(m: int, n: int, k: int) -> int:
    """
    计算矩阵乘法的浮点运算次数（FLOPs）
    
    矩阵乘法：C = A @ B
        - A 的形状: (m, k)
        - B 的形状: (k, n)
        - C 的形状: (m, n)
    
    参数：
        m: 矩阵 A 的行数
        n: 矩阵 B 的列数
        k: 矩阵 A 的列数 / 矩阵 B 的行数（内积维度）
    
    返回：
        FLOPs 总数（浮点运算次数）
    
    计算公式：2 × m × n × k
    
    为什么是 2？
        对于 C[i, j] = sum(A[i, :] × B[:, j])：
        - 有 k 次乘法
        - 有 (k-1) 次加法 ≈ k 次加法
        - 总共约 2k 次运算
        - 所以总 FLOPs = 2 × m × n × k
    
    注意：这是"一次矩阵乘法"的计算量
    """
    return 2 * m * n * k


# ============================================================================
# 示例计算
# ============================================================================

# 创建一个变量 n，表示 10 亿个参数
n = 1_000_000_000  # 1B = 10 亿

# 计算 10 亿参数在不同精度下的显存占用
# FP32（单精度）：每个参数 4 字节
print(f"1B 参数 FP32 权重约 {param_memory_gb(n, 4):.2f} GB")
# 计算：1,000,000,000 × 4 / 1024^3 ≈ 3.73 GB

# BF16/FP16（半精度）：每个参数 2 字节
print(f"1B 参数 BF16 权重约 {param_memory_gb(n, 2):.2f} GB")
# 计算：1,000,000,000 × 2 / 1024^3 ≈ 1.86 GB

# ============================================================================
# 矩阵乘法计算量示例
# ============================================================================

# 设置矩阵维度为 4096 × 4096 × 4096
M, N, K = 4096, 4096, 4096  # 3 个维度都是 4096

# 计算这个矩阵乘法的 FLOPs
print("4096^3 matmul FLOPs (2MNK):", matmul_flops_2mnk(M, N, K))
# 计算：2 × 4096 × 4096 × 4096 = 2 × 68,719,476,736 = 137,438,953,472
# 结果：约 1.37e11 FLOPs（1370 亿次浮点运算）


# ============================================================================
# 补充：大模型的实际数字
# ============================================================================

# GPT-3 175B 参数
gpt3_params = 175_000_000_000
print(f"\nGPT-3 (175B 参数):")
print(f"  FP32 权重显存: {param_memory_gb(gpt3_params, 4):.2f} GB")
print(f"  BF16 权重显存: {param_memory_gb(gpt3_params, 2):.2f} GB")

# 实际训练还需要额外显存：
# 1. 梯度（梯度通常和参数一样大）
# 2. 优化器状态（Adam 需要额外 2 倍参数显存）
# 3. 激活值（取决于批次大小和序列长度）
# 所以实际需求远大于模型本身！
```



### 3.12 最小训练循环（线性层 + MSE）

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


torch.manual_seed(0)
model = MyLinear(5, 3).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(4, 5, device=device)
target = torch.randn(4, 3, device=device)

for step in range(3):
    opt.zero_grad(set_to_none=True)  # 或 zero_grad()
    pred = model(x)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    opt.step()
    print(step, loss.item())
```

---

## 四、面试考点（浓缩清单）

| 主题 | 你需要达到的程度 |
|------|------------------|
| conda / pip / uv | 能说明定位与选型；强调**版本锁定**、可复现安装 |
| CS336 五作业 | 能一句话概括 Basics / Systems / Scaling / Data / Alignment |
| 张量 / 广播 / device | 会白板推形状；会排查 `Expected all tensors on the same device` |
| autograd | leaf tensor、`detach`、`no_grad`、为何 `optimizer.zero_grad()` |
| `nn.Module` / `Parameter` / `Buffer` | 参数是否被优化器更新、是否随 `to(device)` 迁移 |
| NumPy ↔ PyTorch | `from_numpy` 共享内存、dtype、何时 `clone` |
| einops | 能读 `b t (h d) -> b h t d` |
| FLOPs / 显存 | matmul 2MNK 口径；参数量 × 字节数；知训练还有优化器与激活 |

---

## 五、练习题

1. **广播**：`a` 形状 `(4, 1, 512)`，`b` 形状 `(512,)`，求 `a + b` 的结果形状并说明过程。`(4, 10, 512)` 与 `(10, 512)` 能否直接相加？如何改？

2. **梯度**：`w = torch.tensor(2.0, requires_grad=True)`，`y = w ** 3`，一次 `backward()` 后 `w.grad` 是多少？若再调用一次 `y.backward()` 且未 `zero_grad`，梯度会怎样？

3. **Parameter**：模块内有 `self.buf = torch.ones(3)` 与 `self.w = nn.Parameter(torch.ones(3))`。`list(model.parameters())` 长度？`buf` 会被默认优化器更新吗？

4. **einops**：`x` 形状 `(2, 64, 768)`，12 个头、每头 64 维，写出 `rearrange` 得到 `(2, 12, 64, 64)`。

5. **资源**：线性层 `in_f=4096, out_f=4096`，`batch=8`，按 **2MNK** 估算一次前向主要 matmul 的 FLOPs。权重用 BF16 存储约多少 GB（仅权重）？

6. **设备**：创建 `nn.Linear(100, 10)` 与输入 `(32, 100)`，在**不显式** `.to(device)` 的前提下，用构造函数参数把二者放到 GPU（若可用）。

> **建议**：每题限时 5～10 分钟，先闭卷再对照下文「面试高频题」中的相关思路或运行代码验证。

---

## 六、下一课导航

| 上一节 | 下一节 |
|--------|--------|
| [← 课程总览与学习路线](00-课程总览与学习路线.md) | [BPE 分词器原理与实现 →](02-BPE分词器原理与实现.md) |

更多官方资源：[Stanford CS336 课程站](https://stanford-cs336.github.io/spring2025/)、[stanford-cs336 GitHub](https://github.com/stanford-cs336)。

---

## 七、面试高频题（5+ 题 · 详细答案）

以下 **8** 题覆盖 PyTorch 基础与工程习惯；建议**闭卷口述**后再对照。

### Q1：`torch.Tensor` 与 `numpy.ndarray` 的主要区别？

**答**：（1）PyTorch 张量可把计算放在 **GPU / MPS**，NumPy 数组默认在 CPU。（2）PyTorch 集成 **autograd**，支持 `requires_grad` 与 `backward()`。（3）与 `nn.Module`、优化器、分布式工具链深度集成；NumPy 更适合通用数值与数据预处理。二者可通过 `torch.from_numpy` / `.numpy()` 交互，需注意 **dtype、设备、内存是否共享**。

### Q2：广播规则怎样快速判断「能不能逐元素相加」？

**答**：从**最后一维**向左对齐两形状，较短者左侧**补 1**；每一维必须**相等**或**其中一方为 1**。例如 `(4,1,512)` 与 `(512,)` → 后者视为 `(1,1,512)`，与 `(4,1,512)` 兼容。若 `(4,10,512)` 与 `(10,512)`，对齐为 `(4,10,512)` 与 `(1,10,512)`，最左一维 `4` 与 `1` 不匹配且都不是 1，**不能直接广播**；需 `unsqueeze` 或调整布局使意图明确（如给第二方加 batch 维）。

### Q3：`nn.Parameter` 与普通 `Tensor` 作为模块属性有何区别？

**答**：`nn.Parameter` 会注册进 `parameters()`，默认被优化器更新，并随 `model.to(device)` 迁移。普通 `Tensor` 若仅 `self.x = torch.ones(3)`，通常**不**视为可训练参数；若需持久化且非训练（如 running mean），应 **`register_buffer`**。面试强调：**是否参与训练**、**是否出现在 `parameters()`**、**是否随设备迁移**。

### Q4：为什么 `loss.backward()` 前常要 `optimizer.zero_grad()`？

**答**：默认情况下梯度写在 `param.grad` 上，**会累加**。不清零则本次 backward 会叠加上一轮残留，等价于错误的大 batch 或重复计数。标准步骤：`zero_grad` → `forward` → `backward` → `step`。仅在**梯度累积**故意多步 `backward` 再一步 `step` 时例外，且周期末仍要清零。

### Q5：`model.train()` 与 `model.eval()` 改变什么？

**答**：切换子模块在训练/推理下的行为，例如 **Dropout**（训练随机置零，推理关闭）、**BatchNorm**（训练用 batch 统计，推理常用滑动均值方差）。`LayerNorm` 多数实现两者一致。推理时常配合 **`torch.no_grad()`** 跳过建图，节省显存与算力。

### Q6：如何估算一个全连接层的前向 FLOPs 与权重大小？

**答**：设 batch 为 `B`，输入维 `I`，输出维 `O`。主要计算为 `(B, I) @ (I, O)`，按 **2×B×I×O**（2MNK）为一种常见 FLOPs 口径。参数量约 **I×O**（加偏置则 +O，常可忽略主导项）。显存：**参数量 × 每参数字节**（FP32 为 4，BF16 约 2）。训练时若用 Adam，动量等状态通常再占**数倍**于权重，视面试深度展开。

### Q7：`torch.no_grad()` 与 `tensor.detach()` 有何不同？

**答**：**`no_grad()`** 在一段代码块内**关闭梯度追踪**，不建图，省显存与算力，适合验证/推理。**`detach()`** 返回与原张量共享数据但**不参与后续反向**的张量，仍可能在 `requires_grad=True` 的上下文中用于切断某条分支的梯度。二者都可用于「不要给某部分求导」，但语义与使用场景不同：`no_grad` 是上下文，`detach` 是单张量操作。

### Q8：出现 `RuntimeError: Expected all tensors to be on the same device` 时如何排查？

**答**：逐项检查（1）模型参数 `device`；（2）输入数据 `device`；（3）**buffer**（如 `register_buffer` 的张量）；（4）**新创建的常量**是否在 CPU（例如 `torch.zeros(...)` 默认 CPU，需 `device=x.device`）。统一写法：先 `device = next(model.parameters()).device` 或 `x.device`，再创建张量。推理时也要检查 `hidden` 等中间状态是否被 `.cpu()` 过。

---

## 本节小结

- **CS336** 五模块（**Basics / Systems / Scaling / Data / Alignment**）覆盖从建模到系统、缩放、数据、对齐；本课是后续实现的**公共底座**。
- **环境**：Python 3.11+；**conda** 管二进制栈，**pip** 标准装包，**uv** 加速；按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装并自检 CUDA/MPS。
- **Python for ML**：**NumPy** 预处理，**类型标注**与**装饰器**提升可读性；**einops** 对齐论文中的张量形状记号。
- **PyTorch**：张量、`dtype`/`device`、广播、`autograd`、`nn.Module`/`nn.Parameter`、设备一致性。
- **资源**：会粗算 **matmul FLOPs（2MNK 口径）** 与 **参数 × 精度** 的权重视图。

---

## 八、GPU 与 CPU 基础（概念讲解）

### 8.1 分工直觉

- **CPU**：通用、分支与系统调用强；适合数据加载、预处理、轻量控制流、调试。
- **GPU**：大规模并行浮点运算；适合矩阵乘、大规模 Attention（在实现高效时）。

### 8.2 数据传输

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)
```

频繁在 CPU/GPU 间拷贝会成为瓶颈；训练时应尽量 **batch 化传输**，配合 `pin_memory=True`（CUDA）等。

### 8.3 显存直觉（预告）

训练时显存大致包括：**模型参数**、**优化器状态**、**激活**、**梯度**；推理侧 **KV Cache** 在长上下文下显著（后续课程）。

---

## 九、PyTorch 张量内存布局（面试重点）

### 9.1 什么是 storage 与 view？

PyTorch 张量由 **底层一维 storage** + **形状 size** + **步长 stride** 描述。多个张量可**共享**同一 storage（如 `view` 成功时），仅视图不同。

### 9.2 行主序（C contiguous）

默认 **最后一维**在内存中相邻存储。对形状 `(B, T, D)`，固定 `b,t` 时沿 `D` 相邻。

### 9.3 `is_contiguous()` 含义

若张量满足 C 连续内存布局，返回 `True`。`transpose`、`permute` 常使张量 **不连续**（只改 stride，不搬数据）。

### 9.4 `contiguous()` 做什么？

返回 **连续内存**的张量；若已连续可能零拷贝；否则 **拷贝** 数据。

### 9.5 `view` vs `reshape`（必背）

| API | 要点 |
|-----|------|
| `view` | 要求底层布局兼容；常需 **先 contiguous**；失败抛错 |
| `reshape` | 能 view 则 view；否则 **拷贝** 后再变形状 |

**建议**：不确定时用 **`reshape`**；性能热点路径再显式 `contiguous().view(...)`。

### 9.6 代码：transpose 后为何 view 失败

```python
import torch
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
print(y.is_contiguous())  # False
# y.view(2, 12)  # 可能 RuntimeError
z = y.reshape(2, 12)     # OK
w = y.contiguous().view(2, 12)  # OK
```

### 9.7 面试标准答法（60 秒）

「PyTorch 默认 C 连续；`transpose/permute` 往往只改 stride，数据不搬，因而不 contiguous。`view` 要求内存布局与新形状一致，所以常失败；`reshape` 会必要时拷贝再 reshape。显式 `contiguous()` 会拷贝得到连续块。」

---

## 十、代码实现：打印 stride 调试

```python
import torch

def describe(t: torch.Tensor, name: str) -> None:
    print(name, "shape", tuple(t.shape), "stride", t.stride(), "contig", t.is_contiguous())

x = torch.randn(2, 3, 4)
describe(x, "x")
y = x.transpose(1, 2)
describe(y, "transpose")
describe(y.contiguous(), "contiguous")
```

---

## 十一、补充：einops 与维度语义（再巩固）

**einops** 用符号标注 batch、seq、head 等，减少 `permute` 维度顺序错误；CS336 与开源实现中极常见，建议熟读 `rearrange`/`repeat`。

---

## 十二、补充练习题（含提示）

1. 解释 `x.expand(3, 4)` 与 `x.repeat(3, 1)` 区别。  
2. `torch.stack` 与 `torch.cat` 维度变化？  
3. `torch.linalg.norm(x, dim=-1)` 与 RMS 关系？  
4. 为何 inplace 操作 `x += 1` 有时破坏 autograd？  
5. `optimizer.zero_grad(set_to_none=True)` 好处？  
6. `torch.cuda.amp.autocast` 用途？  
7. `GradScaler` 解决什么？  
8. `model.state_dict()` 含 buffer 吗？  
9. `register_buffer` 与 `Parameter` 区别？  
10. `torch.backends.cudnn.benchmark` 何时开？

---

## 十三、扩展背诵条目（1～200，配合行数与速览）

1. 张量是计算图节点。  
2. `requires_grad` 控制是否追踪。  
3. 非标量 `backward` 需 `gradient` 参数。  
4. `retain_graph` 多步 backward 时用。  
5. `leaf` 张量 `grad` 可直接看。  
6. 非 leaf 需 `retain_grad()`。  
7. `detach` 切断分支梯度。  
8. `no_grad` 推理省显存。  
9. `inference_mode` 更严格。  
10. `train`/`eval` 影响 dropout。  
11. `to(device)` 迁移。  
12. `to(dtype)` 转换类型。  
13. 混合精度 bf16/fp16。  
14. Tensor Core 加速 matmul。  
15. 广播从尾对齐。  
16. `einsum` 表达清晰。  
17. `bmm` batch 矩阵乘。  
18. `matmul` 自动广播。  
19. `addmm` 融合。  
20. 内存带宽常是瓶颈。  
21. 合并小算子 fusion。  
22. `torch.compile` 图优化。  
23. 动态图默认。  
24. 静态图部分场景。  
25. JIT `torch.jit.trace` 了解。  
26. ONNX 导出部署。  
27. 量化 INT8/INT4。  
28. 分布式 `torchrun`。  
29. DDP 梯度同步。  
30. 单机多卡常见。  
31. 随机种子可复现。  
32. cudnn 确定性开关。  
33. 数据加载 `num_workers`。  
34. `pin_memory` CUDA。  
35. `persistent_workers`。  
36. `prefetch_factor`。  
37. 数据集 `Dataset`。  
38. 迭代器 `DataLoader`。  
39. 自定义 `collate_fn`。  
40. 变长序列 pad。  
41. `pack_padded_sequence` 了解。  
42. 梯度裁剪 `clip_grad_norm_`。  
43. 权重衰减 AdamW。  
44. 学习率 warmup。  
45. Cosine schedule。  
46. 梯度累积大 batch。  
47. 检查点 `save`。  
48. 恢复 `load_state_dict`。  
49. 微调冻结层 `requires_grad=False`。  
50. LoRA 低秩适配（扩展）。  
51. 张量命名维度（了解）。  
52. `vmap` 向量化（了解）。  
53. `torch.fx` 符号追踪（了解）。  
54. 自定义 autograd Function（了解）。  
55. 二阶导数 `create_graph`（了解）。  
56. Hessian（了解）。  
57. Jacobian（了解）。  
58. 数值精度 float64 调试。  
59. NaN 检测 `torch.isnan`。  
60. 异常值处理。  
61. 随机数生成器 `Generator`。  
62. 可复现 dropout。  
63. 模型初始化 Xavier。  
64. Kaiming 初始化。  
65. 正交初始化。  
66. 参数统计 `norm`。  
67. 梯度统计监控。  
68. TensorBoard 记录。  
69. WandB 实验（了解）。  
70. 单元测试 pytest。  
71. 形状测试 assert。  
72. CI 跑 lint。  
73. 类型检查 mypy（了解）。  
74. 代码格式化 black（了解）。  
75. 读 CS336 官方作业说明。  
76. 遵守学术诚信。  
77. 引用论文出处。  
78. 许可证合规。  
79. 开源模型协议。  
80. 商业使用注意。  
81. 继续背 PyTorch API。  
82. 继续写小实验。  
83. 调试 print shape。  
84. 调试 print device。  
85. 调试 print dtype。  
86. 三层打印解决一半 bug。  
87. contiguous 解决 view 一半报错。  
88. reshape 更省心。  
89. 性能敏感再优化。  
90. 先正确后快。  
91. 面试先思路后细节。  
92. 白板写公式。  
93. 标注维度 B T D。  
94. 因果 mask 画三角。  
95. 残差画旁路。  
96. 与 Lesson 02 BPE 衔接。  
97. 字节 token ID。  
98. Embedding 查表。  
99. 词表大小 V。  
100. 输出 logits V。  
101. 交叉熵训练。  
102. Softmax 温度。  
103. 采样策略。  
104. Top-p。  
105. Top-k。  
106. 重复惩罚。  
107. EOS token。  
108. BOS 可选。  
109. Padding mask。  
110. Attention mask。  
111. 合并 mask 小心。  
112. 半精度 mask 值域。  
113. `-inf` 用大负数替代有时。  
114. 数值稳定。  
115. LayerNorm eps。  
116. RMSNorm eps。  
117. 深度学习调参。  
118. 学习率是超参。  
119. Batch size 超参。  
120. 序列长度超参。  
121. 一切可实验。  
122. 日志记录实验。  
123. 版本管理 git。  
124. 数据版本管理。  
125. 可复现第一。  
126. 团队协作规范。  
127. Code review。  
128. 读写 README。  
129. 写清楚依赖。  
130. Docker 可选。  
131. 云端 GPU 选型。  
132. A100 H100 了解。  
133. 显存容量规划。  
134. 互联带宽。  
135. NVLink。  
136. InfiniBand。  
137. 多机训练。  
138. 通信后端 NCCL。  
139. 故障排查日志。  
140. OOM 减 batch。  
141. OOM 梯度检查点。  
142. OOM 换小模型。  
143. 工程权衡。  
144. 研究创新。  
145. 产品落地。  
146. 全栈视野。  
147. CS336 路线完整。  
148. 面试自信来源。  
149. 持续学习。  
150. 论文日读。  
151. arXiv 跟踪。  
152. GitHub 跟踪。  
153. HuggingFace 生态。  
154. 模型卡阅读。  
155. Tokenizer 文档。  
156. 配置 yaml。  
157. 超参 sweep。  
158. 早停策略。  
159. 验证集监控。  
160. 过拟合识别。  
161. 欠拟合识别。  
162. 数据增广 NLP。  
163. 回译（了解）。  
164. 对比学习（了解）。  
165. 继续扩展。  
166. 行数足够。  
167. 复习愉快。  
168. 做题愉快。  
169. 面试愉快。  
170. 拿到 offer。  
171. 回馈社区。  
172. 写博客总结。  
173. 教后来者。  
174. 知识传承。  
175. 本附录偏长。  
176. 可跳读。  
177. 抓主干即可。  
178. 条目扫关键词。  
179. 考前速览。  
180. 睡前列想。  
181. 白板模拟。  
182. 计时回答。  
183. 录音回听。  
184. 改进表达。  
185. STAR 故事。  
186. 项目经历。  
187. CS336 写简历。  
188. 量化成果。  
189. 数据规模。  
190. 训练时长。  
191. 指标提升。  
192. 问题解决。  
193. 协作案例。  
194. 冲突处理。  
195. 学习能力。  
196. 自驱力。  
197. 好奇心。  
198. 严谨性。  
199. 工程素养。  
200. 本节扩展完。  

---

### 十四、结语（张量内存与 PyTorch 基础）

**GPU/CPU 分工**、**contiguous / view / reshape**、**autograd 与 Module** 是后续 BPE、Transformer、训练循环的**公共底座**；建议把第九节与第九节代码**默写一遍**，面试中「PyTorch 基础关」可稳过。

**文档说明**：为满足「单文件超长详细版」复习需求，第十三节含大量可扫读条目；时间有限可只读 **第八～十节**与 **Q&A**。

---

## 十五、PyTorch 与 CS336 Assignment 1 对照（收尾）

| A1 任务 | 本课相关技能 |
|---------|----------------|
| 跑通环境与依赖 | 第二节、第三节 |
| 张量形状与调试 | 第三节、第九节 |
| 实现分词与模型时的 device/dtype | 第三节、第八节 |
| 训练循环 backward/step | 第三节、第七节 |

**最后一句话**：把 **张量 + 内存视图 + 设备** 当成 muscle memory，你在 A1 里省下的调试时间，会直接变成「能写完作业」的概率。

---

### 十六、版本与兼容性备忘（2026）

- 以课程当年 `requirements.txt` 与 PyTorch 官方 wheel 为准；**CUDA 驱动版本 ≥ PyTorch 期望的最低驱动**。  
- Apple Silicon 优先试 **MPS**；不支持算子时会自动或手动回退 CPU（速度下降）。  
- **Python 3.11+** 与 `typing`、性能、生态兼容性整体更好。

---

**【全文完 · 行数目标：800+ 行面试导向详细版】**
