import torch
import numpy as np

# # 打印版本，确认与项目要求一致
# print("torch version:", torch.__version__)
#
# # NVIDIA GPU：需要正确安装 CUDA 版 PyTorch 与驱动
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU name:", torch.cuda.get_device_name(0))
#
# # Apple Silicon：常用 MPS 后端
# print("MPS available:", torch.backends.mps.is_available())
#
#
# # 从 Python 列表创建；默认在 CPU，dtype 常为 float32
# x = torch.tensor([1.0, 2.0, 3.0])
# print(x.shape, x.dtype, x.device)
#
#
#
# B,T,D = 2,8,16 # 批大小、序列长度、隐藏维度
# a = torch.zeros(B,T,D) #全 0 张量,形状 (2,8,16)
# b = torch.randn(B,T,D) # 标准正态分布随机张量
#
# # 根据环境选择设备（面试常写成一个函数）
# def pick_device() -> torch.device:
#     if torch.cuda.is_available(): # 有GPU走这条 → cuda
#         return torch.device("cuda")
#     if torch.backends.mps.is_available(): # 苹果 Mac 走这条(你用不到)
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")  # 兜底
#
# device = pick_device()
#
# #这段代码创建了一个全1矩阵，并直接把它放到你事先定义好的设备（CPU或GPU）上，
# # 然后打印出它的位置，方便确认张量到底在哪。
# c = torch.ones(3, 4, device=device, dtype=torch.float32)
# print(c.device)
#
# x_gpu = x.to(device).to(torch.float64)
# print(x_gpu.device)  # cuda:0
# print(x_gpu.dtype)   # torch.float64

# 随机 numpy 数组，显式 float32 与 torch 常见训练精度一致
arr = np.random.randn(4,8).astype(np.float32)

# # from_numpy：与 arr 共享底层内存，改一方可能影响另一方
# x = torch.from_numpy(arr)
# print(x[0,0])
# arr[0,0]=999.0
# print(x[0,0]) # 可能也是 999.0，演示共享内存
#
# # 需要独立副本时用 torch.tensor 或 clone
# y = torch.tensor(arr)
# print(y[0,0])
# arr[0,0]=888.0
#
# # 不受后续 arr 修改影响（取决于是否仍共享，tensor(arr) 一般为拷贝）
# print(y[0,0]) # 888.0

from typing import Tuple
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


#20260829-早
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


# #Broadcasting
# a = torch.randn(32,1,128)
# b =torch.randn(128)
# c = a + b
# print(f"a 的形状: {a.shape}")      # torch.Size([32, 1, 128])
# print(f"b 的形状: {b.shape}")      # torch.Size([128])
# print(f"c 的形状: {c.shape}")      # torch.Size([32, 1, 128])
#
# logits =torch.randn(4,10,50257)
# bias=torch.randn(50257)
# bias = bias.view(1,1,-1)
# out =logits+bias
# print(f"\nlogits 的形状: {logits.shape}")   # torch.Size([4, 10, 50257])
# print(f"bias 的形状:   {bias.shape}")     # torch.Size([1, 1, 50257])
# print(f"out 的形状:    {out.shape}")      # torch.Size([4, 10, 50257])


import torch

w= torch.randn(10,1,requires_grad=True)
x=torch.randn(1,10)
y=(x@w).sum()
y.backward()
print(w.grad.shape)
w.grad.zero_()

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