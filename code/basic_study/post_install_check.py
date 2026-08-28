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