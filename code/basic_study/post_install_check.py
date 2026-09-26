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
# 从 Python 列表创建；默认在 CPU，dtype 常为 float32
# x = torch.tensor([1.0, 2.0, 3.0])
# print(x.shape, x.dtype, x.device)

#
#
# B,T,D = 2,8,16 # 批大小、序列长度、隐藏维度
# a = torch.zeros(B,T,D) #全 0 张量,形状 (2,8,16)
# print(a.shape)
# print(a)
# b = torch.randn(B,T,D) # 标准正态分布随机张量
# print(b.shape)
# print(b)
# #
# # 根据环境选择设备（面试常写成一个函数）
# def pick_device() -> torch.device:
#     if torch.cuda.is_available(): # 有GPU走这条 → cuda
#         print("CUDA available:", torch.cuda.is_available())
#         print("GPU name:", torch.cuda.get_device_name(0))
#         return torch.device("cuda")
#     if torch.backends.mps.is_available(): # 苹果 Mac 走这条(你用不到)
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")  # 兜底

# device = pick_device()
#
# #这段代码创建了一个全1矩阵，并直接把它放到你事先定义好的设备（CPU或GPU）上，
# # 然后打印出它的位置，方便确认张量到底在哪。
# c = torch.ones(3, 4, device=device, dtype=torch.float32)
# print(c.device)
#
# x_gpu = c.to(device).to(torch.float64)
# print(x_gpu.device)  # cuda:0
# print(x_gpu.dtype)   # torch.float64
#
# 随机 numpy 数组，显式 float32 与 torch 常见训练精度一致
arr = np.random.randn(4,8).astype(np.float32)

# from_numpy：与 arr 共享底层内存，改一方可能影响另一方
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
# print(y[0,0]) #  999.0
#
# from typing import Tuple
# import torch
#
# # 导入 PyTorch 的神经网络模块，里面包含了所有神经网络的层（比如 Linear、Conv2d 等）
# import torch.nn as nn


# def split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
#     """
#     这个函数的作用：把输入张量重新排列，为"多头注意力机制"做准备
#
#     大白话解释：
#     输入是一个 3 维数据，形状是 (B, T, 总特征数)
#     我们要把它变成 4 维数据，形状是 (B, 头数, T, 每个头的特征数)
#
#     类比：就像把一个班级的学生（总特征）分成若干个小组（头数），
#          每个小组负责处理自己的那部分特征
#
#     参数：
#         x: 输入张量，形状为 (B, T, n_heads * head_dim)
#            - B: batch size（批次大小），可以理解为一次处理多少个独立的样本
#            - T: 序列长度（sequence length），比如一句话有多少个词
#            - 最后一个数字: n_heads * head_dim，即所有头的总特征维度
#
#         n_heads: 多头数量（要分成几个小组）
#         head_dim: 每个头的特征维度（每个小组处理多少特征）
#
#     返回：
#         重新排列后的张量，形状为 (B, n_heads, T, head_dim)
#     """
#
#     # 获取输入张量的形状，解包成三个变量
#     # b = batch size（批次大小）
#     # t = 序列长度（sequence length）
#     # c = 特征总数（即 n_heads * head_dim）
#     b, t, c = x.shape
#
#     # 检查一下：特征总数必须等于 头数 × 每个头的维度
#     # 如果不相等，说明数据格式不对，程序会报错停下来
#     # 比如：n_heads=4, head_dim=8，那么 c 必须等于 32
#     assert c == n_heads * head_dim
#
#     # 核心操作（分两步）：
#
#     # 第一步：使用 view() 改变形状，但不改变数据顺序
#     # 从 (B, T, n_heads*head_dim) 变成 (B, T, n_heads, head_dim)
#     # 就是把最后一大坨特征，按照"头数 × 每头维度"的方式重新分组
#     x = x.view(b, t, n_heads, head_dim)
#
#     # 第二步：使用 transpose() 交换维度位置
#     # 原来的顺序是 (批次, 序列长度, 头数, 每头维度)
#     # 我们想要 (批次, 头数, 序列长度, 每头维度)
#     # 所以把第1维（序列长度）和第2维（头数）交换位置
#     x = x.transpose(1, 2)
#
#     # 返回处理后的张量
#     return x
#
#
# class DummyModel(nn.Module):
#     """
#     这是一个简单的神经网络模型，用于演示如何自定义模型
#
#     大白话解释：
#     在 PyTorch 里，所有神经网络都要继承 nn.Module 这个类
#     就像你要做一个玩具，必须先有一个"玩具"的模子（nn.Module）
#
#     这个模型特别简单：输入什么维度，输出什么维度，中间只经过一个线性变换
#     就像一个"翻译器"，把数字从一种形式翻译成另一种形式，但大小不变
#     """
#
#     def __init__(self, d_model: int) -> None:
#         """
#         初始化函数：当创建这个模型时，会自动调用这个函数
#
#         参数：
#             d_model: 模型的维度，即输入和输出的特征数量
#                     比如 d_model=512，表示输入是512维，输出也是512维
#
#         大白话解释：
#         就像你要开一家工厂，需要先买好机器设备（这里就是买一个线性层）
#         """
#
#         # 调用父类 nn.Module 的初始化函数
#         # 这一行必须写，否则 PyTorch 无法正常管理这个模型
#         # 就像你开店必须先办营业执照一样，这是必须的手续
#         super().__init__()
#
#         # 创建一个线性层（全连接层），把它作为这个模型的"零件"
#         # nn.Linear(d_model, d_model) 的意思是：
#         # 输入维度是 d_model，输出维度也是 d_model
#         # 这个线性层做的事情：y = x * W + b
#         # 其中 W 是权重矩阵，b 是偏置项，它们都是模型需要学习的参数
#         # 把这个线性层存到 self.proj 里，这样模型就能记住自己有这个零件
#         self.proj = nn.Linear(d_model, d_model)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播函数：数据进入模型后，该怎么流动
#
#         大白话解释：
#         就像工厂的生产线，原材料（输入数据）进来后，
#         经过一道工序（self.proj 线性变换），
#         生产出产品（输出数据）再送出去
#
#         参数：
#             x: 输入张量，形状可以是任意的，但最后一维必须是 d_model
#
#         返回：
#             经过线性变换后的张量，形状和输入一样
#
#         注意：在 PyTorch 中，你通常不直接调用这个函数，
#         而是调用 model(x)，PyTorch 会自动调用 forward()
#         """
#
#         # 把输入 x 传给 self.proj（线性层），得到输出
#         # 相当于：output = x * W + b
#         # 然后把这个结果返回给调用者
#         return self.proj(x)


# #20260829-早
# # 导入 functools 模块，它提供了"装饰器"相关的工具函数
# # 装饰器就像给函数穿上一件"外衣"，在不修改原函数代码的情况下增加新功能
# import functools
# # 导入 time 模块，用来计时
# import time
# # 从 typing 导入类型提示相关的工具
# # Any: 任意类型, Callable: 可调用对象（函数）, TypeVar: 类型变量
# from typing import Any, Callable, TypeVar
#
# # 导入 PyTorch 核心库
# import torch
#
# # 定义一个类型变量 F，表示"可调用对象"（即函数）
# # bound=Callable[...] 限制 F 只能是函数类型，不能是 int、str 等其他类型
# # 这样装饰器就能精准保留被装饰函数的原始类型信息，IDE 会有正确的代码补全
# #
# # Callable[..., Any] 的含义：
# #   - ...（三个点）：表示函数的参数列表【任意】，数量和类型都不限制
# #   - Any：表示函数的返回值类型【任意】，可以是任何类型
# #   合起来就是：一个"参数和返回值都不限定"的函数类型
# F = TypeVar("F", bound=Callable[..., Any])
#
#
# def timeit(fn: F) -> F:
#     """
#     装饰器：给被装饰的函数添加"计时"功能
#
#     大白话：给函数戴上秒表，执行前开始计时，执行后打印耗时
#
#     参数：
#         fn: 要被装饰的函数
#
#     返回：
#         包装后的函数（原函数 + 计时功能）
#     """
#
#     # functools.wraps 保留原函数的名字、文档字符串等信息
#     @functools.wraps(fn)
#     def wrapper(*args: Any, **kwargs: Any) -> Any:
#         """
#         包装函数：在调用原函数前后加上计时逻辑
#
#         *args: 所有位置参数
#         **kwargs: 所有关键字参数
#         """
#
#         # 记录开始时间（time.perf_counter() 是高精度计时器，单位秒）
#         t0 = time.perf_counter()
#
#         # 调用原函数
#         out = fn(*args, **kwargs)
#
#         # 记录结束时间，计算并打印耗时（转换为毫秒）
#         t1 = time.perf_counter()
#         print(f"{fn.__name__}: {(t1 - t0) * 1000:.2f} ms")
#
#         # 返回原函数的执行结果
#         return out
#
#     return wrapper  # type: ignore[return-value]
#
#
# @torch.no_grad()
# def eval_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
#     """
#     模型推理函数（评估/测试阶段使用）
#
#     作用：
#     1. 把模型切换到评估模式（model.eval()）
#     2. 用模型做一次前向传播
#
#     好处：
#     - 关闭梯度计算，节省显存/内存
#     - 加快推理速度
#
#     参数：
#         model: PyTorch 模型
#         x: 输入数据（张量）
#
#     返回：
#         模型的预测结果（张量）
#     """
#
#     # 切换到评估模式
#     # 会影响 Dropout（关闭随机丢弃）和 BatchNorm（用全局统计量）等层的行为
#     model.eval()
#
#     # 前向传播（@torch.no_grad() 保证这里不会记录梯度）
#     return model(x)


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


# import torch
#
# w= torch.randn(10,1,requires_grad=True)
# x=torch.randn(1,10)
# y=(x@w).sum()
# y.backward()
# print(w.grad.shape)
# w.grad.zero_()
#
# # 导入 PyTorch 核心库
# import torch
# # 导入 PyTorch 神经网络模块
# import torch.nn as nn
#
#
# # ============================================================================
# # 自定义线性层（全连接层）
# # ============================================================================
# # 这是一个从零实现的全连接层，功能等同于 nn.Linear(in_features, out_features)
# #
# # 大白话：建立一个"翻译器"，输入 in_features 个数字，输出 out_features 个数字
# # 比如：输入4个特征（身高、体重、年龄、学历），输出2个值（收入预测、信用评分）
# # ============================================================================
#
# class TinyLinear(nn.Module):
#     """
#     自定义线性层（全连接层）
#
#     数学公式：y = x @ W.T
#     其中 W 是权重矩阵，形状为 (out_features, in_features)
#     """
#
#     def __init__(self, in_features: int, out_features: int):
#         """
#         初始化线性层
#
#         参数：
#             in_features:  输入特征数量（比如 4）
#             out_features: 输出特征数量（比如 2）
#
#         大白话：买好"翻译器"的零件（权重矩阵），准备开始工作
#         """
#
#         # 调用父类 nn.Module 的初始化（必须的"办营业执照"步骤）
#         super().__init__()
#
#         # ============================================================
#         # 创建权重矩阵（模型需要学习的参数）
#         # ============================================================
#         # nn.Parameter() 的作用：
#         #   - 把张量"包装"成模型参数
#         #   - 这样 model.parameters() 才能识别并收集它
#         #   - 训练时优化器会自动更新它
#         #
#         # 形状：torch.randn(out_features, in_features)
#         #   - 为什么是 (out_features, in_features)？
#         #     因为矩阵乘法时：x @ W.T
#         #     输入 x 形状: (batch, in_features)
#         #     权重 W 形状: (out_features, in_features)
#         #     W.T 转置后: (in_features, out_features)
#         #     这样 (batch, in_features) @ (in_features, out_features)
#         #     = (batch, out_features) ✅
#         #
#         # * 0.02：小随机初始化
#         #   - 标准正态分布（均值0，方差1）乘以 0.02
#         #   - 让初始值非常小（标准差只有0.02）
#         #   - 为什么？避免一开始数值太大导致"饱和"
#         #     （比如激活函数是 Sigmoid 或 Tanh 时，大数值会进入平坦区，梯度消失）
#         #   - 在真正的 nn.Linear 中，初始化策略更复杂（如 Kaiming 初始化）
#         self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播：数据流过这个层
#
#         参数：
#             x: 输入张量，形状为 (batch_size, in_features)
#                比如 (32, 4) 表示 32 个样本，每个 4 个特征
#
#         返回：
#             输出张量，形状为 (batch_size, out_features)
#                比如 (32, 2) 表示 32 个样本，每个 2 个输出值
#
#         计算公式：y = x @ W.T
#
#         为什么用 W.T（转置）？
#             输入 x: (batch, in_features)
#             权重 W: (out_features, in_features)  ← 我们存储的格式
#             W.T:   (in_features, out_features)   ← 转置后
#             x @ W.T: (batch, out_features)       ← 结果
#
#             如果不转置直接用 W：(batch, in_features) @ (out_features, in_features)
#             ❌ 矩阵乘法不合法（列数 in_features ≠ 行数 out_features）
#         """
#
#         # 矩阵乘法：输入 @ 权重的转置 = 输出
#         return x @ self.weight.T
#
#
# # ============================================================================
# # 使用示例
# # ============================================================================
#
# # 创建一个线性层：4个输入特征 → 2个输出特征
# m = TinyLinear(4, 2)
#
# # 打印参数个数
# # p.numel() 返回张量中元素的总个数（number of elements）
# # 权重矩阵是 (2, 4)，所以有 2 × 4 = 8 个参数
# # sum(...) 把所有参数的个数加起来
# print("参数个数:", sum(p.numel() for p in m.parameters()))
# # 输出: 参数个数: 8
#
#
# import torch
# from einops import rearrange,repeat
#
# B,T,HD =2,64,128
# H,D=8,16
# assert H*D==HD
# x= torch.randn(B,T,H*D)
#
# #命名维度：论文里常见的（B，T，H，D）拆分
# x_heads = rearrange(x,"b t (h d)->b h t d",h=H,d=D)
# print(x_heads.shape)
#
# y=torch.randn(3,1)
# y_rep=repeat(y,"a b->a (repeat b)",repeat=4)
# print(y_rep.shape)
#
# def param_memory_gb(num_params:int,bytes_per_param:int =4)->float:
#     """"仅权重占用 不含优化器状态与激活"""
#     return num_params*bytes_per_param/(1024**3)
#
# def matmul_flops_2mnk(m:int,n:int,k:int)->int:
#     return 2*m*n*k
#
# n=1_000_000_000
# print(f"1B 参数 FP32 权重约 {param_memory_gb(n, 4):.2f} GB")
# print(f"1B 参数 BF16 权重约 {param_memory_gb(n, 2):.2f} GB")
#
# M,N,K =4096,4096,4096
# print("4096^3 matmul FLOPs (2MNK):", matmul_flops_2mnk(M, N, K))
#
# # GPT-3 175B 参数
# gpt3_params = 175_000_000_000
# print(f"\nGPT-3 (175B 参数):")
# print(f"  FP32 权重显存: {param_memory_gb(gpt3_params, 4):.2f} GB")
# print(f"  BF16 权重显存: {param_memory_gb(gpt3_params, 2):.2f} GB")

"""
BPE 教学实现 — CS336 Lesson 02
逐行详细注释版本
"""
# 允许注解里使用还未定义的类型，老python版本兼容
# from __future__ import annotations
# 导入多进程库，用于并行统计语料pair频次
import multiprocessing as mp
# Counter用于统计频次；defaultdict带默认值的字典
from collections import Counter, defaultdict
# 类型注解，标明输入输出的数据结构
from typing import Dict, List, Optional, Sequence, Tuple
# regex第三方正则库，支持unicode属性\p{L}\p{N}，比内置re更强
import regex as re

# GPT‑2 的预分词正则：先按规则切块，再对每块做 BPE
# 把文本切为后缀、单词、数字、符号、空白，防止跨空格合并token
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# ============================================================
# 函数作用：GPT2预分词，把原始字符串切为独立片段chunk列表
# 阶段：【训练阶段、推理阶段 两者都用】
# 输入：text：待切分原始字符串
# 输出：List[str]，切分后的各个预分词片段
# ============================================================
def pretokenize_gpt2(text: str) -> List[str]:
    # finditer遍历正则匹配到的全部片段；m.group(0)拿到匹配到的字符串
    # 列表推导收集所有chunk返回
    return [m.group(0) for m in GPT2_SPLIT_PATTERN.finditer(text)]

# ============================================================
# 函数作用：单个字符串片段转UTF‑8字节id列表，每个字节映射为0‑255整数
# 阶段：【训练阶段、推理阶段 两者都用】
# 输入：chunk：一个预分词得到的字符串片段
# 输出：List[int]，字节id序列，全部取值0~255
# ============================================================
def bytes_to_ids(chunk: str) -> List[int]:
    # .encode("utf-8") 将字符串转为utf8字节bytes对象
    # list()把bytes拆解成一个个0~255的int，就是初始id序列
    return list(chunk.encode("utf-8"))

# ============================================================
# 函数作用：统计id序列内部所有相邻pair出现频次，可外部传入字典累加
# 阶段：【工具函数，训练阶段内部使用】
# 输入：ids：字节/子词id序列；counts：可选，外部传入的pair计数字典用于累加
# 输出：Dict[Tuple[int,int], int]，pair→出现次数的统计字典
# ============================================================
def get_stats(ids: Sequence[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    # 如果没有传入counts，新建空字典；否则复用传入的字典实现累加
    counts = counts if counts is not None else {}
    # 遍历ids，取每一组相邻id对
    for i in range(len(ids) - 1):
        # 取出相邻两个id，组成pair元组作为key
        pair = (ids[i], ids[i + 1])
        # dict.get(key,默认值)，key不存在返回0，再+1计数
        counts[pair] = counts.get(pair, 0) + 1
    # 返回统计好的pair频次字典
    return counts

# ============================================================
# 函数作用：序列内把所有相邻的(a,b)pair替换为new_id，从左到右非重叠替换
# 阶段：【工具函数，训练阶段、推理阶段都调用】
# 输入：ids：待处理id列表；pair：待合并(a,b)；new_id：合并后的新token id
# 输出：List[int]，执行替换之后新的id列表
# ============================================================
def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    # 解包待合并的两个id
    a, b = pair
    # 新建空列表存放输出结果
    out: List[int] = []
    # i循环游标；n是输入序列总长度
    i, n = 0, len(ids)
    # while循环遍历整个ids序列
    while i < n:
        # 判断：不是最后一个元素，并且当前位置正好匹配(a,b)待合并pair
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            # 匹配成功，写入合并后的new_id
            out.append(new_id)
            # 向后跳2位，跳过已经合并的两个元素（非重叠）
            i += 2
        else:
            # 不匹配，直接把当前id原样写入输出
            out.append(ids[i])
            # 游标向后走1位
            i += 1
    # 返回完成合并后的新id序列
    return out

# ============================================================
# 函数作用：基于chunk频次加权统计pair；每个chunk_tuple乘上自身出现频次统计pair
# 阶段：【工具函数，训练阶段train_bpe内部调用】
# 输入：chunk_freqs：Dict[Tuple[int,...], int] key=chunk的id元组 value=该片段出现频次
# 输出：Dict[Tuple[int,int], int]，全局pair加权计数字典
# ============================================================
def count_pairs_for_chunks(chunk_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    # 初始化空字典保存pair统计结果
    stats: Dict[Tuple[int, int], int] = {}
    # 遍历每一个chunk元组，以及该chunk在语料中出现的频次freq
    for chunk_tuple, freq in chunk_freqs.items():
        # 元组转list得到id序列
        ids = list(chunk_tuple)
        # 遍历这个chunk内部全部相邻pair
        for i in range(len(ids) - 1):
            # 取出相邻id对
            p = (ids[i], ids[i + 1])
            # 加权累加：这个chunk出现freq次，所以pair计数 += freq，而不是+1
            stats[p] = stats.get(p, 0) + freq
    # 返回加权统计后的pair频次
    return stats

# ============================================================
# 函数作用：BPE主训练逻辑，输入原始语料，迭代多轮merge生成vocab与merges表
# 阶段：【训练阶段】
# 输入：text_corpus：训练原始大语料字符串；num_merges：要执行多少轮合并；pattern：预分词正则
# 输出：(vocab, merges)
#       vocab:Dict[int,bytes] key=token_id value=对应字节；merges:List[(left,right,new_id)]合并规则列表
# ============================================================
def train_bpe(text_corpus: str, num_merges: int, pattern: re.Pattern = GPT2_SPLIT_PATTERN):
    # 字典：key是chunk的id元组，value是该片段出现次数；Counter专门做计数
    chunk_freqs: Dict[Tuple[int, ...], int] = Counter()
    # 正则遍历语料所有预分词chunk
    for m in pattern.finditer(text_corpus):
        # 将chunk字符串转为字节id列表，再转元组（list不能做字典key，tuple可以）
        t = tuple(bytes_to_ids(m.group(0)))
        # 该chunk计数+1
        chunk_freqs[t] += 1

    # 初始化词表：0~255分别对应单个原始字节
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # merges保存所有合并规则：(左id,右id,新生成id)
    merges: List[Tuple[int, int, int]] = []
    # 第一个新token id从256开始，0‑255是原始字节
    next_id = 256

    # 循环num_merges次，执行指定轮数的BPE合并
    for _ in range(num_merges):
        # 加权统计当前所有chunk上的pair频次
        pair_stats = count_pairs_for_chunks(chunk_freqs)
        # 如果没有任何pair可以合并，提前退出循环
        if not pair_stats:
            break
        # 频次相同时按实际 token bytes 的字典序取最大的 pair，不按数值 id 排序。
        best_pair = max(
            pair_stats,
            key=lambda pair: (pair_stats[pair], vocab[pair[0]], vocab[pair[1]]),
        )
        # 解包本轮要合并的两个id
        left, right = best_pair

        # defaultdict(int) key不存在时默认value=0，用于保存合并之后新chunk的频次
        new_chunk_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
        # 遍历全部旧chunk，逐个执行merge
        for chunk_tuple, freq in chunk_freqs.items():
            # 对当前chunk执行一次best_pair合并
            merged = merge(list(chunk_tuple), best_pair, next_id)
            # 合并后的id列表转为tuple，累加到新的频次字典，带上原来的freq
            new_chunk_freqs[tuple(merged)] += freq
        # 更新chunk_freqs为本轮全部合并完成后的结果
        chunk_freqs = dict(new_chunk_freqs)

        # 更新词表：新token的bytes等于左右两个token的bytes拼接
        vocab[next_id] = vocab[left] + vocab[right]
        # 将本轮合并规则记录进merges列表
        merges.append((left, right, next_id))
        # id自增，下一轮使用下一个id
        next_id += 1
    # 返回训练产物：词表、合并规则
    return vocab, merges

# ============================================================
# 函数作用：构建pair→rank字典，rank越小代表越早训练生成，推理要优先合并
# 阶段：【工具函数，推理阶段使用】
# 输入：merges：训练输出的合并规则列表 List[(a,b,nid)]
# 输出：Dict[Tuple[int,int], int] pair对应合并优先级rank
# ============================================================
def build_merge_ranks(merges: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    # enumerate拿到merges下标r(就是rank)和每一条(a,b,_)；pair做key，rank做value
    return {(a, b): r for r, (a, b, _) in enumerate(merges)}

# ============================================================
# 函数作用：单chunk BPE合并实现；每轮扫描选rank最小pair合并；同rank取最左位置
# 阶段：【推理阶段】
# 输入：ids：单个预分词chunk的原始字节id列表；merges：训练产出合并规则
# 输出：List[int]，BPE合并完成后的子词token id序列
# ============================================================
def encode_piece_by_rank(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    # 构建pair到new_id的映射，方便查到合并后的id
    pair_to_new = {(a, b): nid for a, b, nid in merges}
    # 构建合并优先级rank字典
    merge_ranks = build_merge_ranks(merges)
    # 拷贝输入序列，不修改原输入
    seq = ids[:]

    # 循环：不断找当前序列中可以合并的最优pair，直到没有可合并pair
    while True:
        # best_rank记录当前最优合并的优先级；pos记录位置，初始None代表无
        best_rank, pos = None, None
        # 遍历当前seq，查找全部可合并pair
        for i in range(len(seq) - 1):
            # 取出当前相邻pair
            p = (seq[i], seq[i + 1])
            # 如果这个pair不在merges规则里，跳过
            if p not in merge_ranks:
                continue
            # 获取该pair的合并优先级rank
            r = merge_ranks[p]
            # 如果还没有选pair，或者当前pair rank更小，或者rank相等位置更靠左，则更新最优
            if best_rank is None or r < best_rank or (r == best_rank and pos is not None and i < pos):
                best_rank, pos = r, i
        # best_rank/pos为None，代表没有可以合并的pair，跳出while循环
        if best_rank is None or pos is None:
            break
        # 获取要合并的pair
        p = (seq[pos], seq[pos + 1])
        # 根据pair查到合并后的new_id
        new_id = pair_to_new[p]
        # 切片替换：pos位置两个元素替换成new_id，生成新序列
        seq = seq[:pos] + [new_id] + seq[pos + 2 :]
    # 返回反复合并完成后的token id序列
    return seq

# ============================================================
# 函数作用：单chunk BPE合并实现，按merges列表顺序逐条执行全部合并规则
# 阶段：【推理阶段】
# 输入：ids：单个预分词chunk原始字节id列表；merges：训练产出合并规则
# 输出：List[int]，BPE合并完成后的子词token id序列
# ============================================================
def encode_piece_sequential(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    # 拷贝输入序列，不改动外部传入的ids
    seq = ids[:]
    # 按merges训练生成的顺序，逐条取出合并规则
    for left, right, new_id in merges:
        # 调用merge函数：在整个seq上把所有(left,right)替换成new_id，得到新seq
        seq = merge(seq, (left, right), new_id)
    # 全部规则应用完毕，返回结果
    return seq

# ============================================================
# 函数作用：完整文本BPE编码对外入口：预分词→逐chunk转字节id→逐chunkBPE合并→拼接结果
# 阶段：【推理阶段】
# 输入：text：待编码原始字符串；merges：训练得到合并规则；pattern：GPT2预分词正则
# 输出：List[int]，整篇文本编码后的完整token id序列
# ============================================================
def bpe_encode(text: str, merges: List[Tuple[int, int, int]], pattern: re.Pattern = GPT2_SPLIT_PATTERN) -> List[int]:
    # 初始化空列表保存整篇文本所有token id
    out: List[int] = []
    # 对输入文本做GPT‑2预分词，遍历每一个chunk片段
    for piece in (m.group(0) for m in pattern.finditer(text)):
        # piece字符串转字节id列表；调用encode_piece_sequential做BPE合并；extend把结果追加到out
        out.extend(encode_piece_sequential(bytes_to_ids(piece), merges))
    # 返回整篇文本的token id序列
    return out

# ============================================================
# 函数作用：BPE解码，token id序列还原回原始字符串
# 阶段：【推理阶段】
# 输入：ids：待解码token id序列；vocab：词表字典 key=token_id value=bytes
# 输出：str，还原得到字符串；非法字节自动替换为�不抛异常
# ============================================================
def bpe_decode(ids: Sequence[int], vocab: Dict[int, bytes]) -> str:
    # 1.遍历ids，每个id查表vocab拿到对应的bytes对象
    # 2.b"".join把所有小bytes拼接成一整个大bytes
    # 3.decode("utf‑8")转为字符串；errors="replace"遇到非法字节不会崩溃，替换为�符号
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# ============================================================
# 函数作用：多进程worker子进程，统计给定一批文本行内部pair局部频次
# 阶段：【训练阶段，仅被parallel_count_pairs调用】
# 输入：lines：分给当前worker的一批文本行；pattern：预分词正则
# 输出：Dict[Tuple[int,int], int]，本分片pair局部计数字典
# ============================================================
def worker_count_pairs(lines: List[str], pattern: re.Pattern) -> Dict[Tuple[int, int], int]:
    # 本worker分片内pair的局部计数字典
    local: Dict[Tuple[int, int], int] = {}
    # 遍历分配给本进程的每一行文本
    for line in lines:
        # 正则预分词，遍历该行切分出的每一个chunk
        for m in pattern.finditer(line):
            # chunk字符串转为字节id列表
            ids = bytes_to_ids(m.group(0))
            # 遍历chunk全部相邻id对
            for i in range(len(ids) - 1):
                # 取出pair
                p = (ids[i], ids[i + 1])
                # 局部计数+1
                local[p] = local.get(p, 0) + 1
    # 返回本分片统计结果，交给主进程汇总
    return local

# ============================================================
# 函数作用：多进程并行统计全语料pair频次；切分语料分片给worker，汇总各进程计数
# 阶段：【训练阶段】
# 输入：corpus_lines：全部训练语料按行组成list；num_workers：进程数量；pattern：预分词正则
# 输出：Dict[Tuple[int,int], int]，全语料汇总pair全局频次字典
# ============================================================
def parallel_count_pairs(
    corpus_lines: List[str], num_workers: int = 4, pattern: re.Pattern = GPT2_SPLIT_PATTERN
) -> Dict[Tuple[int, int], int]:
    # 如果进程数≤1，不走多进程开销，直接单线程执行
    if num_workers <= 1:
        return worker_count_pairs(corpus_lines, pattern)
    # 计算每个分片分配多少行；max(1, ...)防止总行数小于worker数时分片大小为0
    chunk_size = max(1, len(corpus_lines) // num_workers)
    # 将全部语料切分成多个分片shard，每个shard交给一个worker
    shards = [corpus_lines[i : i + chunk_size] for i in range(0, len(corpus_lines), chunk_size)]
    # 创建进程池，with退出时自动关闭进程池，释放资源
    with mp.Pool(num_workers) as pool:
        # starmap把每个shard和pattern传入worker_count_pairs，并行执行，收集各个worker返回结果
        parts = pool.starmap(worker_count_pairs, [(s, pattern) for s in shards])
    # 主进程全局总pair计数字典
    total: Dict[Tuple[int, int], int] = {}
    # 遍历每一个worker返回的局部统计字典
    for d in parts:
        # 遍历该worker的每一组(pair,频次)
        for k, v in d.items():
            # 不同worker相同pair的计数累加，得到全局总频次
            total[k] = total.get(k, 0) + v
    # 返回汇总完成的全局pair频次字典
    return total

# ============================================================
# 脚本主入口演示：训练BPE模型，再编码解码，断言校验无损还原
# 阶段：【演示脚本，训练+推理链路完整跑通做验证】
# 输入：脚本内置sample测试字符串，无外部入参
# 输出：控制台打印token ids与OK；断言失败抛出AssertionError，无函数返回值
# ============================================================
if __name__ == "__main__":
    # 测试样例文本，包含重复英文、中文，覆盖ASCII与UTF‑8多字节字符
    sample = "hello hello hello 你好"
    # 执行BPE训练，做10轮合并，得到词表vocab、合并规则merges
    vocab, merges = train_bpe(sample, num_merges=10)
    # 使用训练好的merges对sample文本做推理编码，得到token id序列
    ids = bpe_encode(sample, merges)
    # 解码id序列还原字符串；断言：解码结果必须等于原始sample，不等直接抛AssertionError
    assert bpe_decode(ids, vocab) == sample
    # 在控制台打印编码得到的token ids
    print("token ids:", ids)
    # 断言没有报错，打印OK，代表整套链路运行正常
    print("OK")
