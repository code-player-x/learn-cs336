# Lesson 09：GPU 架构与内存层级

> **Stanford CS336**：Language Modeling from Scratch — 面向面试的体系化笔记（第 09 节）

**本节定位**：从 **CPU / GPU 设计差异** 与 **SIMT 并行模型** 出发，系统梳理 **GPU 硬件（SM、CUDA Core、Tensor Core、Warp）** 与 **内存层级**（寄存器 → 共享内存 → L1/L2 → HBM → 主机 DRAM），建立 **算术强度、Roofline、计算/访存瓶颈** 的分析框架；结合 **A100 / H100 / H200** 规格对照、**Profiling** 工具链（`torch.profiler`、Nsight、py-spy）与 **MFU / 带宽利用率** 等性能指标；延伸到 **PyTorch CUDA API**、**训练显存估算** 与 **激活值重计算（activation checkpointing）**，为 **FlashAttention**（Lesson 10）打好访存直觉。

**先修**：并行计算基本概念、PyTorch 张量与 `cuda` 设备（Lesson 01 及前序实验）。

**面试热度**：★★★★★（极高频：体系结构 + 性能 + 显存 + 工具链）

---

## 概念讲解

本节按「**为什么用 GPU → 硬件长什么样 → 内存有多快/多大 → 怎么判断瓶颈 → 卡怎么选 → 怎么量 → PyTorch 怎么用 → 显存怎么估**」的顺序展开，尽量用**初学者友好**的语言，数字以**数据中心常见 SKU 的教学量级**为准，精确值以 NVIDIA 官方 datasheet 与实测为准。

### 1. 为什么深度学习主要用 GPU：并行度与 SIMT

**（1）问题形态匹配**  
Transformer 与卷积网络的主体是 **稠密线性代数**：大矩阵乘（GEMM）、批量矩阵乘（BMM）、归约、逐元素运算。这类 workload 的共性是：**同一套指令要对海量数据重复执行**，天然适合 **单指令、多数据** 风格的硬件。

**（2）吞吐量优先，而非单线程延迟**  
CPU 为 **操作系统、分支预测、缓存层次** 优化，追求 **少量强核、低延迟**。GPU 则堆叠 **大量较简单的执行单元**，用 **海量并发** 把 **内存访问延迟**「藏」在别的线程/ warps 后面（latency hiding）。对 LLM 训练而言，**总吞吐（tokens/s、samples/s）** 比单线程延迟更重要。

**（3）SIMT（Single Instruction, Multiple Threads）**  
NVIDIA CUDA 设备上，**同一时刻、同一指令** 往往由 **一组线程** 共同执行，这组线程的典型调度单位是 **Warp（32 个线程）**。可以粗略理解为：**一条指令广播给 32 个线程**，每个线程用自己的 **寄存器** 操作自己的数据。这与 CPU 的 SIMD（向量指令）有相似之处，但 **SIMT 更强调「大量独立线程 + 分支掩码」**，编程模型上是 **显式并行线程**。

**面试一句话**：GPU 适合深度学习，是因为 **数据并行度高 + 计算密集块（尤其 GEMM）与 Tensor Core 匹配**，SIMT / Warp 模型把 **并行执行** 固化在硬件与 ISA 层面。

### 2. GPU 硬件架构概览

#### 2.1 Streaming Multiprocessors（SM）

**SM** 是 NVIDIA GPU 上 **调度与执行的基本「集群」**：每个 SM 内有 **Warp 调度器、寄存器文件、共享内存、L1（常与共享内存共享部分存储体）**，以及 **CUDA Core、Tensor Core、Load/Store 单元** 等。  
**多 SM 并行** 构成整张卡的 **大规模并行**；**Occupancy（占用率）** 指每个 SM 上 **活跃 warps / 最大可驻留 warps** 的比例，与 **寄存器用量、共享内存用量、block 大小** 等相关。

**初学者直觉**：把 **SM** 想成「**许多工位组成的车间**」，每个车间同时处理 **多个 warp**；车间总数 × 每车间吞吐 ≈ 整卡吞吐。

#### 2.2 CUDA Core vs Tensor Core

| 单元 | 角色 | 面试怎么说 |
|------|------|------------|
| **CUDA Core** | **标量** FP32/FP64 等通用浮点 ALU，数量多 | 负责 **通用逐元素、非矩阵乘** 类运算，以及部分 **FP32 路径** |
| **Tensor Core** | **矩阵乘累加** 专用（GEMM 类），从 Volta 起成为数据中心 GPU 标配 | **大矩阵乘、混合精度训练** 的峰值 TFLOPS 主要来自 Tensor Core |

**要点**：**大规格 GEMM** 在 **FP16/BF16/TF32/FP8** 等路径上走 Tensor Core，**有效 TFLOPS** 接近规格表峰值；**纯逐元素** 或 **小矩阵** 可能更多受 **CUDA Core / 访存** 限制。

#### 2.3 Warp（32 线程）执行模型

- **调度粒度**：一个 **warp = 32 个线程**，是 **指令发射与执行** 的常见粒度（同一 warp 同一 PC，除非分支发散）。
- **分支发散（divergence）**：若 warp 内线程走 **不同分支**，硬件用 **掩码** 串行化不同路径，**有效利用率下降**。因此 GPU **不鼓励深度嵌套、不规则分支**。
- **合并访存（coalescing）**：若一个 warp 内线程访问的地址 **可被合并成少量 cache line / segment**，则 **有效带宽** 高；随机或不规则访问则 **带宽利用率** 差。

**与 FlashAttention 的联系（预告）**：手写 kernel 时常通过 **tiling** 让 **一个 block 内线程协作**，把 **Q/K/V 分块** 放进 **共享内存或寄存器**，减少对 **HBM** 的往返——这正是 **理解内存层级** 后读 FlashAttention 的钥匙。

### 3. 内存层级（理解 FlashAttention 的前置知识）

深度学习 kernel 的性能，往往由 **「数据放在哪、搬了多少次」** 决定。下面自快到慢、自小到大梳理；**容量与带宽为常见教学锚点**，不同架构/SKU 会变化。

#### 3.1 寄存器（Registers）

- **每线程私有**，容量 **极小（通常 KB 量级/线程，由 ISA 与编译器决定）**，**延迟最低**，紧挨 ALU。
- **寄存器溢出** 时，编译器可能把变量放到 **Local Memory**（往往落到 **片外**，性能骤降），或削减 **Occupancy**。

#### 3.2 共享内存 Shared Memory / 片上 SRAM

- **同一线程块（block）内可见**，速度 **远快于 HBM**，带宽可达 **约 10～20 TB/s 量级**（教学中常记 **~19 TB/s** 作为 **片上 SRAM 路径** 的峰值锚点）。
- **单 SM 可用容量** 常见 **约 128～228 KB 量级**（依架构与配置而定，且与 L1 划分有关——面试说 **「一百多 KB / SM 量级」** 即可）。
- **用途**：CUDA 手写分块 GEMM、规约、FlashAttention 类 **tile 数据驻留**，以及 **融合 kernel** 的中间结果。

#### 3.3 L1 / L2 Cache

- **L1**：通常 **更贴近 SM**，容量 **较小**；与 **共享内存** 在部分架构上 **共享存储体**，需查具体白皮书。
- **L2**：**全 GPU 共享**，容量 **数十 MB 量级**，位于 SM 与 **HBM** 之间，缓解 **重复访问** 与 **部分不规则访问**。

#### 3.4 全局显存 Global Memory / HBM

- **容量**：数据中心卡常见 **40～80 GB** 等；**H200** 等 SKU 可达 **约 141 GB 量级**。
- **带宽**：**H100** 常见峰值讨论量级 **约 1.5～3.35 TB/s**（依 SKU、HBM 代际而定）；**A100** 常见 **约 1.5～2 TB/s 量级**。
- **地位**：**容量最大、相对最慢的一级「主存」**；**memory-bound** kernel 的最终瓶颈常常落在 **HBM 有效带宽**（而非峰值 TFLOPS）。

#### 3.5 CPU DRAM（主机内存）

- **容量**：服务器常见 **512 GB～2 TB** 等（视配置）。
- **带宽**：内存本身 **数十～数百 GB/s**；但 **GPU 与 CPU 之间** 经 **PCIe** 或 **NVLink**，**有效端到端带宽** 常见讨论量级 **约 50～400 GB/s**（强依赖链路代际、是否 pinned、是否异步流水线重叠）。

### 4. 内存层级对照表（容量与带宽）

| 层级 | 典型容量量级 | 典型带宽量级 | 延迟直觉 |
|------|----------------|----------------|----------|
| **寄存器** | 每线程 **极少量（KB 级以下/线程）** | **极高（随执行单元）** | **最低** |
| **Shared Memory / SRAM** | **~128～228 KB / SM**；全卡合计 **~十 MB 量级** | **~10～20 TB/s（如 ~19 TB/s 锚点）** | **很低** |
| **L1** | **每 SM 较小** | **高** | **低** |
| **L2 Cache** | **~数十 MB（全卡）** | **介于 Shared 与 HBM 之间** | **低～中** |
| **HBM（全局显存）** | **~40～80 GB（常见）；更大 SKU 如 ~141 GB** | **~1.5～3.35 TB/s（依代际）** | **相对片上更高** |
| **CPU DRAM** | **数百 GB～TB** | **内存本体 ~50～400 GB/s；跨 GPU 有效常更低** | **跨设备最高** |

**记忆口诀**：**越靠近 ALU，越小越快；HBM 大但仍远慢于片上 SRAM 路径；跨到 CPU 往往最慢。**

### 5. 计算受限 vs 访存受限（Compute-bound vs Memory-bound）

#### 5.1 算术强度（Arithmetic Intensity）

\[
\text{Arithmetic Intensity} = \frac{\text{FLOPs（或有效计算量）}}{\text{Bytes Transferred（与实现相关的内存流量）}}
\]

单位常用 **FLOPs/Byte**：含义是 **每从内存体系搬运 1 字节，平均做多少次浮点运算**。  
**强度越高**，越可能 **吃满算力**；**强度越低**，越可能 **吃满内存带宽**。

#### 5.2 Roofline 模型

- **算力屋顶**：由 **Tensor Core / CUDA Core 峰值 TFLOPS** 决定（Roofline 图上为 **水平线**）。
- **带宽屋顶**：由 **有效内存带宽** 决定；在 **强度–性能** 平面上是 **过原点的斜线**（斜率与 TB/s 相关）。

对给定 Kernel：**可达性能 ≈ min(算力屋顶, 强度 × 有效带宽)**。  
若实际性能 **贴近算力水平线** → **compute-bound（计算密集型）**；若 **贴近带宽斜线** → **memory-bound（访存密集型）**。

#### 5.3 为什么 Attention（朴素实现）常是 memory-bound

- **朴素 Attention** 往往产生 **大尺寸中间张量**（如 \(O(B \cdot H \cdot T^2)\) 量级），并多次 **读写 HBM**。
- 相对 **GEMM**，**每字节对应的有效 FLOPs** 不够高，或 **实现上 IO 次数过多**，容易 **先触及 HBM 带宽** 而非 Tensor Core 峰值。
- **FlashAttention** 通过 **分块、融合、重算**，**减少 HBM 往返**，把瓶颈向 **计算** 方向推——这是 Lesson 10 的核心动机之一。

### 6. GPU 规格对比：A100 vs H100 vs H200

| 项目 | **A100（典型 80GB SXM）** | **H100（典型 80GB SXM）** | **H200（典型）** |
|------|---------------------------|----------------------------|------------------|
| **架构** | Ampere | Hopper | Hopper 系（更大 HBM） |
| **显存** | **80GB HBM2e**（常见讨论） | **80GB HBM3** | **更大容量（如 ~141GB 级 SKU）** |
| **显存带宽** | **~2 TB/s 量级** | **~3.35 TB/s 量级** | **更高（代际提升）** |
| **Tensor Core** | 第三代 | **第四代**（FP8 等） | 在 H100 基础上强化大模型/长上下文场景 |
| **面试表述** | 上一代训练主力 | **算力 + HBM 带宽** 相对 A100 全面提升 | **更大显存 + 更高带宽**，显存敏感 workload 友好 |

### 7. Profiling 工具：torch.profiler、Nsight、py-spy

| 工具 | 解决什么问题 | 典型用法 |
|------|----------------|----------|
| **`torch.profiler`** | Python / ATen **算子级** 时间线、CUDA kernel 名称、可选显存 | `profile(activities=[CPU, CUDA])`，`key_averages()` 找热点 |
| **NVIDIA Nsight Systems** | **系统级** timeline：CPU 线程、CUDA API、kernel、NVLink、PCIe、D2H/H2D | 找 **流水线气泡**、**数据加载 vs 计算** 是否重叠 |
| **NVIDIA Nsight Compute** | **单 kernel** 微观：occupancy、内存事务、warp 效率、指令吞吐 | 针对 **关键 kernel** 深度优化 |
| **py-spy** | **Python 层采样**，低开销 | 找 **GIL、Python 热点、DataLoader 主线程** 等 **CPU 侧** 瓶颈；与 GPU profiler **互补** |

**注意**：GPU 内核 **异步**；计时时务必在区间边界 **`torch.cuda.synchronize()`**，否则会 **严重低估** GPU 时间。

### 8. 性能分析：MFU、带宽利用率与瓶颈识别

#### 8.1 FLOPS 利用率与 MFU（Model FLOPS Utilization）

- **MFU** 定义为：**模型一次前向+反向（或指定步）的实际 achieved FLOPS** 与 **GPU 峰值 FLOPS** 的比值（有时按 **理论模型 FLOPs/步** 与 **墙钟时间** 估算 achieved）。
- **意义**：衡量 **算法+实现** 是否 **吃满硬件算力**；大模型训练中 **MFU 低** 可能来自 **访存、通信、小 kernel launch、低 occupancy** 等。

#### 8.2 内存带宽利用率

- **有效 TB/s** / **峰值 TB/s**：若 **长期接近带宽顶** 且 **强度偏低**，说明 **memory-bound**。
- **Nsight** 可看 **内存事务、L2 命中、HBM 吞吐**；与 **Roofline** 对照。

#### 8.3 瓶颈识别（实操顺序）

1. **端到端**：Nsight Systems 看 **是否有大段 CPU 等待、拷贝是否与计算重叠**。
2. **GPU 热点**：`torch.profiler` 看 **哪些 op / kernel 占时**。
3. **单 kernel**：Nsight Compute 看 **occupancy、内存模式**。
4. **模型级**：估算 **算术强度** 与 **MFU**，判断 **算力顶还是带宽顶**。

### 9. PyTorch GPU 操作：设备与 `torch.cuda` API

- **设备**：`torch.device("cuda")`、`cuda:0`，多卡 `cuda:1` …
- **当前设备**：`torch.cuda.current_device()`；**设默认**：`torch.cuda.set_device(i)`。
- **同步**：`torch.cuda.synchronize()`。
- **显存统计**：`torch.cuda.memory_allocated()`、`max_memory_allocated()`、`memory_reserved()`；**快照**：`memory_summary()`（调试时有用）。
- **缓存**：`torch.cuda.empty_cache()` **只影响 PyTorch 缓存分配器**，**不保证** 立刻向 OS/驱动归还 **nvidia-smi** 可见显存。
- **流**：`torch.cuda.Stream` 用于 **异步并发**（高级用法）。

### 10. 训练显存估算：参数 + 梯度 + 优化器 + 激活

**（1）静态部分（以 FP32 训练为例，单位：字节/参数）**

| 组成部分 | FP32 估算 |
|----------|------------|
| **参数** | 4 B |
| **梯度** | 4 B |
| **Adam m** | 4 B |
| **Adam v** | 4 B |

**合计约 16 B/参数**（仅权重相关静态张量）。再加上 **框架 buffer、梯度聚合、碎片**，教学常记 **约 16～20× 参数字节数** 为 **粗算上界量级**。

**（2）激活（Activations）**  
与 **batch、序列长度、隐藏维、层数、是否 checkpoint、是否重计算 attention** 强相关；常占 **大头**，且 **梯度检查点（activation checkpointing）** 用 **重算前向** 换 **存更少的激活**，是 **用计算换显存**。

**（3）混合精度与 ZeRO/FSDP**  
**BF16 权重**、**FP32 master weight**、**分片优化器状态** 等会 **显著改变** 上述估算；面试要能说出 **定性方向**，不必背死每一个变体数字。

### 11. 激活值重计算（Activation Checkpointing）原理（简）

- **前向** 时 **不保存** 部分中间激活，**反向** 时 **重新计算** 这些激活以求梯度。
- **代价**：**前向计算量增加**（约 **增加 0.33～1× 前向** 量级，依分段策略而异）。
- **收益**：**峰值激活显存下降**，使 **更大 batch 或更长序列** 成为可能。与 **FlashAttention** 一样，体现 **用算力换带宽/存储** 的系统思维。

---

## 代码示例

### 1. 设备与同步

```python
import torch

assert torch.cuda.is_available()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

x = torch.randn(1024, 1024, device=device)
torch.cuda.synchronize()
```

### 2. 正确计时：`torch.cuda.synchronize()`

```python
import torch
import time

def bench(fn, repeats=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats
```

### 3. `torch.profiler` 最小示例

```python
import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

def step():
    return torch.matmul(x, w)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(8):
        step()
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 4. 显存查询与 `empty_cache()`

```python
import torch

x = torch.randn(1024, 1024, device="cuda")
print("allocated MiB:", torch.cuda.memory_allocated() / 1024**2)
print("max MiB:", torch.cuda.max_memory_allocated() / 1024**2)

del x
torch.cuda.empty_cache()
```

### 5. 算术强度与 Roofline 示意（教学用）

```python
import torch

def roofline_hint(flops, bytes_moved, peak_tflops=300, peak_tbs=3.0):
    """
    flops: 算子浮点操作数估计
    bytes_moved: 读写字节总量估计（与实现强相关）
    peak_tflops: 峰值算力（TFLOPS）
    peak_tbs: 峰值内存带宽（TB/s）
    """
    intensity = flops / max(bytes_moved, 1e-9)
    roof_compute = peak_tflops
    roof_mem = intensity * peak_tbs * 1000
    attainable = min(roof_compute, roof_mem)
    return {
        "intensity_flops_per_byte": intensity,
        "attainable_tflops_order": attainable,
        "likely": "compute-bound" if roof_mem >= roof_compute else "memory-bound",
    }

M, N, K = 8192, 8192, 8192
flops = 2 * M * N * K
bytes_moved = 4 * (M * N + N * K + M * K)
print(roofline_hint(flops, bytes_moved))
```

### 6. 梯度检查点（概念示例）

```python
import torch
from torch.utils.checkpoint import checkpoint

m = torch.nn.Linear(1024, 1024, device="cuda")

def segment(x):
    return torch.relu(m(x))

x = torch.randn(8, 1024, device="cuda", requires_grad=True)
# 将 segment 作为一段重算单元（真实场景按层/块划分）
y = checkpoint(segment, x, use_reentrant=False)
y.sum().backward()
```

---

## 面试要点

### 必背清单

1. **架构**：CPU **少核强核、低延迟** vs GPU **多核弱核、高吞吐**；LLM **稠密 GEMM** 匹配 GPU。
2. **SM / Warp**：**SM** 为执行集群；**Warp=32 线程** 为调度粒度；**分支发散** 降效。
3. **CUDA Core vs Tensor Core**：**CUDA Core** 通用浮点；**Tensor Core** 专吃 **GEMM 类** 峰值。
4. **层级顺序**：寄存器 → Shared → L1 → L2 → HBM →（跨设备）CPU DRAM；**片上远快于 HBM**。
5. **锚点（量级）**：Shared **~128～228KB/SM**，片上带宽 **~19 TB/s 量级**；H100 **HBM ~80GB、~3.35 TB/s**；CPU↔GPU **~50～400 GB/s 有效**。
6. **强度与 Roofline**：**FLOPs/Byte**；**min(算力顶, 带宽顶)**。
7. **算子**：**大 GEMM → compute-bound**；**逐元素 → memory-bound**；**朴素 Attention → memory-bound**；**FlashAttention → 减 HBM IO**。
8. **代际**：**A100** Ampere；**H100** Hopper；**H200** 更大 HBM/带宽。
9. **指标**：**MFU** 衡量算力利用；带宽利用率对照 **Roofline**。
10. **工具**：**`synchronize()`**；**torch.profiler**；**Nsight Systems / Compute**；**py-spy** 看 Python/CPU。
11. **显存**：FP32 静态 **~16 B/参数**；总 **+ 激活 + 并行策略**。
12. **Checkpointing**：**重算换显存**。

### 面试高频题（10+ 详解）

**1. GPU 和 CPU 的主要区别？**  
**答**：**设计目标不同**。CPU **核心数少、单核强**，有 **大缓存、乱序执行、分支预测**，适合 **操作系统、复杂控制流、低延迟串行逻辑**。GPU **核心/线程数极多、单线程较瘦**，采用 **SIMT**，用 **海量并行隐藏内存延迟**，片上 **ALU 与 Tensor Core** 追求 **吞吐**。深度学习 **稠密矩阵乘与规则并行** 与 GPU 高度匹配；**不规则稀疏、强分支** 更吃 CPU 或专用架构。

**2. SRAM 和 HBM 的区别？各自的容量和带宽？**  
**答**：**SRAM** 指 **片上静态存储**（含 **共享内存、缓存、寄存器文件** 等路径），**容量小**（如 **每 SM 一百多 KB 共享内存**、L2 **数十 MB 全卡**），但 **带宽极高**（共享内存路径教学锚点 **~19 TB/s 量级**）。**HBM（High Bandwidth Memory）** 是 **GPU 主存**，**容量大**（常见 **40～80 GB**，H200 更大），**带宽** 低于片上 SRAM，但高于传统 DDR，常见讨论 **~1.5～3.35 TB/s**。面试强调：**越小越快越贵；HBM 是大容量主战场**。

**3. 什么是计算密集型 vs 访存密集型操作？**  
**答**：**计算密集型（compute-bound）** 指 **性能主要受峰值 FLOPS 限制**，内存带宽 **未饱和**，提升 **算力或 Tensor Core 利用率** 更有效。**访存密集型（memory-bound）** 指 **性能主要受内存带宽限制**，算力 **吃不满**，优化重点是 **减少读写字节、融合 kernel、改善访存合并与缓存复用**。

**4. 如何判断一个操作是计算瓶颈还是访存瓶颈？**  
**答**：三步走：**（1）** 估算或测量 **算术强度 FLOPs/Byte**；**（2）** 用 **Roofline** 看 **min(算力顶, 强度×带宽)**；**（3）** 用 **profiler**（`torch.profiler`、Nsight）观察 **kernel 是否接近带宽顶**、**是否大量 memory stall**。若 **大 GEMM** 且 Tensor Core 活跃、**MFU 高**，多 **compute-bound**；若 **逐元素链**、**朴素 Attention** 且 **内存事务占比高**，多 **memory-bound**。

**5. 训练一个模型需要多少 GPU 显存？如何估算？**  
**答**：分块估算：**静态** = **参数 + 梯度 + 优化器状态**（Adam 常用 **FP32 下约 16 B/参数** 粗算）+ **框架开销（常记 16～20× 参数量字节为量级）**；**动态** = **激活**，与 **batch、T、d、层数、checkpoint** 有关；再加上 **分布式** 下的 **分片、通信缓冲**。实操可用 **`torch.cuda.max_memory_allocated()`** 与 **逐步打开**（无 checkpoint → 有 checkpoint → ZeRO）观测。

**6. Tensor Core 的作用是什么？**  
**答**：**Tensor Core** 是 **矩阵乘累加** 专用单元，在 **FP16/BF16/TF32/FP8** 等格式上对 **GEMM** 提供 **远高于 CUDA Core 标量乘加** 的 **峰值吞吐**。大模型训练 **主力算力** 来自 Tensor Core；**小矩阵、不规则访存** 可能无法充分发挥。

**7. 什么是 MFU（Model FLOPS Utilization）？**  
**答**：**MFU** 衡量 **模型在真实步进中 achieved FLOPS** 相对 **硬件峰值 FLOPS** 的比例，反映 **实现与调度** 是否 **吃满算力**。**MFU 低** 时需结合 **访存、通信、kernel 粒度、Python 开销** 排查；与 **仅看 GPU-Util** 相比，更贴近 **有效训练吞吐** 讨论。

**8. H100 相比 A100 的主要提升？**  
**答**：**同代际 Hopper vs Ampere**：**H100** 在 **第四代 Tensor Core**、**峰值算力**、**HBM3 带宽（常见 ~3.35 TB/s 量级）**、**新数据类型（如 FP8）与系统特性** 上整体强于 **A100（~2 TB/s 量级带宽的常见讨论）**；具体数值以 **官方规格** 为准。面试答 **算力 + 内存带宽 + 新特性** 三类即可。

**9. 如何使用 profiler 分析性能瓶颈？**  
**答**：**（1）** 先 **`torch.cuda.synchronize()`** 保证计时正确；**（2）** 用 **`torch.profiler`** 找 **Top CUDA ops** 与 **Python 热点**；**（3）** 用 **Nsight Systems** 看 **DataLoader、H2D、计算是否重叠**；**（4）** 对关键 kernel 用 **Nsight Compute** 看 **occupancy 与内存**；**（5）** 若怀疑 **GIL/Python**，用 **py-spy** 采样。最终把结论映射到 **Roofline：算力顶还是带宽顶**。

**10. 激活值重计算（activation checkpointing）的原理？**  
**答**：**反向传播** 需要 **前向中间激活** 计算梯度。**Checkpointing** 在 **前向** 中 **不保存** 部分段落的激活，在 **反向** 经过该段时 **再前向一次** 以恢复激活。**代价** 是 **额外前向计算**；**收益** 是 **峰值激活显存下降**。与 **FlashAttention** 同属 **用计算换存储/带宽** 的系统策略。

**11.（补充）Kernel 融合为什么能加速？**  
**答**：融合后 **中间结果可留在寄存器/共享内存**，**减少对 HBM 的读写次数**，并 **降低 kernel launch 开销**；对 **memory-bound** 的逐元素链 **效果尤其明显**。

**12.（补充）`torch.cuda.empty_cache()` 能否立刻让 nvidia-smi 下降？**  
**答**：**不一定**。它主要释放 **PyTorch 缓存分配器** 中的空闲块；**CUDA 驱动与上下文** 仍可能占用，**OS 级归还** 也不保证即时。

---

## 练习

1. **推导**：对 \(A\in\mathbb{R}^{M\times K}, B\in\mathbb{R}^{K\times N}\)，FP32 GEMM 的 **FLOPs** 与 **朴素读写字节上界**；写出 **算术强度** 表达式；当 \(M=N=K\to\infty\) 时强度趋势？

2. **对比**：**LayerNorm → Dropout → MatMul** 中，哪段更可能 **memory-bound**？为什么？

3. **估算**：**13B 参数**，FP32 **参数+梯度+Adam**，粗算多少 **GB**？若 **BF16 参数 + FP32 优化器状态**，静态部分如何变化（定性）？

4. **工具**：设计一次实验，用 **Nsight Systems** 区分 **DataLoader 瓶颈** 与 **GPU kernel 瓶颈**。

5. **Roofline**：画一张草图（纸笔即可）：横轴 **算术强度**，纵轴 **GFLOPS**；标出 **算力水平线** 与 **带宽斜线**。

6. **FlashAttention**：用一句话解释 **IO 复杂度** 直觉（**分块 + 重算**）。

7. **（进阶）**：逐元素链 **融合前** 每元素 **5 次读写过 HBM**，**融合后** **1 次读 + 1 次写**，**理论带宽需求** 下降多少倍？对 **memory-bound** 场景意味着什么？

8. **MFU**：若某训练 **achieved 200 TFLOPS**，GPU **峰值 1000 TFLOPS**，**MFU** 是多少？可能原因列举三项。

---

## 附录：面试速记卡片

| 关键词 | 一句话 |
|--------|--------|
| SIMT / Warp | 32 线程一组调度，分支发散伤性能 |
| SM | 调度与执行的「车间」，多 SM 构成整卡并行 |
| CUDA Core | 通用浮点 ALU |
| Tensor Core | GEMM 峰值算力担当 |
| HBM | 全局显存主战场，容量大、带宽低于片上 SRAM |
| Shared | 块内共享，带宽可达 ~19 TB/s 量级（锚点） |
| Intensity | FLOPs / Bytes，Roofline 取 min |
| GEMM | 大规格多为 compute-bound |
| Elem-wise | 常为 memory-bound，融合救命 |
| MFU | achieved FLOPS / 峰值 FLOPS |
| Adam | 静态约 16B/参数（FP32 m,v） |
| empty_cache | 不保证 OS/驱动立刻归还 |
| Fusion | 少读写 HBM、少 launch |
| A100 / H100 / H200 | Ampere → Hopper 主力 → 更大 HBM 与带宽 |

---

## 导航

### 本节小结

- **CPU vs GPU**：**少而强、低延迟** 对比 **多而简、高吞吐**；**SIMT + Warp** 是调度核心。
- **硬件**：**SM** 为执行集群；**CUDA Core** 通用、**Tensor Core** 专吃 **GEMM**。
- **内存层级**：**寄存器 → Shared（~128～228KB/SM，~19TB/s 量级锚点）→ L1/L2 → HBM（~40～80GB+，~1.5～3.35TB/s）→ CPU DRAM（跨设备 ~50～400GB/s 有效）**。
- **性能模型**：**算术强度** + **Roofline**；**GEMM 偏 compute-bound**，**逐元素与朴素 Attention 偏 memory-bound**。
- **代际**：**A100 → H100 → H200** 算力与 **HBM 容量/带宽** 递进。
- **工具**：**torch.profiler**、**Nsight Systems/Compute**、**py-spy**；**synchronize** 正确计时。
- **分析**：**MFU**、**带宽利用率**、**瓶颈识别流程**。
- **工程**：**torch.cuda** 设备与显存 API；**训练显存 = 静态权重相关 + 激活 + 并行**；**checkpointing = 重算换显存**。

### 相关链接

- **下一节**：[Lesson 10：FlashAttention 原理与 Triton](./10-FlashAttention原理与Triton.md)
- **相关复习**：[Lesson 03：Transformer 架构详解](./03-Transformer架构详解.md)、[Lesson 04：多头注意力与 RoPE](./04-多头注意力与RoPE.md)
- **总览**：[Lesson 00：课程总览与学习路线](./00-课程总览与学习路线.md)

---

**版本说明**：文中 **容量、带宽、延迟** 均为 **教学量级**；生产环境请以 **NVIDIA 官方规格** 与 **实测 profiling** 为准。

**延伸阅读（非面试必答）**：Hopper **TMA（Tensor Memory Accelerator）**、**异步拷贝** 等特性进一步降低 **访存与计算流水** 之间的空隙；若投递 **CUDA / 推理引擎** 方向，可结合 **Nsight Compute** 的 **memory workload analysis** 做深挖。
