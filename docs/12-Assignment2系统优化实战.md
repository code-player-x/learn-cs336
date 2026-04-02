# Lesson 12：Assignment 2 系统优化实战

> **Stanford CS336**：Language Modeling from Scratch — 面试导向学习指南（第 12 节）

**先修**：[Lesson 10：FlashAttention 原理与 Triton](./10-FlashAttention原理与Triton.md)、[Lesson 11：DDP 分布式训练](./11-DDP分布式训练.md)。

**面试热度**：★★★★★（系统 / 推理 / 训练工程岗极高频；常与「性能分析 → 内核 → 分布式」三连问绑定）

---

## 标题与定位

本节对应 **CS336 Assignment 2（Systems）**：把 **性能剖析（Profiling）**、**Triton 版 FlashAttention-2**、**DDP 多卡训练** 串成一条「先测量、再改内核、最后规模化训练」的闭环。学完应能：用工具 **定位算力/访存瓶颈**；理解 **Triton 块编程与在线 softmax**；独立写出 **`torchrun` + DDP + AMP** 的可复现脚本；并在面试中用 **数据与 trace** 而非形容词讲清优化故事。

**文档结构**：概念（Concepts）→ 代码走读（Code Walkthrough）→ 面试要点（Interview Points）→ 练习（Practice）→ 导航（Navigation）→ **面试高频题详解（10+ 题）**。

---

## 一、概念讲解（Concepts）

### 1.1 Assignment 2 总览：Profiling、FlashAttention-2（Triton）、DDP

| 模块 | 核心技能 | 面试官想听的关键词 |
|------|-----------|---------------------|
| **Part 1：Profiling & Benchmarking** | 层级/算子时间分解、基线、指标口径 | `torch.profiler`、Chrome trace、compute-bound / memory-bound、吞吐与延迟 |
| **Part 2：FlashAttention-2 in Triton** | 分块、在线 softmax、块大小与资源 | tiling、online softmax、`tl.dot`、SRAM、与 naive/参考实现对比 |
| **Part 3：DDP Training** | 进程组、梯度同步、混合精度、多卡脚本 | `init_process_group`、`DistributedDataParallel`、`all_reduce`、NCCL、AMP |

**一句话**：Assignment 2 练的是 **「测量驱动的系统优化」** —— 先证明瓶颈在哪，再在 **正确性约束** 下改内核与训练栈。

---

### 1.2 Part 1：Profiling 与 Benchmarking

**要解决什么问题**：把「模型慢」拆成 **可操作的子问题** —— 是 **某层 attention**、**FFN GEMM**、**Embedding**，还是 **CPU 数据管线**、**分布式通信**？

**推荐工作流**：

1. **固定基线（baseline）**：同一 GPU 型号、驱动、CUDA、PyTorch 版本；固定 `batch × seq × dtype`；记录 **步时（ms/step）**、**tokens/s**、**峰值显存**。
2. **层级 profiling**：对 `forward` / `backward` 分段打点（`record_function` 或模块 hook），看 **时间占比**；Transformer 常见热点在 **注意力** 与 **FFN**。
3. **算子与 kernel 级**：导出 **Chrome trace**（Perfetto），观察 **kernel 序列**、**是否存在大量 `memcpy`、小 kernel 风暴、意外同步**。
4. **区分 compute vs memory**：
   - **Compute-bound**：大 GEMM/Conv 占主导，提高 **Tensor Core 利用率**、算子融合、更大 tile 可能有效。
   - **Memory-bound**：访存带宽打满、算术强度低；**减少 HBM 往返**（如 FlashAttention）、融合、避免多余 `contiguous`/中间张量更有效。
5. **系统级**：`DataLoader` 是否让 GPU **空转**；分布式下 **all_reduce 气泡** 是否主导。

**性能指标（面试必会定义）**：

| 指标 | 含义 | 典型用途 |
|------|------|----------|
| **Throughput（吞吐）** | 单位时间处理样本数或 token 数（samples/s、tokens/s） | 训练效率对比 |
| **Latency（延迟）** | 单次迭代或前向耗时（ms/step、ms/batch） | 推理、同步开销、通信敏感场景 |
| **Memory（显存）** | 峰值分配（GB）、allocator 统计、activation 峰值 | OOM 排查、checkpoint、ZeRO 决策 |

**补充**：可结合 **roofline 模型**（算术强度 vs 硬件峰值算力/带宽）判断「算力顶」还是「带宽顶」；深度优化常用 **Nsight Systems（nsys）** 看 CPU–GPU 并发，**Nsight Compute（ncu）** 看单 kernel 的内存吞吐与占用。

---

### 1.3 Part 2：FlashAttention-2 与 Triton 实现要点

**FlashAttention-2**（相对第一代）通常强调 **更少的非 matmul 开销、更合理的工作划分与并行策略**（细节以课程讲义与论文为准）。在 **Triton** 中实现时，典型关注点包括：

- **沿序列维分块（tiling）**：外层按 **query 块** 调度，内层遍历 **K/V 块**。
- **Online softmax**：按行维护 **运行最大值 \(m\)**、**运行归一化因子 \(\ell\)**（与 exp 和相关）、**输出累加 \(\mathbf{o}\)**；每来一个新块做 **rescale** 合并，避免完整物化 \(N\times N\) 注意力矩阵到 HBM。
- **融合**：在 **少量 kernel** 内完成 \(QK^\top\)、缩放、mask、softmax 与对 \(V\) 的加权，显著降低 **HBM traffic**。

**Triton 内核开发流程（可写进简历/面试）**：

1. **规格与形状**：固定 `B, H, T, D`；先写清 **因果 / 非因果**、**dtype**（fp16/bf16）。
2. **参考实现**：PyTorch **朴素注意力**（小 `T`）、`F.scaled_dot_product_attention`（环境允许时）作为 **golden**。
3. **最小可运行内核**：单头或小 `B`，只实现 forward；对齐 **mask 与 `1/sqrt(d)`**。
4. **在线 softmax**：严格按递推式实现 \(m,\ell,\mathbf{o}\)，注意 **数值稳定**（减 max）。
5. **调块与并行**：调整 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`（或课程命名），观察 **寄存器 spill、shared memory、occupancy**。
6. **性能对比**：与 **naive**、**SDPA** 对比 **耗时与显存**；用 **nsys/ncu** 佐证瓶颈类型。

**块大小选择策略**：

- **资源上限**：shared memory、寄存器随块尺寸增长；过大导致 **occupancy 下降** 或 **编译失败**。
- **硬件对齐**：内积维常对齐到 **16/32/64** 等以适配 Tensor Core；以目标 GPU 与 Triton 文档为准。
- **经验法**：从 **小块保证正确** 起步，再 **逐步放大** 扫吞吐；观察是否 **memory-bound** 或 **寄存器溢出**。

**正确性**：在 **atol/rtol** 约定下与参考对齐；覆盖 **长序列、边界 `T`、因果对角线、多 head**；fp16 需放宽阈值并记录 **最大误差位置** 辅助调试。

**性能对比维度**：**naive** 通常 **HBM 读写多、峰值激活大**；**FlashAttention** 在 **长序列** 上 **时间、显存** 往往显著更优（具体比例依赖硬件与形状）。

---

### 1.4 Part 3：DDP 训练实现

**DDP（Distributed Data Parallel）**：每进程 **一份完整模型**，各卡 **不同 micro-batch**；反向得到本地梯度后，通过 **`all_reduce`** 得到 **全局平均梯度**（常见语义），再各卡 **相同地** 更新参数。

**进程组与初始化**：

- `torch.distributed.init_process_group(backend="nccl", ...)`：单机多卡/多机多卡均常用 **NCCL**。
- 环境变量：`RANK`、`WORLD_SIZE`、`LOCAL_RANK`（`torchrun` 自动注入）；`torch.cuda.set_device(local_rank)` **一进程一卡**。

**梯度同步与 hooks**：

- `DistributedDataParallel` 在 **`backward`** 中注册 **gradient accumulation hooks**：梯度就绪后按 **bucket** 触发 **`all_reduce`**，并与 **反向计算重叠**（实现细节随 PyTorch 版本演进）。
- 一般业务代码 **无需手写** `all_reduce`；若自定义通信（如 **gradient compression**），才需了解 **hook 时机** 与 **bucket**。

**混合精度（AMP）与 DDP**：

- 前向：`torch.cuda.amp.autocast` 或 `torch.amp.autocast("cuda", dtype=...)`。
- 反向：`GradScaler`（fp16 常用）做 **loss scaling**，`scaler.step` / `scaler.update`。
- **各 rank 控制流须一致**；**梯度裁剪** 常在 **unscale 之后**。

**多 GPU 训练脚本入口**：

- 推荐 **`torchrun`**（或 `python -m torch.distributed.run`）：自动设 `RANK`/`WORLD_SIZE`/`LOCAL_RANK`，支持 **多机** 时传 `--nnodes`、`--node_rank`、`--master_addr` 等。

---

### 1.5 分布式训练调试：死锁、NCCL、显存「泄漏」

**死锁与同步**：

- **不同 rank 执行不同分支**：例如仅部分 rank 调用 `barrier`、`all_gather`，或 `if rank==0` 内额外集合通信 → **极易死锁**。
- **顺序不一致**：某些 rank 多一次 `backward`、或少一次 `step`，集合通信 **次数不匹配**。
- **Sampler**：`DistributedSampler` + 多 epoch 必须 **`sampler.set_epoch(epoch)`**，否则 **shuffle 可重复/错配**（数据正确性问题，有时表现为「诡异」指标）。

**NCCL 常见问题与思路**：

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| 卡住无输出 | 某 rank 掉队、IO、编译 | `NCCL_DEBUG=INFO`；对齐各 rank 步数 |
| `unhandled system error` | 驱动/拓扑/权限 | 升级驱动；检查 PCIe/NVLink/IB |
| `invalid usage` | tensor 不在 GPU、dtype 不一致 | 统一 `device` 与 `dtype` |
| 多机超时 | 网络、防火墙 | 检查 `MASTER_ADDR/PORT`；必要时调超时参数（版本相关） |

**调试环境变量（按需）**：`NCCL_DEBUG=INFO`、`TORCH_DISTRIBUTED_DEBUG=DETAIL`（名称以当前 PyTorch 文档为准）。

**「显存泄漏」感**：

- 实为 **缓存未释放**、**Python 引用未清**、**每步 `clone` 堆积**；`empty_cache` **不能替代**根因分析。
- 多进程注意 **DataLoader worker**、**自定义 CUDA 扩展** 生命周期。

---

### 1.6 性能优化检查清单（Checklist）

**测量**

- [ ] 可复现 baseline（版本、种子、配置、硬件）。
- [ ] 区分 **纯训练步** 与 **数据加载**（profiler CPU/GPU 时间线）。
- [ ] 记录 **吞吐、延迟、峰值显存** 三联。

**Attention / 显存**

- [ ] 能否用 **SDPA / FlashAttention** 替代朴素实现？
- [ ] 是否需要 **gradient checkpointing**？
- [ ] 序列与 batch 是否 **分阶段** 增大？

**分布式**

- [ ] 全局 batch = `local_batch × world_size × grad_accum`。
- [ ] 学习率是否随 **全局 batch** 做合理缩放（如线性缩放规则及例外）。
- [ ] 是否了解 **`find_unused_parameters`** 等对性能的影响（有则按需开启）。

**混合精度**

- [ ] `autocast` 覆盖主要矩阵运算；敏感层是否 **fp32**（视模型而定）。
- [ ] `GradScaler` 与 **clip** 顺序正确。

---

### 1.7 常见陷阱与对策

| 陷阱 | 对策 |
|------|------|
| 无 `cuda.synchronize()` 测 GPU 时间 | 计时前后 **同步**；profiler 也要注意异步 launch |
| 只看 GPU util 高 | 可能在跑 **低效 kernel**；结合 **有效吞吐** 与 **ncu** |
| Triton 数值偏差 | 对齐 **scale/mask**；小形状单测；fp32 参考 |
| DDP 打印 local loss 不一致 | 本地 batch 不同本就可能不同；报告 **聚合后** loss |
| OOM 只减 batch | 先 profiler 看 **激活峰值层**；配合 checkpoint / FA |

---

### 1.8 面试中如何呈现 Assignment 2（STAR）

- **S（情境）**：Systems 作业要求 **profiler 找热点**、**Triton 实现类 FA2**、**多卡 DDP+AMP**。
- **T（任务）**：朴素注意力 **memory-bound + 显存峰值高**；单卡训练 **吞吐不足**。
- **A（行动）**：分块 + online softmax；**单测/数值对齐**；`torchrun`+`DDP`+`GradScaler`；**trace 与 benchmark 表**留存。
- **R（结果）**：用 **tokens/s、ms/step、峰值 GB** 量化；能解释 **为何快**（减少 HBM、融合）。

---

## 二、代码走读（Code Walkthrough）

> 以下为 **教学向伪代码**，与官方作业 **API 可能不同**，以课程 **starter code 与测试** 为准。

### 2.1 使用 `torch.profiler` 做层与算子级分析

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def train_step(model, x, y, optimizer):
    with record_function("forward"):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss

# 示例：抓取 CPU/GPU 活动；可配合 schedule 做 wait/warmup/active
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(16):
        loss = train_step(model, batch_x, batch_y, optimizer)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
# prof.export_chrome_trace("trace.json")  # Perfetto / chrome://tracing 打开
```

**读表与 trace 的关注点**：

- **Self CUDA time** 高的算子：是否 **非预期 memcpy**、**过多小 kernel**？
- **Attention**：融合前后 **kernel 数量** 与 **总耗时** 对比。
- **CPU**：`aten::` 与 DataLoader 是否长时间占用导致 GPU **饥饿**？

---

### 2.2 Benchmarking：对比不同 attention 实现

```python
import time
import torch
import torch.nn.functional as F

def benchmark(fn, warmup=10, steps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps

def attn_naive(q, k, v, is_causal=True):
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) * (d ** -0.5)
    if is_causal:
        t = scores.size(-1)
        mask = torch.triu(torch.ones(t, t, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return p @ v

# 吞吐：tokens/s ≈ (B * T * steps) / total_seconds（forward 单次）
```

**要点**：**必须** `torch.cuda.synchronize()`；否则测到的是 **异步排队**。

---

### 2.3 Triton：正确性与性能对比流程

```python
# 正确性（示例阈值，以作业要求为准）
torch.manual_seed(0)
B, H, T, D = 2, 8, 512, 64
q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)

ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out = triton_flash_attn(q, k, v, causal=True)
torch.testing.assert_close(ref, out, rtol=2e-2, atol=2e-2)
```

**对比 naive vs FlashAttention（汇报用）**：

| 维度 | Naive / 未融合 | Triton FlashAttention-2 类实现 |
|------|-----------------|----------------------------------|
| HBM 访问 | 常显著更高（物化大方阵等） | 分块融合，降低 traffic |
| 峰值显存 | \(O(N^2)\) 级中间结果风险 | 通常更低（实现相关） |
| 调优抓手 | 有限 | block、occupancy、融合度 |

---

### 2.4 DDP：进程组、`torchrun`、混合精度

**启动（shell）**：

```bash
torchrun --nproc_per_node=4 train.py --config configs/lm.yaml
```

**训练骨架（Python）**：

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size, local_rank = setup()
    model = MyLM().to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        # find_unused_parameters=False,  # 默认 False；有未参与 loss 的参数时需谨慎
    )

    dataset = MyDataset(...)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(local_rank, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**batch).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    cleanup()
```

**梯度同步直觉**：`DDP` 在 backward 中 **按 bucket** `all_reduce` 梯度；优化器 step 前各卡应对齐 **同一份平均梯度**（在标准同步 SGD 语义下）。

---

## 三、面试要点（Interview Points）

### 3.1 电梯演讲（30 秒）

1. **FlashAttention 为何快**：减少 **HBM 往返**，在 **片上** 完成分块注意力；数学上与标准 softmax 注意力一致（非随意近似）。
2. **如何论证 memory-bound**：roofline、trace 里 **memcpy** 占比、**ncu** 的 **DRAM throughput vs 算力**。
3. **DDP 同步什么**：同步 **梯度**（通常平均）；参数在 **相同初始化与相同梯度** 下保持一致更新。
4. **AMP + DDP**：**loss scale**、**unscale 与 clip 顺序**、各 rank **控制流一致**。

### 3.2 高频追问

- **块大小怎么选**：资源约束 → 实测吞吐 → 是否 spill。
- **Online softmax 三个量**：\(m,\ell,\mathbf{o}\) 与 **rescale**。
- **DDP vs ZeRO**：DDP **每卡全参**；ZeRO **切分优化器状态/梯度/参数**（进阶）。

---

## 四、练习（Practice）

### 4.1 思考题

1. Profiler 显示 **Embedding** 占比高：如何区分 **IO/采样** vs **kernel**？
2. 为什么 **过大的 Triton block** 可能反而变慢？
3. 用文字描述一种 **必死锁** 的分布式写法。
4. 全局 batch 翻倍，学习率 **是否必翻倍**？依据与例外？

### 4.2 动手题

- 导出 **Chrome trace**，标注 **attention** 与 **FFN** 时间段。
- 画 **随序列长度** 的 naive vs Triton **耗时曲线**。
- `NCCL_DEBUG=INFO` 跑两卡，保存 **首次 all_reduce** 相关日志作面试素材。

---

## 五、导航（Navigation）

- **上一节**：[Lesson 11：DDP 分布式训练](./11-DDP分布式训练.md)
- **下一节**：[Lesson 13：Scaling Laws 缩放定律](./13-Scaling-Laws缩放定律.md)

---

## 六、面试高频题（详细参考答案）

### Q1：如何分析模型的性能瓶颈？

**答**：采用 **系统分层** 思路：**迭代级 → 模块级 → 算子/kernel 级 → 系统级**。

1. **可复现基线**：固定硬件与软件版本、输入形状、精度；记录 **ms/step**、**tokens/s**、**峰值显存**。
2. **模块级**：`torch.profiler` + `record_function` 看 **forward/backward** 各段占比；Transformer 常见 **注意力 + FFN** 双热点。
3. **算子/kernel 级**：`export_chrome_trace` 用 **Perfetto** 看时间线；识别 **memcpy**、**小算子碎片化**、**意外同步**。
4. **瓶颈类型**：**算术强度** 与 **roofline** 辅助判断 **compute-bound** 还是 **memory-bound**；必要时 **ncu** 看单 kernel。
5. **系统级**：**DataLoader**、磁盘、CPU 预处理是否导致 GPU **饥饿**；分布式下 **通信气泡** 占比。

**加分**：主动提 **nsys / ncu** 与 **roofline** 的适用分工。

---

### Q2：profiling 工具的使用方法？

**答**：

1. **PyTorch Profiler**：`profile(activities=[CPU, CUDA])` 包裹训练步；`key_averages().table(sort_by="cuda_time_total")` 看排名。
2. **进阶选项**：`record_shapes=True` 查 **频繁 reshape**；`profile_memory=True` 看 **显存热点算子**；`with_stack=True` 关联 **Python 源码栈**。
3. **可视化**：`export_chrome_trace("trace.json")` 导入 **Perfetto** 做 **时间线分析**。
4. **外部工具**：**Nsight Systems** 看 **端到端并发**；**Nsight Compute** 看 **单 kernel** 细节。
5. **计时注意**：GPU 默认异步，benchmark 与部分计时需 **`torch.cuda.synchronize()`**。

---

### Q3：FlashAttention Triton 实现的关键步骤？

**答**：

1. **Grid 划分**：按 **query 块**（或 `(batch, head, q_tile)`）映射到 program id。
2. **加载 Q tile** 到片上（寄存器/shared，依实现）。
3. **K/V 内层循环**：`tl.dot` 等计算 **块内 logits**（注意 **scale** 与 **mask**）。
4. **Online softmax**：更新 \(m,\ell,\mathbf{o}\)，新块 **rescale** 历史输出。
5. **写回**：将 **输出 tile** 写回全局内存；若作业要求 backward，常涉及 **重算** 或 **保存最小中间量**（依课程定义）。

**一句话**：**不在 HBM 物化完整 \(N\times N\)**，并尽量减少 **round-trips**。

---

### Q4：如何验证 kernel 的正确性？

**答**：

1. **参考**：朴素实现、`F.scaled_dot_product_attention`（若可用）。
2. **测试分层**：手工小例子 → 随机张量 → **边界**（`T=1`、因果边界、极端 `d`）。
3. **dtype**：fp16/bf16 设合理 **rtol/atol**；必要时 **fp32 黄金标准**。
4. **定位技巧**：不对时先关 **因果**、减 **head/batch**、缩小 **block**，二分定位。
5. **加分**：统计 **最大误差位置**、多 **seed** 压力测。

---

### Q5：DDP 训练中遇到的典型问题？

**答**：

1. **设备/进程**：未按 `LOCAL_RANK` **绑卡**；部分 tensor 在 CPU。
2. **采样**：未用 `DistributedSampler` 或忘 **`set_epoch`**。
3. **语义**：把 **local batch** 当 **global batch**，导致 **学习率/日志** 错误。
4. **通信**：`DataParallel` 与 `DDP` 混用；或 **rank 间控制流不一致** 导致 **死锁**。
5. **性能**：`find_unused_parameters=True` 等带来的 **额外开销**（有未用参数时不得已）。

**处理**：统一 **`torchrun`**；只用 **DDP**；开 **`TORCH_DISTRIBUTED_DEBUG`**；日志看 **聚合指标**。

---

### Q6：如何优化通信和计算的重叠？

**答**：

1. **DDP bucket**：梯度 **分桶 all_reduce**，与 **backward** 流水线 **重叠**（实现随版本迭代）。
2. **异步数据**：`non_blocking=True`、`pin_memory`、合适 **`num_workers`**。
3. **梯度累积**：改变 **通信频率** 与 **有效 batch** 的折中。
4. **高级**：自定义 **communication hook**、**压缩梯度**（偏研究/infra 岗）。

**核心叙述**：减少 **通信气泡**，让 **NIC/NCCL** 在 GPU 仍计算时 **并行推进**。

---

### Q7：混合精度训练在 DDP 中如何使用？

**答**：

1. **autocast** 包裹前向主体；对 **数值敏感** 模块可 **禁用** 或 **fp32**（按模型）。
2. **GradScaler**：`scaler.scale(loss).backward()` → `unscale_` →（可选）**`clip_grad_norm_`** → `scaler.step` → `scaler.update`。
3. **DDP**：每个 rank **同样执行** scaler 逻辑；避免 **仅 rank0** 做会改变图的操作。
4. **bf16**：若硬件支持，有时可 **减弱** 对 scaler 的依赖（仍取决于框架版本与数值稳定性策略）。

---

### Q8：性能优化的一般方法论？

**答**：**Measure → Identify → Optimize → Verify**。

1. **Measure**：无测量不优化；保留 **版本与配置** 可复现。
2. **Identify**：区分 **算力 / 访存 / 通信 / IO**；用 trace 与 roofline **定位主因**。
3. **Optimize**：先做 **低风险高收益**（SDPA、AMP、DataLoader），再 **内核**，再 **分布式策略**。
4. **Verify**：**正确性** + **性能不退化** + **训练指标** 三重验收。

---

### Q9：如何计算 GPU 利用率？

**答**：

1. **采样型 util**：如 **NVML** 的 GPU-Util（一段时间内 **至少一个 SM 活跃** 的比例），**粗粒度**。
2. **框架**：部分环境暴露 `torch.cuda.utilization()`（视版本/平台）。
3. **更可靠**：**nsys** timeline 看 **kernel 覆盖**；**ncu** 看 **SM 活跃周期、achieved occupancy**。
4. **训练侧**：**tokens/s**、**迭代分解** 往往比单一 util 数字更有业务意义。

**辨析**：**高 util 不等于高效**；可能充满 **低效率 memcpy** 或 **极短 kernel**。

---

### Q10：内存优化的常用手段？

**答**：

1. **算子/内核**：FlashAttention、**融合**、避免 **大张量物化**。
2. **重计算**：**gradient checkpointing** 换显存。
3. **精度**：fp16/bf16；优化器状态 **fp32** 存权重更新常用。
4. **框架**：减少 **碎片**（ allocator 配置因版本而异）；避免 **无谓 retain graph**。
5. **分布式进阶**：**ZeRO**、**offload**（超出基础 DDP）。

**注意**：checkpoint **降低吞吐换显存**，需 **联合评估**。

---

### Q11（附加）：NCCL 报错如何排查？

**答**：**先稳定复现 → 开日志缩小范围**：`NCCL_DEBUG=INFO`；检查 **tensor device/dtype 一致性**、**各 rank 步数对称**；多机查 **网络与防火墙**；单机可 **对比** `P2P` 开关做诊断（仅实验）；关注 **驱动/CUDA/PyTorch** 版本组合。

---

### Q12（附加）：如何向面试官展示 Assignment 2？

**答**：**问题—方法—证据—反思**：热点如何用 **profiler 证明**；Triton 如何实现 **分块与 online softmax**；**与参考的误差策略**；DDP+AMP **脚本与扩展性**；用 **表格/trace** 展示 **前后吞吐/显存**；诚实讲 **失败的调参尝试** 反而加分。

---

**结语**：Assignment 2 的面试价值在于 **可验证的优化链条**：**profiler 结论 → kernel 正确性测试 → 多卡训练日志与 benchmark** —— 把「我优化过模型」变成 **可展示的工程证据**。
