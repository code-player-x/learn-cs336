# Lesson 11：DDP 分布式训练（Distributed Data Parallel）

> CS336 面试导向学习指南 · 单机多卡与多机多卡训练的核心机制

---

## 一、为什么要做分布式训练？

### 1.1 单卡装不下：显存与参数量

现代大语言模型（LLM）的参数量可达数十亿到万亿级别。即使采用混合精度（FP16/BF16）、梯度检查点（activation checkpointing）等技巧，**单张 GPU 的显存仍可能无法容纳完整模型、优化器状态与激活值**。分布式训练通过将模型、数据或优化器状态分散到多张卡或多台机器上，突破单设备容量上限。

### 1.2 缩短训练时间：并行与吞吐

即使单卡能放下模型，**完整数据集上的训练迭代次数巨大**，单卡训练可能需要数周甚至数月。通过多卡并行处理不同数据子集（数据并行）或拆分模型（模型并行等），可以在理想情况下接近线性加速，显著缩短 wall-clock 时间。

### 1.3 小结

| 动机 | 说明 |
|------|------|
| 显存瓶颈 | 模型/优化器/激活过大，需分片或多副本策略 |
| 时间成本 | 希望用更多算力换更短训练周期 |
| 工程现实 | 生产环境普遍为多卡服务器或 K8s 多节点集群 |

---

## 二、并行范式概览：DP / MP / PP / TP

### 2.1 数据并行（Data Parallelism, DP）

- **思想**：每张卡保存**完整模型副本**，各卡处理**不同的 mini-batch 子集**，前向与反向在本地完成，再**同步梯度**（如 AllReduce 平均）。
- **特点**：实现相对简单；通信量与模型大小、梯度规模相关；适合模型能放进单卡但希望加速训练的场景。
- **典型代表**：PyTorch `DataParallel`（单进程多线程，已不推荐）、**DDP**（多进程，每进程一卡）。

### 2.2 模型并行（Model Parallelism, MP）

- **思想**：将**模型的不同层或模块**放在不同设备上，前向按拓扑顺序在设备间传递中间激活。
- **特点**：可缓解单卡放不下整模的问题；但容易产生**流水线气泡**（若未与流水线并行结合）或设备间串行依赖。
- **与“张量并行”关系**：广义的 MP 可包含按层切分；业界常把**按张量维度切分**单独称为 TP。

### 2.3 流水线并行（Pipeline Parallelism, PP）

- **思想**：将模型按**层**划分为多个 stage，不同 stage 在不同设备上；通过 **micro-batch** 流水线填充，减少设备空闲。
- **特点**：适合**超深网络**、单卡放不下所有层时；需要处理 **bubble**、调度策略（如 1F1B、interleaved 等）。

### 2.4 张量并行（Tensor Parallelism, TP）

- **思想**：在**单层内**对权重矩阵按列/行切分，矩阵乘分块在多设备上完成，中间结果通过通信（如 AllGather / ReduceScatter）组合。
- **特点**：单 layer 计算与通信交织；常见于 Megatron-LM 等；与 **序列并行** 等结合可进一步降低显存。

### 2.5 对比小结

| 范式 | 切分对象 | 主要通信 | 典型用途 |
|------|----------|----------|----------|
| DP | 数据 batch | 梯度 AllReduce | 加速、易与 ZeRO/FSDP 结合 |
| MP | 层/模块 | 激活传递 | 模型过大 |
| PP | 层序列 | 激活 + 流水线控制 | 极深模型 |
| TP | 张量块 | 高频 collective | 单层过大、Megatron 类 |

实际大模型训练常**组合**多种策略（如 DP + TP + PP）。

---

## 三、DDP（Distributed Data Parallel）深入

### 3.1 架构：每卡一份完整模型

在经典 DDP 中，**每个进程对应一个 GPU**，每个进程维护**相同的模型参数副本**。不存在参数分片（那是 FSDP/ZeRO 的方向）。

### 3.2 数据划分

全局 batch size 记为 \(B\)，若有 \(N\) 个进程，通常每个进程的 **local batch size** 为 \(B_{\text{local}} = B / N\)（需整除）。各进程从各自数据子集取样，保证**每个 step 各卡数据不同**，等价于增大吞吐。

### 3.3 梯度同步：AllReduce

反向传播后，各卡得到**本地 batch 上的梯度**。为使所有副本等价于在全局 batch 上训练，需对梯度做**平均**（或等价缩放后再同步）：

\[
\bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i
\]

实现上常用 **AllReduce**：所有进程最终都得到相同的规约结果。若先 Reduce 到 rank 0 再 Broadcast，语义可一致但效率通常不如 AllReduce。

### 3.4 通信原语（Collective operations）

| 原语 | 行为简述 |
|------|----------|
| **Broadcast** | 根进程将张量发给所有进程，大家得到相同数据 |
| **Reduce** | 将各进程张量按运算（如 sum）规约到一个目标 rank |
| **AllReduce** | 规约（如 sum）后**广播**到所有进程，结果一致 |
| **AllGather** | 各进程有不同分片，收集后每进程得到完整拼接结果 |
| **ReduceScatter** | 先 reduce，再按分片 scatter，每进程只得一部分规约结果 |

DDP 梯度同步核心是 **AllReduce**；部分优化器分片或 FSDP 会用到 **ReduceScatter / AllGather** 组合。

### 3.5 Ring-AllReduce 算法（直观）

**目标**：在 \(N\) 个进程上对向量做求和（或平均），使每进程最终都有全局和。

**环形拓扑**：进程排成环 \(0 \to 1 \to \cdots \to N-1 \to 0\)。

**两阶段**（以 sum 为例）：

1. **Reduce-Scatter 阶段**：数据向量切成 \(N\) 块。经过 \(N-1\) 步，每步每个进程把**自己负责的一块**在环上传递并累加；结束后，**每个进程完整拥有某一块的全局部分和**（不同块在不同进程上）。
2. **AllGather 阶段**：再经过 \(N-1\) 步，把各块在环上转一圈，使**每个进程拼出完整的全局和向量**。

**带宽直觉**：在理想环形与均衡切分下，总时间近似与数据量、链路带宽相关；比朴素“集中到 rank0”更充分利用**双向链路**。

### 3.6 梯度分桶（Gradient Bucketing）与重叠

若每产生一个参数的梯度就做一次 AllReduce，**通信次数多、消息小**，GPU 计算与网络都难以饱和。

**做法**：将多个相邻参数的梯度**拼接成大张量**（bucket），在 bucket 级别做 AllReduce；同时利用 **CUDA stream / 异步**，在**当前 bucket 通信**时，让 GPU 继续算**后续层的反向**。

这就是 **computation-communication overlap**：通过 bucketing + 异步 collective，隐藏部分通信延迟。

---

## 四、从零实现 DDP 风格训练（CS336 Assignment 2 思路）

以下代码为**教学示意**：展示进程组、hook、手动 AllReduce 与训练循环骨架，便于面试中口述“如何实现 DDP”。

### 4.1 环境与进程组（NCCL）

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn

def setup():
    # torchrun 会设置这些环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()
```

- **NCCL**：NVIDIA GPU 间通信后端，多机时需正确配置 `MASTER_ADDR` / `MASTER_PORT`。
- **每进程一 GPU**：`set_device(local_rank)` 避免设备冲突。

### 4.2 模型与优化器（每进程各一份）

```python
def build_model_on_device(local_rank):
    model = nn.Linear(1024, 1024).cuda(local_rank)
    return model
```

### 4.3 梯度 Hook：在 backward 后触发 AllReduce

教学版：对所有参数注册 hook，在梯度就绪后做 **all_reduce**（注意与真实 PyTorch DDP 的 bucket 顺序、异步细节有差异）。

```python
def allreduce_grads(model, world_size):
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
            p.grad.div_(world_size)  # 等价于全局平均梯度
```

更贴近 DDP 的写法是用 **hook** 在反向中排队通信（示意）：

```python
def register_ddp_hooks(model, world_size):
    handles = []

    def make_hook(param):
        def hook(grad):
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=False)
            grad.div_(world_size)
            return grad
        return hook

    for p in model.parameters():
        if p.requires_grad:
            handles.append(p.register_hook(make_hook(p)))
    return handles
```

**面试要点**：真实 `torch.nn.parallel.DistributedDataParallel` 使用 **Reducer**、**bucket**、**autograd hook** 与 **prepare_for_backward** 等，与上述简化版相比更注重**重叠与顺序**。

### 4.4 DataLoader 与 DistributedSampler

```python
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def make_loader(rank, world_size, batch_size):
    # 示例：合成数据
    x = torch.randn(1000, 1024)
    y = torch.randn(1000, 1024)
    ds = TensorDataset(x, y)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    return loader, sampler
```

每个 epoch 开始需 **`sampler.set_epoch(epoch)`** 以保证 shuffle 在不同 epoch 可复现且正确。

### 4.5 完整训练循环骨架

```python
def train_one_epoch(model, loader, sampler, optimizer, rank, world_size, epoch):
    model.train()
    sampler.set_epoch(epoch)
    criterion = nn.MSELoss()

    for batch_x, batch_y in loader:
        batch_x = batch_x.cuda(rank, non_blocking=True)
        batch_y = batch_y.cuda(rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()

        # 若未用 register_hook 自动同步，则在此处手动 allreduce
        # allreduce_grads(model, world_size)

        optimizer.step()

def main():
    rank, world_size, local_rank = setup()
    model = build_model_on_device(local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader, sampler = make_loader(rank, world_size, batch_size=32)

    # 可选：教学用 hook 版 DDP
    # handles = register_ddp_hooks(model, world_size)

    for epoch in range(10):
        train_one_epoch(model, loader, sampler, optimizer, rank, world_size, epoch)

    cleanup()

if __name__ == "__main__":
    main()
```

**启动**：`torchrun --nproc_per_node=NUM_GPUS train.py`（多机时再加 `nnodes`、`node_rank` 等）。

---

## 五、FSDP（Fully Sharded Data Parallel）概览

### 5.1 核心思想

经典 DDP：**每卡存全量参数、梯度、优化器状态**。FSDP 将**参数、梯度、优化器状态**在数据并行维度上**分片（shard）**，在需要计算某层时通过 **AllGather** 临时拼出该层完整权重，计算后再释放，从而降低**每卡显存占用**。

### 5.2 与 DDP 的对比（高层）

| 维度 | DDP | FSDP |
|------|-----|------|
| 参数存储 | 每卡全副本 | 分片，按需 AllGather |
| 通信模式 | 主要梯度 AllReduce | AllGather / ReduceScatter 等组合 |
| 显存 | 较高 | 通常更低 |
| 实现复杂度 | 相对简单 | 更高，需处理包装与流 |

### 5.3 适用场景

显存紧张、希望在**不增加太多节点**的前提下训练更大模型或更大 batch 时，优先考虑 FSDP 或 ZeRO。

---

## 六、DeepSpeed ZeRO 阶段（Stage 1 / 2 / 3）

ZeRO（Zero Redundancy Optimizer）通过**消除数据并行中的冗余状态**来省显存：

| Stage | 分片内容 | 效果直觉 |
|-------|----------|----------|
| **ZeRO-1** | 只分片**优化器状态** | 显存下降明显，实现相对简单 |
| **ZeRO-2** | 优化器状态 + **梯度** | 进一步降低每卡占用 |
| **ZeRO-3** | 优化器 + 梯度 + **模型参数** | 最省显存，通信模式更复杂（参数预取等） |

与 FSDP 思想相近处：**分片 + 按需聚合**；差异多在生态、API、与流水线/张量并行的集成方式。

---

## 七、多机训练注意事项

1. **网络**：机间带宽常低于机内 NVLink/NVSwitch；AllReduce 可能成为瓶颈，需关注 **NCCL_IB**、网卡、拓扑。
2. **初始化**：`MASTER_ADDR`、`MASTER_PORT` 必须可达；防火墙放行。
3. **进程数**：`world_size = 节点数 × 每节点 GPU 数`；`rank` 全局唯一。
4. **数据**：`DistributedSampler` 与数据集路径在各节点一致或可访问共享存储。
5. **确定性**：多机下浮点顺序可能略有非确定性；调试可固定种子、注意 cudnn benchmark。
6. **容错**：生产常结合 checkpoint 与弹性训练（超出本课范围但面试可能提及）。

---

## 八、通信开销分析（定性 + 常用量级）

### 8.1 单次 AllReduce 数据量

对 FP32 梯度，参数量为 \(P\)，AllReduce 传输量常按**算法与实现**在 \(O(P)\) 到约 \(2P\) 等量级估算（Ring 等需多轮，但带宽模型常用有效带宽近似）。

### 8.2 与 batch、模型关系

- **数据并行**：通信量主要随**模型大小（梯度维度）**增长，与 local batch 大小无线性关系（batch 只影响计算时间）。
- **瓶颈**：当 **计算时间 \(\ll\) 通信时间** 时，扩展效率下降。

### 8.3 重叠的意义

通过 **bucket** 与异步 NCCL，使 **通信与反向计算并行**，等效降低“暴露的通信时间”。

---

## 九、扩展效率（Scaling Efficiency）

### 9.1 理想线性加速

若用 \(N\) 张卡，理想 wall-clock 变为原来的 \(1/N\)。定义 **加速比** \(S(N) = T_1 / T_N\)，理想 \(S(N)=N\)。

### 9.2 实际因素

- **通信开销**：AllReduce 等随卡数与拓扑变化。
- **straggler**：某 GPU 慢拖累全局。
- **小 batch 效应**：per-GPU batch 过小，计算效率低。
- **全局 batch 变化**：若保持 per-GPU batch 不变而增加 GPU，全局 batch 增大，可能需调学习率（linear scaling 等经验规则）。

### 9.3 扩展效率公式

常定义 **scaling efficiency** 为 \(\eta(N) = S(N) / N\)。若 \(\eta(N)\) 随 \(N\) 快速下降，说明通信或负载不均占主导。

---

## 十、面试高频题详解（10+）

### Q1：DDP 的工作原理？

**答**：DDP 为**数据并行**：每个进程一张 GPU，持有**相同模型副本**。每步各进程用 **DistributedSampler** 取**不同 mini-batch**，独立前向与反向，得到本地梯度。然后通过 **AllReduce** 将各卡梯度求和并平均（或等价缩放），使各卡参数更新一致，等价于在更大全局 batch 上训练。实现上常用 **多进程**、**NCCL**、**梯度 bucket** 与 **hook** 做通信与计算重叠。

---

### Q2：AllReduce 是什么？Ring-AllReduce 如何工作？

**答**：**AllReduce** 是集合通信：每个进程提供输入张量，对所有进程的输入做规约（如求和），**每个进程都得到相同的规约结果**。  
**Ring-AllReduce** 将数据分块，进程排成环，分 **Reduce-Scatter** 与 **AllGather** 两阶段，每阶段约 \(N-1\) 步，使每步通信可与邻居进行，**充分利用环形带宽**，避免单节点成为中心瓶颈。最终每进程都有完整向量的全局和，再除以 \(N\) 即得平均。

---

### Q3：DDP 和 DP（DataParallel）的区别？

**答**：**DataParallel（DP）**：单进程多线程，主卡聚合梯度再广播，**GIL** 与单进程多流易导致扩展性差，**不推荐**。  
**DDP**：**每进程一 GPU**，梯度 AllReduce 在进程间用 NCCL，**多进程**无 GIL 问题，支持多机，**性能与可扩展性更好**。DDP 需要 `torchrun`/`launch` 启动，并正确使用 `DistributedSampler`。

---

### Q4：梯度同步的通信开销如何计算？

**答**：粗略上，与**梯度总字节数**和 **AllReduce 的有效带宽** 有关。FP32 下梯度约 \(4 \times P\) 字节（\(P\) 为参数量）；BF16/FP16 减半。实际时间 \(\approx\) 传输量 / 有效带宽 + 延迟；Ring 等多步算法用**带宽模型**估算。DDP 中若 **bucket 重叠**成功，**暴露**的通信时间小于未重叠情形。多机时机间带宽常是瓶颈。

---

### Q5：如何实现计算和通信的重叠？

**答**：（1）**梯度分桶**：多个参数梯度合并为大消息，减少 launch 次数；（2）**异步 AllReduce**（`async_op=True`）与 **CUDA stream** 协调，在通信进行的同时继续反向计算后续层；（3）PyTorch DDP 内部 **Reducer** 按拓扑顺序调度 bucket。目标是让 **NCCL kernel** 与 **GEMM/反向算子** 时间轴重叠。

---

### Q6：FSDP 和 DDP 的区别？

**答**：**DDP**：每卡**全量**参数与优化器状态，通信以**梯度 AllReduce** 为主。  
**FSDP**：参数/梯度/优化器状态**分片**存储，前向反向时对当前层 **AllGather** 权重，反向后 **ReduceScatter** 等更新分片，**显存更省**，通信模式更复杂。二者都是数据并行家族，FSDP 更接近 ZeRO-3 一类思路。

---

### Q7：DeepSpeed ZeRO 三个阶段分别优化什么？

**答**：**ZeRO-1**：仅**分片优化器状态**（如 Adam 的动量等），每卡不再存完整优化器副本。**ZeRO-2**：在 1 基础上再分片**梯度**。**ZeRO-3**：进一步分片**模型参数**，需在前向/反向时**按层收集参数**，通信与调度最复杂，显存节省最大。

---

### Q8：分布式训练中如何保证梯度一致性？

**答**：各卡本地梯度是对**本地 batch** 的平均（或和）；通过 **AllReduce SUM + 除以 world_size** 得到**全局平均梯度**，等价于在**拼接后的全局 batch** 上的梯度（在标准平均损失定义下）。所有进程使用**相同规约结果**和相同优化器公式更新，故参数保持一致。前提是 **随机种子、Sampler 划分、数值顺序** 在实现上无 bug，且无不参与同步的参数。

---

### Q9：多机训练和单机多卡的区别？

**答**：**单机多卡**：通常 **NVLink/NVSwitch** 带宽高、延迟低，NCCL 易优化。**多机**：依赖 **以太网或 IB**，**机间带宽**常更低、**延迟**更高，AllReduce 更易成为瓶颈；需配置 **MASTER_ADDR/PORT**、**RDMA/NCCL 环境变量**，并注意 **数据路径** 与 **时钟/同步**。算法上仍是 DDP，但**网络拓扑与故障域**不同。

---

### Q10：如何计算分布式训练的扩展效率？

**答**：测单机单卡（或单节点基准）一步时间 \(T_1\)，与 \(N\) 卡（或 \(N\) 节点）下一步时间 \(T_N\)。**加速比** \(S(N)=T_1/T_N\)，**扩展效率** \(\eta(N)=S(N)/N\)。若 \(\eta\) 明显低于 1，分析：**通信占比**、**batch 过小**、**IO**、**straggler**、**学习率与 batch 缩放**是否匹配。也可用 **吞吐量（tokens/s）** 随资源增长是否接近线性来评估。

---

### Q11（补充）：Ring-AllReduce 相对 tree 或 master 的优势？

**答**：**避免单点带宽瓶颈**：中心节点或树根易饱和；环形每步仅与邻居通信，**带宽利用更均衡**。适合 GPU 集群全连接或环形拓扑下的高效实现（具体依 NCCL 算法选择而定）。

---

### Q12（补充）：DDP 中 `find_unused_parameters` 是做什么的？

**答**：若图中**部分参数未参与当前 iteration 的 loss**（多分支网络等），默认 DDP 可能报错。`find_unused_parameters=True` 会标记未用参数，**有额外遍历开销**；更好的做法是**结构设计避免未用参数**或保证每步参与前向的子图一致。

---

## 十一、实践建议（Practice）

1. **最小实验**：用 `torchrun --nproc_per_node=2` 跑通官方 MNIST/CIFAR DDP 示例，改 `world_size` 观察 `DistributedSampler` 行为。
2. **计时**：在 `backward` 前后与 `optimizer.step` 前后打时间戳，对比 **无重叠 / 手动同步** 与 **原生 DDP**。
3. **阅读源码**：浏览 `torch/nn/parallel/distributed.py` 中 Reducer、bucket 相关注释（面试常考“bucket 顺序”）。
4. **对比 FSDP**：同一小模型，记录每卡 **峰值显存** 与 **迭代时间**。
5. **多机模拟**：若有两台机器，练习 `torchrun` 多节点参数与 `NCCL_DEBUG=INFO` 排查。

---

## 十二、导航

| 上一课 | [10-FlashAttention原理与Triton.md](./10-FlashAttention原理与Triton.md) |
|--------|------------------------------------------------------------------------|
| 下一课 | [12-Assignment2系统优化实战.md](./12-Assignment2系统优化实战.md) |

---

## 附录：术语中英对照

| 中文 | 英文 |
|------|------|
| 分布式数据并行 | Distributed Data Parallel (DDP) |
| 全归约 | AllReduce |
| 进程组 | Process Group |
| 梯度分桶 | Gradient Bucketing |
| 全分片数据并行 | Fully Sharded Data Parallel (FSDP) |
| 集合通信 | Collective Communication |

---

*文档版本：与 CS336 课程主题对齐，代码为教学骨架，生产环境请使用 `torch.nn.parallel.DistributedDataParallel` 并参考官方最佳实践。*
