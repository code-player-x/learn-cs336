"""
Minimal distributed data parallel helpers (gradient AllReduce).
分布式数据并行最小示例：梯度 AllReduce 同步。
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed(backend: str = "nccl") -> tuple[int, int]:
    """
    Initialize the default process group (env: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE).
    初始化进程组。单机多卡常用 `torchrun --nproc_per_node=N` 注入环境变量。

    Note / 说明: NCCL 适用于 GPU；纯 CPU 多机请改用 `gloo`。
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_distributed() -> None:
    """Destroy process group / 销毁进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()


class SimpleDDP(nn.Module):
    """
    Wraps a module and AllReduces gradients across processes after backward.
    包装模型：反向时对每个参数的梯度做 AllReduce 并平均。

    Ring-AllReduce / 环形 AllReduce（概念注释）:
    PyTorch `dist.all_reduce` 在 NCCL 后端上通常实现为 ring 或 tree 等算法。
    Ring-AllReduce 将梯度切分为 N 段，在环状拓扑上轮流发送与归约，
    使每步只与邻居通信，总通信量约 O(2*(N-1)/N * data)，带宽利用率高，
    适合大规模张量。此处直接调用 `all_reduce(SUM)` + 平均，等价于各卡梯度求和后同步。
    """

    def __init__(self, module: nn.Module, process_group: Any | None = None) -> None:
        super().__init__()
        self.module = module
        self.process_group = process_group
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._hooks: list[Callable] = []

        if self._world_size > 1:

            def _make_hook() -> Callable[[torch.Tensor], torch.Tensor]:
                group = self.process_group
                ws = self._world_size

                def _hook(grad: torch.Tensor) -> torch.Tensor:
                    if grad is None:
                        return grad
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
                    grad.div_(ws)
                    return grad

                return _hook

            for p in self.module.parameters():
                if p.requires_grad:
                    self._hooks.append(p.register_hook(_make_hook()))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def ddp_train_step(
    model: SimpleDDP,
    x: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """
    One forward-backward-optimizer step under SimpleDDP (gradients synced via hooks).
    单次训练步：前向、反向（钩子内 AllReduce）、优化器更新。
    """
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    return loss.detach()
