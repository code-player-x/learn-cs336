"""
FlashAttention-style forward (tiling + online softmax) in pure PyTorch.
纯 PyTorch 的 FlashAttention 风格前向：分块 + 在线 softmax。
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import torch
import torch.nn.functional as F


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Standard dot-product attention: softmax(QK^T / sqrt(d)) V.
    标准点积注意力，用于与 flash 实现对照。
    """
    d = Q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    logits = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = F.softmax(logits, dim=-1)
    return torch.matmul(attn, V)


def flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Memory-efficient attention via block tiling and online softmax.
    通过分块与在线 softmax（维护行方向 running max m 与 sum l）避免物化完整 T×T 分数矩阵。

    Shapes: (..., T, D); returns (..., T, D). Last dim is head_dim.
    """
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError("Q, K, V must have the same shape")
    d = Q.size(-1)
    scale = 1.0 / math.sqrt(d)

    T = Q.size(-2)
    out = torch.empty_like(Q)

    # Flatten batch heads to 3D for the kernel loop
    q3 = Q.reshape(-1, T, d)
    k3 = K.reshape(-1, T, d)
    v3 = V.reshape(-1, T, d)
    o3 = out.reshape(-1, T, d)

    br = bc = block_size
    for b in range(q3.size(0)):
        for tr in range(0, T, br):
            tr_end = min(tr + br, T)
            qb = q3[b, tr:tr_end, :]  # (rows, d)
            rows = qb.size(0)
            m = torch.full((rows,), -float("inf"), device=Q.device, dtype=Q.dtype)
            l = torch.zeros((rows,), device=Q.device, dtype=Q.dtype)
            o = torch.zeros((rows, d), device=Q.device, dtype=Q.dtype)

            for tc in range(0, T, bc):
                tc_end = min(tc + bc, T)
                kb = k3[b, tc:tc_end, :]
                vb = v3[b, tc:tc_end, :]
                s = torch.matmul(qb, kb.transpose(-2, -1)) * scale  # (rows, block)

                m_block = s.max(dim=-1).values
                m_new = torch.maximum(m, m_block)
                p = torch.exp(s - m_new.unsqueeze(-1))
                l_new = torch.exp(m - m_new) * l + p.sum(dim=-1)
                o = (
                    torch.exp(m - m_new).unsqueeze(-1) * o
                    + torch.matmul(p, vb)
                )
                m, l = m_new, l_new

            o3[b, tr:tr_end, :] = o / l.unsqueeze(-1).clamp(min=torch.finfo(o.dtype).tiny)

    return out


def benchmark_flash_vs_standard(
    seq_len: int = 512,
    dim: int = 64,
    batch: int = 2,
    block_size: int = 64,
    device: str | None = None,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> Tuple[dict, dict]:
    """
    Compare peak memory and wall time of standard vs flash attention.
    对比标准注意力与 Flash 风格的峰值显存与耗时（若可用 CUDA 则记录显存）。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    q = torch.randn(batch, seq_len, dim, device=dev, dtype=torch.float32)
    k = torch.randn(batch, seq_len, dim, device=dev, dtype=torch.float32)
    v = torch.randn(batch, seq_len, dim, device=dev, dtype=torch.float32)

    def run_peak_mem(fn):
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n_warmup):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        for _ in range(n_iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = (time.perf_counter() - t0) / n_iters
        peak = (
            torch.cuda.max_memory_allocated(dev)
            if dev.type == "cuda"
            else None
        )
        return elapsed, peak

    std_elapsed, std_peak = run_peak_mem(lambda: standard_attention(q, k, v))
    flash_elapsed, flash_peak = run_peak_mem(
        lambda: flash_attention_forward(q, k, v, block_size=block_size)
    )

    standard_stats = {"time_ms": std_elapsed * 1000, "peak_mem_bytes": std_peak}
    flash_stats = {"time_ms": flash_elapsed * 1000, "peak_mem_bytes": flash_peak}
    return standard_stats, flash_stats
