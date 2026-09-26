"""Cosine learning-rate schedule with linear warmup.

带线性 warmup 的余弦学习率调度。
"""

from __future__ import annotations

import math


class CosineWithWarmup:
    """Linear warmup, then cosine decay from ``max_lr`` to ``min_lr``.

    第一个 warmup 更新使用 ``max_lr / warmup_steps``，随后升到峰值并余弦衰减。
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer.")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if warmup_steps > max_steps:
            raise ValueError("warmup_steps must not exceed max_steps.")
        if not math.isfinite(max_lr) or not math.isfinite(min_lr) or not 0 <= min_lr <= max_lr:
            raise ValueError("Learning rates must be finite and satisfy 0 <= min_lr <= max_lr.")
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)

    def get_lr(self, step: int) -> float:
        """Learning rate at global step ``step`` (0-indexed). / 第 ``step`` 步的学习率。"""
        step = int(step)
        if step < 0:
            step = 0

        if step >= self.max_steps:
            return self.min_lr

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)

        decay_steps = max(self.max_steps - self.warmup_steps, 1)
        t = float(step - self.warmup_steps) / float(decay_steps)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr + (self.max_lr - self.min_lr) * cos
