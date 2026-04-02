"""Cosine learning-rate schedule with linear warmup.

带线性 warmup 的余弦学习率调度。
"""

from __future__ import annotations

import math


class CosineWithWarmup:
    """Linear warmup, then cosine decay from ``max_lr`` to ``min_lr``.

    前 ``warmup_steps`` 步线性从 0 升到 ``max_lr``，随后在剩余步数上余弦衰减至 ``min_lr``。
    """

    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)

    def get_lr(self, step: int) -> float:
        """Learning rate at global step ``step`` (0-indexed). / 第 ``step`` 步的学习率。"""
        step = int(step)
        if step < 0:
            step = 0

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)

        if step >= self.max_steps:
            return self.min_lr

        decay_steps = max(self.max_steps - self.warmup_steps, 1)
        t = float(step - self.warmup_steps) / float(decay_steps)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr + (self.max_lr - self.min_lr) * cos
