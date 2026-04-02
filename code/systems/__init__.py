"""CS336 systems utilities: FlashAttention (tiled) and minimal DDP."""

from .ddp import (
    SimpleDDP,
    cleanup_distributed,
    ddp_train_step,
    setup_distributed,
)
from .flash_attention import (
    benchmark_flash_vs_standard,
    flash_attention_forward,
    standard_attention,
)

__all__ = [
    "SimpleDDP",
    "benchmark_flash_vs_standard",
    "cleanup_distributed",
    "ddp_train_step",
    "flash_attention_forward",
    "setup_distributed",
    "standard_attention",
]
