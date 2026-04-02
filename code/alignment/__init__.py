"""Alignment: SFT & GRPO. / 对齐：监督微调与 GRPO。"""
from .grpo import (
    GRPOTrainer,
    compute_group_advantages,
    compute_rewards,
    generate_solutions,
    grpo_loss,
)
from .sft import SFTTrainer, compute_sft_loss, create_sft_dataset

__all__ = [
    "SFTTrainer",
    "compute_sft_loss",
    "create_sft_dataset",
    "generate_solutions",
    "compute_rewards",
    "compute_group_advantages",
    "grpo_loss",
    "GRPOTrainer",
]
