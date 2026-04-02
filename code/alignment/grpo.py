"""GRPO: sample → reward → policy update + KL to ref. / 组相对策略优化与 KL 约束。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


def generate_solutions(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], num_samples: int,
    max_new_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, device: Optional[str] = None,
) -> List[str]:
    """``num_samples`` completions per prompt (order: p0×K, p1×K, …). / 每提示 K 条续写，按提示展平。"""
    device = device or next(model.parameters()).device
    model.eval()
    out: List[str] = []
    for prompt in prompts:
        enc = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt", add_special_tokens=True).items()}
        for _ in range(num_samples):
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=max(1e-5, temperature), top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            plen = enc["input_ids"].shape[1]
            out.append(tokenizer.decode(gen[0, plen:], skip_special_tokens=True).strip())
    return out


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def compute_rewards(solutions: Sequence[str], ground_truths: Sequence[str]) -> torch.Tensor:
    """1 if normalized match else 0. / 规则奖励：答案匹配为 1。"""
    if len(solutions) != len(ground_truths):
        raise ValueError("length mismatch")
    return torch.tensor([1.0 if _norm(a) == _norm(b) else 0.0 for a, b in zip(solutions, ground_truths)], dtype=torch.float32)


def compute_group_advantages(rewards: torch.Tensor, num_samples: int) -> torch.Tensor:
    """A = r - mean(r) within each group of ``num_samples``. / 组内减均值优势。"""
    if rewards.numel() % num_samples:
        raise ValueError("len(rewards) must divide num_samples")
    g = rewards.view(-1, num_samples)
    return (g - g.mean(dim=1, keepdim=True)).reshape(-1)


def _sum_logprobs_on_suffix(model: PreTrainedModel, input_ids: torch.Tensor, start: int) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    lp = F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    L, seq = lp.size(1), input_ids.size(1)
    lo, hi = max(0, start - 1), seq - 1
    mask = torch.zeros(L, device=input_ids.device, dtype=lp.dtype)
    if lo < hi:
        mask[lo:hi] = 1.0
    return (lp * mask).sum(dim=1)


def grpo_loss(
    model: PreTrainedModel, ref_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str], solutions: Sequence[str], advantages: torch.Tensor, kl_coef: float,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-E[A·log π] + kl_coef·(logπ - logπ_ref) on suffix. / 续写段策略梯度 + KL。"""
    device = device or next(model.parameters()).device
    if len(prompts) != len(solutions) or len(solutions) != advantages.numel():
        raise ValueError("align prompts, solutions, advantages")
    advantages = advantages.to(device)
    pg_terms, kl_terms = [], []
    for prompt, sol, adv in zip(prompts, solutions, advantages):
        enc = {k: v.to(device) for k, v in tokenizer(prompt + sol, return_tensors="pt", add_special_tokens=True).items()}
        plen = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].shape[1]
        ids = enc["input_ids"]
        if ids.shape[1] <= plen:
            continue
        lp = _sum_logprobs_on_suffix(model, ids, plen)
        with torch.no_grad():
            lr_ = _sum_logprobs_on_suffix(ref_model, ids, plen)
        af = adv.float() if adv.dim() == 0 else adv.squeeze().float()
        pg_terms.append(-(af * lp.squeeze(0)))
        kl_terms.append((lp - lr_.detach()).squeeze(0))
    if not pg_terms:
        z = next(model.parameters()).sum() * 0.0
        return z, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    loss = torch.stack(pg_terms).mean() + kl_coef * torch.stack(kl_terms).mean()
    return loss, torch.stack(kl_terms).mean().detach(), torch.stack([t.detach() for t in pg_terms]).mean()


@dataclass
class GRPOTrainer:
    """Trainable policy + frozen ref; KL monitored. / 可训练策略与冻结参考模型。"""
    model_name_or_path: str
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def train_step(
        self, prompts: List[str], ground_truths: List[str], num_samples: int,
        kl_coef: float = 0.05, lr: float = 1e-6, max_new_tokens: int = 128,
    ) -> dict[str, float]:
        """One round: gen → reward → adv → step. / 单步：生成、奖励、优势、更新。"""
        self.model.eval()
        sols = generate_solutions(self.model, self.tokenizer, prompts, num_samples, max_new_tokens=max_new_tokens, device=self.device)
        gt_rep = [g for g in ground_truths for _ in range(num_samples)]
        adv = compute_group_advantages(compute_rewards(sols, gt_rep), num_samples).to(self.device)
        self.model.train()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad()
        loss, kl_t, pg_t = grpo_loss(
            self.model, self.ref_model, self.tokenizer,
            [p for p in prompts for _ in range(num_samples)], sols, adv, kl_coef, device=self.device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        rm = compute_rewards(sols, gt_rep).mean().item()
        return {"loss": float(loss.item()), "kl": float(kl_t.item()), "pg": float(pg_t.item()), "reward_mean": rm}

    def train(
        self, prompts: List[str], ground_truths: List[str], num_samples: int = 4, steps: int = 10,
        kl_coef: float = 0.05, lr: float = 1e-6, max_new_tokens: int = 128,
    ) -> None:
        for i in tqdm(range(steps), desc="grpo"):
            print(f"step {i+1}: {self.train_step(prompts, ground_truths, num_samples, kl_coef, lr, max_new_tokens)}")
