"""Small token-clipped GRPO example with an optional reference penalty.

Rollouts preserve generated token IDs and detached old-policy probabilities.
The trainer samples the unwarped policy (temperature=1, top_p=1, top_k=0).
This is a single-update teaching example, not a production RL trainer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


@dataclass
class _Rollout:
    input_ids: torch.Tensor
    prompt_length: int
    text: str
    old_logprobs: Optional[torch.Tensor] = None


def _response_logprobs(model: PreTrainedModel, ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """One probability per response token, including any generated EOS."""
    if not 1 <= prompt_length <= ids.numel():
        raise ValueError("a rollout needs a nonempty prompt prefix")
    logits = model(
        input_ids=ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
    ).logits[0, prompt_length - 1:-1].float()
    targets = ids[prompt_length:]
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _generate_rollouts(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], num_samples: int,
    max_new_tokens: int, temperature: float, top_p: float, device: Optional[str], *, record_logprobs: bool,
) -> List[_Rollout]:
    if not isinstance(num_samples, int) or not isinstance(max_new_tokens, int) or num_samples < 1 or max_new_tokens < 1:
        raise ValueError("num_samples and max_new_tokens must be positive")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    if record_logprobs and (temperature != 1.0 or top_p != 1.0):
        raise ValueError("training rollouts must use the unwarped policy")
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    # A fresh config avoids inherited top_k/repetition penalties/forced tokens,
    # which would silently change the behavior distribution.
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        top_p=top_p, top_k=0, pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    out: List[_Rollout] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for prompt in prompts:
                enc = {k: v.to(device) for k, v in tokenizer(prompt, return_tensors="pt", add_special_tokens=True).items()}
                plen = enc["input_ids"].shape[1]
                if plen == 0:
                    raise ValueError("a prompt must tokenize to at least one token")
                for _ in range(num_samples):
                    gen = model.generate(**enc, generation_config=config)
                    ids = gen[0].detach().clone()
                    text = tokenizer.decode(ids[plen:], skip_special_tokens=True).strip()
                    old = _response_logprobs(model, ids, plen).detach().clone() if record_logprobs else None
                    out.append(_Rollout(ids, plen, text, old))
    finally:
        model.train(was_training)
    return out


def generate_solutions(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], num_samples: int,
    max_new_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, device: Optional[str] = None,
) -> List[str]:
    """Inference-only completions ordered p0×K, p1×K, …; preserves model mode.

    Decoded strings alone are not sufficient for a GRPO rollout: training uses
    the internal token-level path so EOS, whitespace and BPE boundaries survive.
    """
    return [r.text for r in _generate_rollouts(
        model, tokenizer, prompts, num_samples, max_new_tokens, temperature, top_p, device, record_logprobs=False,
    )]


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def compute_rewards(solutions: Sequence[str], ground_truths: Sequence[str]) -> torch.Tensor:
    """Rule reward: 1 for a normalized exact match, otherwise 0."""
    if len(solutions) != len(ground_truths):
        raise ValueError("length mismatch")
    return torch.tensor([float(_norm(a) == _norm(b)) for a, b in zip(solutions, ground_truths)], dtype=torch.float32)


def compute_group_advantages(
    rewards: torch.Tensor, num_samples: int, *, normalize_std: bool = True, eps: float = 1e-8,
) -> torch.Tensor:
    """Group-centered rewards, optionally divided by population std + eps.

    Std normalization is the default GRPO variant; disabling it is an explicit
    centered-only variant. Equal-reward groups have zero advantage.
    """
    if not isinstance(num_samples, int) or num_samples < 2 or rewards.ndim != 1 or rewards.numel() % num_samples:
        raise ValueError("rewards must be 1D and its length divisible by num_samples >= 2")
    if not math.isfinite(eps) or eps <= 0 or not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite and eps positive")
    g = rewards.detach().float().reshape(-1, num_samples)
    centered = g - g.mean(dim=1, keepdim=True)
    if normalize_std:
        centered = centered / (g.std(dim=1, keepdim=True, unbiased=False) + eps)
    return centered.reshape(-1)


def grpo_loss(
    model: PreTrainedModel, ref_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str], solutions: Sequence[str], advantages: torch.Tensor, kl_coef: float,
    device: Optional[str] = None, *, old_logprobs: Optional[Sequence[torch.Tensor]] = None,
    clip_eps: float = 0.2, rollouts: Optional[Sequence[_Rollout]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Response-token PPO clipping, averaged within responses then across them.

    Old log probabilities must be captured *before* the update. For the string
    interface they correspond to encode(prompt) + encode(solution, no specials),
    not encode(prompt + solution). Trainer rollouts keep exact generated IDs.
    The nonnegative k3 estimator is exp(ref-new) - (ref-new) - 1: its expectation
    under current-policy sampling is forward KL. On old-policy rollouts it is an
    off-policy approximation; clipping does not remove this sampling bias.
    """
    if len(prompts) != len(solutions) or len(solutions) != advantages.numel():
        raise ValueError("align prompts, solutions, advantages")
    if not 0 < clip_eps < 1 or not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("clip_eps must be in (0, 1), kl_coef finite and nonnegative")
    if not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite")
    device = device or next(model.parameters()).device
    if rollouts is None:
        if old_logprobs is None or len(old_logprobs) != len(solutions):
            raise ValueError("GRPO requires detached old per-response-token log probabilities")
        rollouts = []
        for prompt, sol, old in zip(prompts, solutions, old_logprobs):
            prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            suffix = tokenizer(sol, add_special_tokens=False)["input_ids"]
            ids = torch.tensor(prefix + suffix, dtype=torch.long, device=device)
            rollouts.append(_Rollout(ids, len(prefix), sol, old.detach()))
    if len(rollouts) != len(solutions):
        raise ValueError("rollout count mismatch")
    pg_terms, kl_terms = [], []
    was_training, ref_was_training = model.training, ref_model.training
    # Disable dropout for the same policy used by generation; gradients remain enabled.
    model.eval()
    ref_model.eval()
    try:
        for rollout, adv in zip(rollouts, advantages.detach().to(device).reshape(-1)):
            ids = rollout.input_ids.to(device)
            if rollout.old_logprobs is None:
                raise ValueError("rollout is missing old policy probabilities")
            lp = _response_logprobs(model, ids, rollout.prompt_length)
            old = rollout.old_logprobs.detach().to(device=device, dtype=lp.dtype)
            if old.shape != lp.shape or not torch.isfinite(old).all():
                raise ValueError("old probabilities must be finite and match the response length")
            if lp.numel() == 0:
                continue
            log_ratio = lp - old
            # Equivalent sign-specific log-space min(r*A, clip(r)*A).
            # Avoid exp(huge)*0 and needless overflow for clipped positives.
            if adv > 0:
                pg = -adv * torch.exp(log_ratio.clamp_max(math.log1p(clip_eps)))
            elif adv < 0:
                pg = -adv * torch.exp(log_ratio.clamp_min(math.log1p(-clip_eps)))
            else:
                pg = lp * 0.0
            if kl_coef:
                with torch.no_grad():
                    ref_lp = _response_logprobs(ref_model, ids, rollout.prompt_length)
                delta = ref_lp - lp
                kl = torch.expm1(delta) - delta
            else:
                kl = lp * 0.0
            if not torch.isfinite(pg).all() or not torch.isfinite(kl).all():
                raise FloatingPointError("nonfinite GRPO objective; reduce the update size/check policy probabilities")
            pg_terms.append(pg.mean())
            kl_terms.append(kl.mean())
    finally:
        model.train(was_training)
        ref_model.train(ref_was_training)
    if not pg_terms:
        z = next(model.parameters()).sum() * 0.0
        return z, z.detach(), z.detach()
    pg_mean, kl_mean = torch.stack(pg_terms).mean(), torch.stack(kl_terms).mean()
    return pg_mean + kl_coef * kl_mean, kl_mean.detach(), pg_mean.detach()


@dataclass
class GRPOTrainer:
    """Single on-policy rollout/update round, frozen reference, no critic."""
    model_name_or_path: str
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("tokenizer needs a pad token or EOS token")
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
        """Generate exact tokens + old probs → group rewards → clipped update."""
        if not prompts or len(prompts) != len(ground_truths):
            raise ValueError("need nonempty, aligned prompts and ground_truths")
        if not isinstance(num_samples, int) or num_samples < 2 or not math.isfinite(lr) or lr <= 0:
            raise ValueError("num_samples must be >= 2 and lr must be positive")
        rollouts = _generate_rollouts(
            self.model, self.tokenizer, prompts, num_samples, max_new_tokens, 1.0, 1.0,
            self.device, record_logprobs=True,
        )
        solutions = [r.text for r in rollouts]
        truths = [g for g in ground_truths for _ in range(num_samples)]
        rewards = compute_rewards(solutions, truths)
        adv = compute_group_advantages(rewards, num_samples).to(self.device)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss, kl, pg = grpo_loss(
            self.model, self.ref_model, self.tokenizer,
            [p for p in prompts for _ in range(num_samples)], solutions, adv, kl_coef,
            device=self.device, rollouts=rollouts,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item(), "kl": kl.item(), "pg": pg.item(), "reward_mean": rewards.mean().item()}

    def train(
        self, prompts: List[str], ground_truths: List[str], num_samples: int = 4, steps: int = 10,
        kl_coef: float = 0.05, lr: float = 1e-6, max_new_tokens: int = 128,
    ) -> None:
        for i in tqdm(range(steps), desc="grpo"):
            print(f"step {i+1}: {self.train_step(prompts, ground_truths, num_samples, kl_coef, lr, max_new_tokens)}")
