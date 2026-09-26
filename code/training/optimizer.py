"""AdamW optimizer (decoupled weight decay), implemented without torch.optim.

从零实现的 AdamW（解耦权重衰减），不依赖 torch.optim。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch


class AdamW:
    """AdamW with per-group learning rates and weight decay.

    支持 param_groups，各组可设不同 lr / weight_decay；含一、二阶矩偏差修正。

    Update (per tensor): decay the pre-update parameter, then adaptive step:
    ``θ ← θ - lr * m_hat / (sqrt(v_hat) + ε) - lr * λ * θ``.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[dict]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        items = list(params)
        if not items:
            raise ValueError("AdamW requires at least one parameter group.")
        if isinstance(items[0], dict):
            self.param_groups = [dict(g) for g in items]
        else:
            self.param_groups = [
                {
                    "params": items,
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            ]

        self.state: Dict[torch.Tensor, Dict[str, Any]] = {}
        self._defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        for i, group in enumerate(self.param_groups):
            self.param_groups[i] = self._normalize_group(group)
        seen = set()
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, torch.Tensor):
                    raise TypeError("Optimizer parameters must be tensors.")
                if id(p) in seen:
                    raise ValueError("A parameter must not appear in multiple optimizer entries.")
                seen.add(id(p))

    def _normalize_group(self, group: dict) -> dict:
        g = dict(group)
        g.setdefault("lr", self._defaults["lr"])
        g.setdefault("betas", self._defaults["betas"])
        g.setdefault("eps", self._defaults["eps"])
        g.setdefault("weight_decay", self._defaults["weight_decay"])
        g["params"] = [g["params"]] if isinstance(g["params"], torch.Tensor) else list(g["params"])
        for key in ("lr", "eps", "weight_decay"):
            if not math.isfinite(g[key]) or g[key] < 0:
                raise ValueError(f"{key} must be finite and non-negative.")
        if len(g["betas"]) != 2 or any(not 0 <= beta < 1 for beta in g["betas"]):
            raise ValueError("betas must contain two values in [0, 1).")
        return g

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Clear gradients for all parameters. / 清零梯度。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Perform a single optimization step. / 执行一步参数更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                st = self.state.setdefault(p, {})
                if len(st) == 0:
                    st["step"] = torch.zeros((), dtype=torch.int64, device=p.device)
                    st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = st["exp_avg"], st["exp_avg_sq"]
                st["step"] += 1
                step_t = int(st["step"].item())

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1.0 - beta1**step_t
                bias_c2 = 1.0 - beta2**step_t
                m_hat = exp_avg / bias_c1
                v_hat = exp_avg_sq / bias_c2
                denom = m_hat.div(v_hat.sqrt().add_(eps))

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(denom, alpha=-lr)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Serializable optimizer state (CPU tensors). / 可序列化状态。"""
        flat: List[torch.Tensor] = []
        for group in self.param_groups:
            flat.extend(group["params"])

        state_list: List[Dict[str, Any]] = []
        for p in flat:
            st = self.state.get(p)
            if st is None or len(st) == 0:
                state_list.append({})
            else:
                state_list.append(
                    {
                        "step": st["step"].detach().cpu().clone(),
                        "exp_avg": st["exp_avg"].detach().cpu().clone(),
                        "exp_avg_sq": st["exp_avg_sq"].detach().cpu().clone(),
                    }
                )

        groups_out: List[dict] = []
        offset = 0
        for g in self.param_groups:
            d = {k: v for k, v in g.items() if k != "params"}
            d["params"] = list(range(offset, offset + len(g["params"])))
            offset += len(g["params"])
            groups_out.append(d)

        return {"state": state_list, "param_groups": groups_out}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state; parameters must match current groups. / 加载状态。"""
        state_list = state_dict["state"]
        flat_params: List[torch.Tensor] = []
        for g in self.param_groups:
            flat_params.extend(g["params"])

        if len(flat_params) != len(state_list):
            raise ValueError("Checkpoint parameter count does not match optimizer.")

        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, self.param_groups)
        ):
            raise ValueError("Checkpoint parameter groups do not match optimizer.")
        restored_groups = [
            self._normalize_group({**saved, "params": current["params"]})
            for saved, current in zip(saved_groups, self.param_groups)
        ]
        restored_state = {}
        for p, st_saved in zip(flat_params, state_list):
            if not st_saved:
                continue
            if st_saved["exp_avg"].shape != p.shape or st_saved["exp_avg_sq"].shape != p.shape:
                raise ValueError("Checkpoint moment shapes do not match optimizer parameters.")
            restored_state[p] = {
                "step": st_saved["step"].to(device=p.device, dtype=torch.int64).clone(),
                "exp_avg": st_saved["exp_avg"].to(device=p.device, dtype=p.dtype).clone(),
                "exp_avg_sq": st_saved["exp_avg_sq"].to(device=p.device, dtype=p.dtype).clone(),
            }
        self.param_groups = restored_groups
        self.state = restored_state
