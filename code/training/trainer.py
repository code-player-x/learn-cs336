"""Training loop, evaluation, checkpoints, and text chunking dataset.

训练循环、验证、检查点与文本分块数据集。
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]


PathLike = Union[str, Path]


class TextDataset(Dataset):
    """Load raw text, encode to token ids, and chunk into fixed-length sequences.

    读取文本、编码为 token id，再切为固定长度块；``labels`` 为下一词（右移一位）。
    """

    def __init__(
        self,
        sources: Union[PathLike, Sequence[PathLike]],
        seq_len: int,
        encode: Callable[[str], List[int]],
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        self.seq_len = seq_len
        self.encode = encode

        paths: List[Path] = []
        if isinstance(sources, (str, Path)):
            paths = [Path(sources)]
        else:
            paths = [Path(p) for p in sources]

        pieces: List[str] = []
        for p in paths:
            pieces.append(p.read_text(encoding=encoding))
        text = "".join(pieces)

        ids: List[int] = encode(text)
        if len(ids) < self.seq_len + 1:
            raise ValueError(
                f"Need at least {self.seq_len + 1} tokens after encoding; got {len(ids)}."
            )
        self._data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return (self._data.numel() - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError("TextDataset index out of range")
        start = idx * self.seq_len
        block = self._data[start : start + self.seq_len + 1]
        x = block[:-1].contiguous()
        y = block[1:].contiguous()
        return x, y


class Trainer:
    """LM trainer: one epoch loop, eval, checkpoints, clipping, logging.

    语言模型训练器：单 epoch 循环、验证、检查点、梯度裁剪与日志（loss、PPL、lr、吞吐）。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        max_grad_norm: float = 1.0,
        ignore_index: int = -100,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = float(max_grad_norm)
        self.ignore_index = ignore_index
        self.log_interval = max(1, log_interval)

        self.model.to(self.device)
        self.global_step = 0
        self.epoch = 0

        self._base_lrs: List[float] = [float(g["lr"]) for g in self.optimizer.param_groups]
        self._sched_max_lr = float(getattr(scheduler, "max_lr", max(self._base_lrs) or 1.0))

    def _sync_lr_from_scheduler(self) -> float:
        """Set param groups' lr from scheduler at current global step. / 按调度器更新各组 lr。"""
        lr = float(self.scheduler.get_lr(self.global_step))
        scale = lr / self._sched_max_lr if self._sched_max_lr > 0 else 0.0
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base * scale
        return lr

    def _forward_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        if logits.dim() != 3 or logits.shape[:-1] != y.shape:
            raise ValueError("Expected model(x) logits of shape (B, T, V).")
        valid_tokens = (y != self.ignore_index).sum()
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        return loss / valid_tokens.clamp_min(1)

    def train_one_epoch(self) -> Dict[str, float]:
        """Run a full pass over ``train_loader`` and return average metrics. / 训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.perf_counter()

        iterator = self.train_loader
        if tqdm is not None:
            iterator = tqdm(self.train_loader, desc=f"train epoch {self.epoch}", leave=False)

        for batch in iterator:
            # An entirely masked batch has no learning signal. In particular,
            # do not advance AdamW/weight decay or the schedule for this batch.
            tok = int((batch[1] != self.ignore_index).sum())
            if tok == 0:
                continue
            lr = self._sync_lr_from_scheduler()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(batch)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += float(loss.detach()) * tok
            total_tokens += tok
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                elapsed = time.perf_counter() - t0
                tput = total_tokens / elapsed if elapsed > 0 else 0.0
                ppl = math.exp(min(float(loss.detach()), 20.0))
                msg = (
                    f"step={self.global_step} loss={float(loss):.4f} ppl={ppl:.2f} "
                    f"lr={lr:.2e} tok/s={tput:.0f}"
                )
                if tqdm is None:
                    print(msg)
                else:
                    iterator.set_postfix(loss=float(loss), lr=lr, ppl=ppl, tok_s=tput)

        if total_tokens == 0:
            raise ValueError("train_loader contains no non-ignored target tokens.")
        self.epoch += 1
        avg_loss = total_loss / total_tokens
        elapsed = time.perf_counter() - t0
        out = {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20.0)),
            "lr": self._sync_lr_from_scheduler(),
            "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0.0,
        }
        return out

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Validation loop; returns mean loss and perplexity. / 验证集评估。"""
        if self.val_loader is None:
            raise ValueError("val_loader is not set.")
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        iterator = self.val_loader
        if tqdm is not None:
            iterator = tqdm(self.val_loader, desc="eval", leave=False)

        try:
            for batch in iterator:
                tok = int((batch[1] != self.ignore_index).sum())
                if tok == 0:
                    continue
                loss = self._forward_loss(batch)
                total_loss += float(loss) * tok
                total_tokens += tok
        finally:
            self.model.train(was_training)

        if total_tokens == 0:
            raise ValueError("val_loader contains no non-ignored target tokens.")
        avg_loss = total_loss / total_tokens
        return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20.0))}

    def save_checkpoint(self, path: PathLike, **extra: Any) -> None:
        """Save state and RNGs for epoch-boundary resumption, not a mid-epoch cursor.

        保存模型、优化器、调度参数与随机状态；不保存 epoch 内数据游标。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sched = {
            "max_lr": getattr(self.scheduler, "max_lr", None),
            "min_lr": getattr(self.scheduler, "min_lr", None),
            "warmup_steps": getattr(self.scheduler, "warmup_steps", None),
            "max_steps": getattr(self.scheduler, "max_steps", None),
        }
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": sched,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "base_lrs": self._base_lrs,
            "sched_max_lr": self._sched_max_lr,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "extra": extra,
        }
        if torch.cuda.is_available():
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        if self.train_loader.generator is not None:
            payload["loader_rng_state"] = self.train_loader.generator.get_state()
        try:
            import numpy as np
        except ImportError:  # NumPy is optional for this trainer.
            pass
        else:
            name, keys, pos, has_gauss, cached = np.random.get_state()
            payload["numpy_rng_state"] = (name, keys.tolist(), pos, has_gauss, cached)
        torch.save(payload, path)

    def load_checkpoint(self, path: PathLike, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load weights and optimizer state; restore step counters. / 加载检查点。"""
        path = Path(path)
        loc = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self._base_lrs = [float(x) for x in ckpt.get("base_lrs", self._base_lrs)]
        self._sched_max_lr = float(ckpt.get("sched_max_lr", self._sched_max_lr))
        for key, value in ckpt.get("scheduler", {}).items():
            if value is not None and hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "cuda_rng_states" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_states"]])
        if "loader_rng_state" in ckpt and self.train_loader.generator is not None:
            self.train_loader.generator.set_state(ckpt["loader_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            try:
                import numpy as np
            except ImportError:
                pass
            else:
                name, keys, pos, has_gauss, cached = ckpt["numpy_rng_state"]
                np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
        return ckpt.get("extra", {})
