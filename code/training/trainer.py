"""Training loop, evaluation, checkpoints, and text chunking dataset.

训练循环、验证、检查点与文本分块数据集。
"""

from __future__ import annotations

import math
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
        if logits.dim() != 3:
            raise ValueError("Expected model(x) logits of shape (B, T, V).")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return loss

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
            lr = self._sync_lr_from_scheduler()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(batch)
            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            bs = batch[0].size(0)
            tok = bs * batch[0].size(1)
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

        self.epoch += 1
        avg_loss = total_loss / max(total_tokens, 1)
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
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        iterator = self.val_loader
        if tqdm is not None:
            iterator = tqdm(self.val_loader, desc="eval", leave=False)

        for batch in iterator:
            loss = self._forward_loss(batch)
            bs = batch[0].size(0)
            tok = bs * batch[0].size(1)
            total_loss += float(loss) * tok
            total_tokens += tok

        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20.0))}

    def save_checkpoint(self, path: PathLike, **extra: Any) -> None:
        """Save model, optimizer, scheduler hyperparameters, step, and epoch. / 保存检查点。"""
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
            "extra": extra,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: PathLike, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load weights and optimizer state; restore step counters. / 加载检查点。"""
        path = Path(path)
        loc = map_location or str(self.device)
        ckpt = torch.load(path, map_location=loc)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self._base_lrs = [float(x) for x in ckpt.get("base_lrs", self._base_lrs)]
        self._sched_max_lr = float(ckpt.get("sched_max_lr", self._sched_max_lr))
        return ckpt.get("extra", {})
