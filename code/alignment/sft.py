"""SFT with assistant-only loss. / 监督微调：仅对助手回复计算损失。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

USER_HDR, ASST_HDR = "### User:\n", "### Assistant:\n"


def _format_turns(messages: Sequence[dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    parts, spans = [], []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            parts.append(USER_HDR + content + "\n")
        elif role == "assistant":
            start = sum(len(p) for p in parts) + len(ASST_HDR)
            parts.append(ASST_HDR + content + "\n")
            spans.append((start, start + len(content)))
        else:
            raise ValueError(f"Unknown role: {role}")
    return "".join(parts), spans


def create_sft_dataset(
    conversations: List[dict[str, Any]],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 2048,
) -> List[Any]:
    """Tokenize chats with ``messages``; mask user tokens. tokenizer=None → raw text + char spans only.
    格式化指令数据；tokenizer 为空时返回 formatted_text / assistant_char_spans。
    Example / 示例: ``[{"messages":[{"role":"user","content":"2+2?"},{"role":"assistant","content":"4"}]}]``"""
    if tokenizer is None:
        return [
            {"formatted_text": t, "assistant_char_spans": s}
            for t, s in (_format_turns(c["messages"]) for c in conversations)
        ]
    out: List[dict[str, torch.Tensor]] = []
    for conv in conversations:
        full_text, asst_spans = _format_turns(conv["messages"])
        enc = tokenizer(
            full_text, max_length=max_length, truncation=True,
            return_offsets_mapping=True, add_special_tokens=True,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        for cs, ce in asst_spans:
            for i, (ts, te) in enumerate(enc["offset_mapping"]):
                if ts == te == 0:
                    continue
                if te > cs and ts < ce:
                    loss_mask[i] = 1.0
        labels = input_ids.clone()
        labels[loss_mask == 0] = -100
        out.append({"input_ids": input_ids, "attention_mask": attn, "labels": labels, "loss_mask": loss_mask})
    return out


def compute_sft_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CE on assistant positions only (next-token). / 仅在助手位置交叉熵。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sl, lab, msk = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous(), loss_mask[..., 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1), reduction="none", ignore_index=-100)
    msk = msk.view(-1)
    return (loss * msk).sum() / msk.sum().clamp_min(1.0)


class _SFTDataset(Dataset):
    def __init__(self, samples: List[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.samples[i].items()}


def _collate(batch: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    m = max(b["input_ids"].size(0) for b in batch)
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "labels", "loss_mask"):
        pad_val = -100 if key == "labels" else (0.0 if key == "loss_mask" else 0)
        rows = []
        for b in batch:
            t, pad = b[key], m - b[key].size(0)
            if pad:
                p = torch.full((pad,), pad_val, dtype=t.dtype, device=t.device)
                t = torch.cat([t, p], dim=0)
            rows.append(t)
        out[key] = torch.stack(rows, dim=0)
    return out


@dataclass
class SFTTrainer:
    """Pretrained causal LM + masked SFT train/eval. / 预训练模型、掩码损失、训练与验证。"""
    model_name_or_path: str
    max_length: int = 2048
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path).to(self.device)

    def build_loaders(
        self, train_conversations: List[dict[str, Any]],
        eval_conversations: Optional[List[dict[str, Any]]] = None, batch_size: int = 4,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train = DataLoader(
            _SFTDataset(create_sft_dataset(train_conversations, self.tokenizer, self.max_length)),
            batch_size=batch_size, shuffle=True, collate_fn=_collate,
        )
        ev = None
        if eval_conversations:
            ev = DataLoader(
                _SFTDataset(create_sft_dataset(eval_conversations, self.tokenizer, self.max_length)),
                batch_size=batch_size, shuffle=False, collate_fn=_collate,
            )
        return train, ev

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            tot += compute_sft_loss(
                self.model, batch["input_ids"], batch["labels"], batch["loss_mask"], batch["attention_mask"],
            ).item()
            n += 1
        return tot / max(n, 1)

    def train(
        self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None,
        epochs: int = 1, lr: float = 2e-5, grad_clip: float = 1.0,
    ) -> None:
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for ep in range(epochs):
            pbar = tqdm(train_loader, desc=f"sft {ep+1}/{epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                opt.zero_grad()
                loss = compute_sft_loss(
                    self.model, batch["input_ids"], batch["labels"], batch["loss_mask"], batch["attention_mask"],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            if eval_loader is not None:
                print(f"eval loss: {self.evaluate(eval_loader):.4f}")
