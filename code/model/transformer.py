"""
LLaMA-style Transformer language model (PyTorch).
LLaMA 风格的 Transformer 语言模型（PyTorch 实现）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ffn_hidden_dim(d_model: int) -> int:
    """SwiGLU hidden: round int(8/3 * d_model) up to multiple of 256."""
    raw = int(2 * 4 * d_model // 3)
    return ((raw + 255) // 256) * 256


def init_linear(module: nn.Linear) -> None:
    """Xavier uniform for linear layers (fan-in/out balanced)."""
    nn.init.xavier_uniform_(module.weight, gain=1.0)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_embedding(module: nn.Embedding) -> None:
    """Normal init for embeddings. / 词嵌入正态初始化。"""
    nn.init.normal_(module.weight, mean=0.0, std=0.02)


def init_kaiming_linear(module: nn.Linear) -> None:
    """Kaiming (He) uniform for GLU linear maps (SiLU/ReLU-friendly). / GLU 线性层 Kaiming 均匀初始化。"""
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (no mean centering).
    均方根层归一化（不做去均值）。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half and rotate for RoPE."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to q and k.
    对 q、k 应用旋转位置编码。

    Expected layout: q, k (B, n_heads, T, Dh); cos, sin broadcast to (1, 1, T, Dh).
    期望布局：q、k 为 (B, n_heads, T, Dh)；cos、sin 可广播至 (1, 1, T, Dh)。
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    RoPE: precompute inverse frequencies and cache cos/sin per position.
    RoPE：预计算逆频率并按位置缓存 cos/sin。
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos, sin shaped for apply_rotary_emb (broadcast over batch & heads).
        返回可与 apply_rotary_emb 配合的 cos、sin（可对 batch 与头广播）。
        """
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE, causal mask, and optional GQA.
    带 RoPE、因果掩码与可选 GQA 的多头注意力。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 8192,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, seq_len=seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            else:
                mask = attention_mask.to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward: down(SiLU(gate(x)) * up(x)).
    SwiGLU 前馈网络。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = _ffn_hidden_dim(d_model)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (RMSNorm -> sublayer -> residual).
    Pre-norm Transformer 块。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        max_position_embeddings: int,
        rope_base: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model,
            n_heads,
            n_kv_heads=n_kv_heads,
            max_position_embeddings=max_position_embeddings,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """
    Causal language model: embeddings, stack of blocks, norm, tied LM head.
    因果语言模型：嵌入、堆叠块、归一化、权重绑定的 LM 头。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 8192,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    max_position_embeddings,
                    rope_base,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head: Optional[nn.Linear] = None
        self._init_weights()

    def tie_weights(self) -> None:
        """Tie output projection with input embeddings (LLaMA-style). / 输出层与词嵌入权重绑定。"""
        if self.lm_head is None:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self) -> None:
        init_embedding(self.embed_tokens)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("gate_proj") or name.endswith("up_proj"):
                    init_kaiming_linear(module)
                else:
                    init_linear(module)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns logits of shape (batch, seq_len, vocab_size).
        返回形状为 (batch, seq_len, vocab_size) 的 logits。
        """
        if self.lm_head is None:
            self.tie_weights()

        attention_mask = None
        if self.pad_token_id is not None:
            attention_mask = input_ids != self.pad_token_id

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        assert self.lm_head is not None
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive sampling with temperature and nucleus (top-p).
        使用温度与 nucleus (top-p) 的自回归采样。
        """
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(out)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                mask = cumsum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                probs = F.softmax(sorted_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(sorted_idx, -1, sampled)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return out
