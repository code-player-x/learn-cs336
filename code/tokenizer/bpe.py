# SPDX-License-Identifier: MIT
"""
Byte Pair Encoding (BPE) tokenizer — GPT-2 style pre-tokenization and byte-level merges.

字节对编码（BPE）分词器：GPT-2 风格预分词 + 字节级合并。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable

# GPT-2 / tiktoken style pretokenization (requires `regex` for \\p{L} / \\p{N}).
# 与 GPT-2 / tiktoken 一致的预分词正则（需 `regex` 以支持 Unicode 属性类）。
try:
    import regex

    _GPT2_SPLIT_RE = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:  # pragma: no cover - fallback for environments without `regex`
    _GPT2_SPLIT_RE = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s]+|\s+(?!\S)|\s+"""
    )


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Count frequencies of all adjacent symbol pairs in a token-id sequence.

    统计 token 序列中所有相邻符号对的出现次数。

    Args:
        ids: Sequence of integer token IDs (bytes or merged symbols).

    Returns:
        Mapping (left_id, right_id) -> count.
    """
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every non-overlapping occurrence of `pair` with `new_id`, left-to-right.

    从左到右、非重叠地将所有 `pair` 替换为 `new_id`（标准 BPE 合并规则）。

    Args:
        ids: Current sequence of token IDs.
        pair: Two consecutive IDs to merge (first, second).
        new_id: New merged token ID.

    Returns:
        New sequence after merging.
    """
    a, b = pair
    if not ids:
        return []
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _pretokenize(text: str) -> list[str]:
    """Split text into GPT-2-style chunks (regex). / 使用 GPT-2 正则切分片段。"""
    if not text:
        return []
    return [m.group(0) for m in _GPT2_SPLIT_RE.finditer(text)]


def _build_vocab_from_merges(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Reconstruct id -> bytes mapping from base bytes + merge list. / 由字节基底与合并表重建词表。"""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for a, b in merges:
        vocab[next_id] = vocab[a] + vocab[b]
        next_id += 1
    return vocab


class BPETokenizer:
    """
    Byte-level BPE tokenizer: train merges on UTF-8 bytes, encode/decode text.

    字节级 BPE：在 UTF-8 字节上学习合并，支持文本编码与解码。
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self._target_vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab(self) -> dict[int, bytes]:
        """
        Mapping from token ID to byte sequence (merged tokens are concatenations).

        token ID 到字节串的映射（合并 token 为字节的拼接）。
        """
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (including 256 byte tokens). / 当前词表大小（含 256 个单字节）。"""
        return len(self._vocab)

    def train(self, text: str, vocab_size: int | None = None) -> None:
        """
        Train BPE on a text corpus: start from 256 bytes, iteratively merge pairs.

        在语料上训练 BPE：从 256 字节出发，迭代合并最高频相邻对。

        Args:
            text: Training corpus (UTF-8 string).
            vocab_size: Target vocabulary size; defaults to constructor value.
        """
        target = vocab_size if vocab_size is not None else self._target_vocab_size
        if target < 256:
            raise ValueError("vocab_size must be >= 256")
        num_merges = target - 256
        if num_merges == 0:
            self.merges = []
            self._vocab = {i: bytes([i]) for i in range(256)}
            return

        chunks = _pretokenize(text)
        if not chunks:
            self.merges = []
            self._vocab = {i: bytes([i]) for i in range(256)}
            return

        # Each chunk as list of byte values (ints 0–255).
        seqs: list[list[int]] = [list(ch.encode("utf-8")) for ch in chunks]

        self.merges = []
        self._vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for _ in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq in seqs:
                pair_counts.update(get_stats(seq))
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            candidates = [p for p, c in pair_counts.items() if c == best_count]
            pair = min(candidates)  # tie-break: lexicographic order / 平局按字典序

            seqs = [merge(seq, pair, next_id) for seq in seqs]
            self.merges.append(pair)
            self._vocab[next_id] = self._vocab[pair[0]] + self._vocab[pair[1]]
            next_id += 1

        self._target_vocab_size = target

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs: pretokenize, UTF-8 bytes, apply merges in order.

        将文本编码为 token：预分词 → UTF-8 字节 → 按训练顺序依次应用合并（与训练一致）。
        """
        if not text:
            return []
        out: list[int] = []
        for chunk in _pretokenize(text):
            seq = list(chunk.encode("utf-8"))
            for idx, pair in enumerate(self.merges):
                seq = merge(seq, pair, 256 + idx)
            out.extend(seq)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """
        Decode token IDs back to a UTF-8 string.

        将 token ID 序列解码为 UTF-8 字符串。
        """
        data = bytearray()
        for i in ids:
            if i not in self._vocab:
                raise KeyError(f"Unknown token id: {i}")
            data.extend(self._vocab[i])
        return bytes(data).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """
        Save merges and metadata to JSON (portable, human-readable).

        将合并规则与元数据保存为 JSON（可移植、可读）。
        """
        payload = {
            "model": "bpe_byte_gpt2pretok",
            "merges": [list(p) for p in self.merges],
            "target_vocab_size": self._target_vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        Load tokenizer state from a file written by `save`.

        从 `save` 写入的文件恢复分词器状态。
        """
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        merges_raw = payload.get("merges", [])
        self.merges = [(int(a), int(b)) for a, b in merges_raw]
        self._target_vocab_size = int(payload.get("target_vocab_size", 256 + len(self.merges)))
        self._vocab = _build_vocab_from_merges(self.merges)

    @classmethod
    def from_file(cls, path: str) -> BPETokenizer:
        """Construct tokenizer and load from path. / 构造并从文件加载。"""
        tok = cls()
        tok.load(path)
        return tok


if __name__ == "__main__":
    sample = (
        "Hello, world! 你好，世界！\n"
        "BPE learns frequent pairs like 'th' and multi-byte UTF-8 sequences.\n"
        "The tokenizer uses GPT-2 style regex pretokenization."
    )
    print("Training BPE on sample text... / 在示例文本上训练 BPE...\n")
    tok = BPETokenizer(vocab_size=320)
    tok.train(sample, vocab_size=320)
    print(f"vocab_size = {tok.vocab_size}, merges = {len(tok.merges)}")
    ids = tok.encode(sample)
    back = tok.decode(ids)
    assert back == sample, "Round-trip should preserve text / 往返应无损"
    print(f"encode length: {len(ids)} tokens")
    print(f"decode matches: {back == sample}")
    # save / load round-trip
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        p = tmp.name
    try:
        tok.save(p)
        t2 = BPETokenizer.from_file(p)
        assert t2.encode(sample) == ids
    finally:
        import os

        os.unlink(p)
    print("Demo OK. / 演示通过。")
