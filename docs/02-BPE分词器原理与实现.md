# Lesson 02: BPE分词器原理与实现

> Stanford CS336：Language Modeling from Scratch — 从零构建语言模型

本节是**面试极高频**主题：几乎所有 LLM / NLP 岗位都会追问分词器如何训练、如何编码、与模型如何衔接。建议把本文中的**算法步骤、复杂度、与 WordPiece 对比、代码骨架**背熟并能白板推导。

---

## 本节概览

你将系统掌握：

1. **为什么**语言模型需要分词器，以及词级 / 字级 / 子词级各自的取舍。
2. **BPE（Byte Pair Encoding）** 从数据压缩到 NLP 子词单元的演变与核心思想。
3. **字节级 BPE** 为何成为 GPT-2 / GPT-3 / 多数开源模型的默认方案。
4. **训练**：预分词、频次统计、迭代合并、词表增长；含**数值例题**。
5. **推理**：按合并优先级应用规则、`encode` / `decode` 与 UTF-8 字节流的关系。
6. **实现**：可运行的 Python（`get_stats`、`merge`、训练、编码、解码、GPT-2 正则预分词、多进程统计）。
7. **优化与现象**：局部更新、并行、数字被拆开、中英文 token 数差异等。
8. **对比**：BPE、WordPiece、SentencePiece。
9. **面试题**：10 道以上带标准答法。

**预计学习时间**：精读 2～3 小时；动手跑通代码 1～2 小时。

---

## 漫画导入 (reference: ![漫画](../comics/ch02-BPE分词器.png))

![漫画](../comics/ch02-BPE分词器.png)

（若本地尚无该图片，可将 `comics/ch02-BPE分词器.png` 放入仓库后显示。漫画用于直觉：**长文本像绳子，BPE 不断把「最常一起出现的两股」拧成一股**，直到词表达到目标大小。）

---

## 一、为什么需要分词器？

### 1.1 NLP pipeline：text → tokens → embeddings → model

现代自回归语言模型的典型数据流为：

```
原始文本 (string)
    ↓  分词器 Tokenizer
离散 token ID 序列 (int[])
    ↓  词嵌入 Embedding lookup
连续向量序列 (float tensor)
    ↓  Transformer 堆叠
logits / 下一 token 分布
```

没有分词器，模型无法把可变长字符串变成**固定词表上的离散符号**，也就无法做查表与 softmax。

### 1.2 Why not character-level or word-level?

| 粒度 | 优点 | 缺点 |
|------|------|------|
| **词级**（空格分词等） | 单 token 语义完整；序列短 | 词表巨大；OOV 严重；形态变化浪费参数 |
| **字符级** | 词表较小；保留字符边界 | 未覆盖的字符仍可能 OOV；序列长 |
| **子词级**（BPE、WordPiece、SentencePiece 等） | 词表大小与序列长度折中 | 需训练分词器 |

### 1.3 The vocabulary size tradeoff（词表大小的权衡）

- 词表 **太小**：序列变长，算力与梯度步数上升。
- 词表 **太大**：嵌入与输出层巨大；低频 token 估计差。
- 常见：**几万级**（32k、50k 等）。

**结论**：子词分词（尤其是**字节级 BPE**）已成为 Decoder-only LLM 的主流选择。

---

## 二、BPE（字节对编码）核心原理

### 2.1 History：originally a data compression algorithm

BPE 最初是 **Philip Gage（1994）** 的**数据压缩**思路：反复合并**最频相邻字节对**。在 NLP 中迁移为**子词单元学习**（如 Sennrich et al., 2016）。

### 2.2 Core idea：iteratively merge most frequent adjacent pairs

1. 最细粒度开始（字节级 BPE：**256 个字节值**）。
2. 统计相邻符号对频次（可加权）。
3. 选频次最高的一对，合并为新 token ID。
4. 全语料应用该合并。
5. 重复至目标词表。

**合并顺序必须保存**；推理时按相同顺序应用。

### 2.3 Byte-level BPE vs character-level BPE

- **字符级**：在 Unicode 码点上合并。
- **字节级**：在 UTF-8 **字节**上合并；初始 **256**；任意文本可编码，**字节级无「未知字符」**（与词级 UNK 概念不同【**UNK** 是 **Unknown token**（未知词元）的缩写】）。

### 2.4 Why byte-level is preferred

实现简单、跨语言一致、与 GPT / tiktoken 生态对齐；代价是 CJK 等往往 **token 更多**（UTF-8 多字节）。

**CJK** 是 **Chinese、Japanese、Korean** 三个词的缩写，指代**中文、日文、韩文**这三种语言/文字系统。

---

## 三、BPE 训练流程（详细步骤）

### Step 1：Initialize vocabulary with 256 byte values + special tokens

基础 **256 字节**；特殊 token 单独分配 ID（各实现不同）。

### Step 2：Pre-tokenize using regex（GPT-2 pattern）

用与推理**完全一致**的正则切分为片段，合并**不跨预分词片段边界**。GPT-2 风格正则允许一个前导空格与后面的词同属一片段，空格可以参与片段内合并。特殊 token 要先单独切出，不参与普通 BPE 合并。

### Step 3：Count adjacent byte pair frequencies

每片段 `encode('utf-8')`，在片段内统计相邻对，按片段频次加权。

### Step 4：Find most frequent pair（break ties lexicographically）

`argmax` 频次；本节采用 CS336 的同频规则：选择 **字节串二元组 `(left_bytes, right_bytes)` 字典序最大** 的 pair，不能按 token ID 或拼接后的字节串比较。参见 [官方变更记录](https://github.com/stanford-cs336/assignment1-basics/blob/main/CHANGELOG.md) 中 original bytes 与 tuple comparison 的说明。

### Step 5：Create new token, merge all occurrences

新 ID 常为 `256,257,...`；在每片段上**从左到右非重叠**合并。

### Step 6：Repeat until target vocab size

更新各片段表示，再统计，直至 `num_merges` 或上限。

### 3.1 Worked example（简例）

若 `"hi"` 字节对 `(104,105)` 在全语料中极高频，第一轮可能合并为 ID `256`，该片段由两字节变一 token。下一轮在新的 ID 序列上重新统计。

### 3.2 手算小练习（面试白板友好）

设**不做**复杂正则，语料仅为重复字符串 `"aaab"` 的 UTF-8 字节（字母 `a`=97，`b`=98），且整段作为一个片段：

- 初始序列：`[97,97,97,98]`。
- 第一轮统计：`(97,97)` 出现 **2 次**（有重叠），`(97,98)` 出现 **1 次** → 合并 `(97,97)→256`，从左到右非重叠合并后得 `[256,97,98]`。新序列包含 `(256,97)` 与 `(97,98)` 两个相邻 pair，各出现 1 次。
- 下一轮在**新序列**上重新数对；后续合并由新频次决定。

**面试要点**：口述「**先全局选 max 频对 → 全片段应用 → 再统计**」；**平局**时说明你的 tie-break（如字典序）。

**Tie-break** 中文叫**平局打破规则**或**决胜规则**，指的是：当出现**多个候选并列最优**时，按什么规则从中选一个。

---

## 四、BPE 推理 / 编码流程

### 4.1 Apply merges in training order（rank）

`merges` 为有序列表：越早训练的合并，在编码中优先级越高（实现上可用「按顺序整段应用」或「选当前可合并对中 rank 最小者」的等价策略）。

### 4.2 Pre-tokenize first, then apply merges per piece

与训练相同正则 → 每片段字节序列 → 应用合并。

### 4.3 Encoding：text → bytes → apply merges → token IDs

### 4.4 Decoding：token IDs → bytes → text

`vocab[id]` 为字节串拼接；`bytes.decode('utf-8')` 得文本。

---

## 五、代码实现

依赖：`pip install regex`。

```python
"""
BPE 教学实现 — CS336 Lesson 02
逐行详细注释版本
"""
# 允许注解里使用还未定义的类型，老python版本兼容
from __future__ import annotations
# 导入多进程库，用于并行统计语料pair频次
import multiprocessing as mp
# Counter用于统计频次；defaultdict带默认值的字典
from collections import Counter, defaultdict
# 类型注解，标明输入输出的数据结构
from typing import Dict, List, Optional, Sequence, Tuple
# regex第三方正则库，支持unicode属性\p{L}\p{N}，比内置re更强
import regex as re

# GPT‑2 的预分词正则：先按规则切块，再对每块做 BPE
# 把文本切为后缀、单词、数字、符号、空白，限制合并在预分词片段内部
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# ============================================================
# 函数作用：GPT2预分词，把原始字符串切为独立片段chunk列表
# 阶段：【训练阶段、推理阶段 两者都用】
# 输入：text：待切分原始字符串
# 输出：List[str]，切分后的各个预分词片段
# ============================================================
def pretokenize_gpt2(text: str) -> List[str]:
    # finditer遍历正则匹配到的全部片段；m.group(0)拿到匹配到的字符串
    # 列表推导收集所有chunk返回
    return [m.group(0) for m in GPT2_SPLIT_PATTERN.finditer(text)]

# ============================================================
# 函数作用：单个字符串片段转UTF‑8字节id列表，每个字节映射为0‑255整数
# 阶段：【训练阶段、推理阶段 两者都用】
# 输入：chunk：一个预分词得到的字符串片段
# 输出：List[int]，字节id序列，全部取值0~255
# ============================================================
def bytes_to_ids(chunk: str) -> List[int]:
    # .encode("utf-8") 将字符串转为utf8字节bytes对象
    # list()把bytes拆解成一个个0~255的int，就是初始id序列
    return list(chunk.encode("utf-8"))

# ============================================================
# 函数作用：统计id序列内部所有相邻pair出现频次，可外部传入字典累加
# 阶段：【工具函数，训练阶段内部使用】
# 输入：ids：字节/子词id序列；counts：可选，外部传入的pair计数字典用于累加
# 输出：Dict[Tuple[int,int], int]，pair→出现次数的统计字典
# ============================================================
def get_stats(ids: Sequence[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    # 如果没有传入counts，新建空字典；否则复用传入的字典实现累加
    counts = counts if counts is not None else {}
    # 遍历ids，取每一组相邻id对
    for i in range(len(ids) - 1):
        # 取出相邻两个id，组成pair元组作为key
        pair = (ids[i], ids[i + 1])
        # dict.get(key,默认值)，key不存在返回0，再+1计数
        counts[pair] = counts.get(pair, 0) + 1
    # 返回统计好的pair频次字典
    return counts

# ============================================================
# 函数作用：序列内把所有相邻的(a,b)pair替换为new_id，从左到右非重叠替换
# 阶段：【工具函数，训练阶段、推理阶段都调用】
# 输入：ids：待处理id列表；pair：待合并(a,b)；new_id：合并后的新token id
# 输出：List[int]，执行替换之后新的id列表
# ============================================================
def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    # 解包待合并的两个id
    a, b = pair
    # 新建空列表存放输出结果
    out: List[int] = []
    # i循环游标；n是输入序列总长度
    i, n = 0, len(ids)
    # while循环遍历整个ids序列
    while i < n:
        # 判断：不是最后一个元素，并且当前位置正好匹配(a,b)待合并pair
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            # 匹配成功，写入合并后的new_id
            out.append(new_id)
            # 向后跳2位，跳过已经合并的两个元素（非重叠）
            i += 2
        else:
            # 不匹配，直接把当前id原样写入输出
            out.append(ids[i])
            # 游标向后走1位
            i += 1
    # 返回完成合并后的新id序列
    return out

# ============================================================
# 函数作用：基于chunk频次加权统计pair；每个chunk_tuple乘上自身出现频次统计pair
# 阶段：【工具函数，训练阶段train_bpe内部调用】
# 输入：chunk_freqs：Dict[Tuple[int,...], int] key=chunk的id元组 value=该片段出现频次
# 输出：Dict[Tuple[int,int], int]，全局pair加权计数字典
# ============================================================
def count_pairs_for_chunks(chunk_freqs: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    # 初始化空字典保存pair统计结果
    stats: Dict[Tuple[int, int], int] = {}
    # 遍历每一个chunk元组，以及该chunk在语料中出现的频次freq
    for chunk_tuple, freq in chunk_freqs.items():
        # 元组转list得到id序列
        ids = list(chunk_tuple)
        # 遍历这个chunk内部全部相邻pair
        for i in range(len(ids) - 1):
            # 取出相邻id对
            p = (ids[i], ids[i + 1])
            # 加权累加：这个chunk出现freq次，所以pair计数 += freq，而不是+1
            stats[p] = stats.get(p, 0) + freq
    # 返回加权统计后的pair频次
    return stats

# ============================================================
# 函数作用：BPE主训练逻辑，输入原始语料，迭代多轮merge生成vocab与merges表
# 阶段：【训练阶段】
# 输入：text_corpus：训练原始大语料字符串；num_merges：要执行多少轮合并；pattern：预分词正则
# 输出：(vocab, merges)
#       vocab:Dict[int,bytes] key=token_id value=对应字节；merges:List[(left,right,new_id)]合并规则列表
# ============================================================
def train_bpe(text_corpus: str, num_merges: int, pattern: re.Pattern = GPT2_SPLIT_PATTERN):
    # 字典：key是chunk的id元组，value是该片段出现次数；Counter专门做计数
    chunk_freqs: Dict[Tuple[int, ...], int] = Counter()
    # 正则遍历语料所有预分词chunk
    for m in pattern.finditer(text_corpus):
        # 将chunk字符串转为字节id列表，再转元组（list不能做字典key，tuple可以）
        t = tuple(bytes_to_ids(m.group(0)))
        # 该chunk计数+1
        chunk_freqs[t] += 1

    # 初始化词表：0~255分别对应单个原始字节
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # merges保存所有合并规则：(左id,右id,新生成id)
    merges: List[Tuple[int, int, int]] = []
    # 第一个新token id从256开始，0‑255是原始字节
    next_id = 256

    # 循环num_merges次，执行指定轮数的BPE合并
    for _ in range(num_merges):
        # 加权统计当前所有chunk上的pair频次
        pair_stats = count_pairs_for_chunks(chunk_freqs)
        # 如果没有任何pair可以合并，提前退出循环
        if not pair_stats:
            break
        # 同频时比较原始字节串二元组，选最大者；不能直接比较 token ID
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], (vocab[p[0]], vocab[p[1]])))
        # 解包本轮要合并的两个id
        left, right = best_pair

        # defaultdict(int) key不存在时默认value=0，用于保存合并之后新chunk的频次
        new_chunk_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
        # 遍历全部旧chunk，逐个执行merge
        for chunk_tuple, freq in chunk_freqs.items():
            # 对当前chunk执行一次best_pair合并
            merged = merge(list(chunk_tuple), best_pair, next_id)
            # 合并后的id列表转为tuple，累加到新的频次字典，带上原来的freq
            new_chunk_freqs[tuple(merged)] += freq
        # 更新chunk_freqs为本轮全部合并完成后的结果
        chunk_freqs = dict(new_chunk_freqs)

        # 更新词表：新token的bytes等于左右两个token的bytes拼接
        vocab[next_id] = vocab[left] + vocab[right]
        # 将本轮合并规则记录进merges列表
        merges.append((left, right, next_id))
        # id自增，下一轮使用下一个id
        next_id += 1
    # 返回训练产物：词表、合并规则
    return vocab, merges

# ============================================================
# 函数作用：构建pair→rank字典，rank越小代表越早训练生成，推理要优先合并
# 阶段：【工具函数，推理阶段使用】
# 输入：merges：训练输出的合并规则列表 List[(a,b,nid)]
# 输出：Dict[Tuple[int,int], int] pair对应合并优先级rank
# ============================================================
def build_merge_ranks(merges: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    # enumerate拿到merges下标r(就是rank)和每一条(a,b,_)；pair做key，rank做value
    return {(a, b): r for r, (a, b, _) in enumerate(merges)}

# ============================================================
# 函数作用：单chunk BPE合并实现；每轮扫描选rank最小pair合并；同rank取最左位置
# 阶段：【推理阶段】
# 输入：ids：单个预分词chunk的原始字节id列表；merges：训练产出合并规则
# 输出：List[int]，BPE合并完成后的子词token id序列
# ============================================================
def encode_piece_by_rank(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    # 构建pair到new_id的映射，方便查到合并后的id
    pair_to_new = {(a, b): nid for a, b, nid in merges}
    # 构建合并优先级rank字典
    merge_ranks = build_merge_ranks(merges)
    # 拷贝输入序列，不修改原输入
    seq = ids[:]

    # 循环：不断找当前序列中可以合并的最优pair，直到没有可合并pair
    while True:
        # best_rank记录当前最优合并的优先级；pos记录位置，初始None代表无
        best_rank, pos = None, None
        # 遍历当前seq，查找全部可合并pair
        for i in range(len(seq) - 1):
            # 取出当前相邻pair
            p = (seq[i], seq[i + 1])
            # 如果这个pair不在merges规则里，跳过
            if p not in merge_ranks:
                continue
            # 获取该pair的合并优先级rank
            r = merge_ranks[p]
            # 如果还没有选pair，或者当前pair rank更小，或者rank相等位置更靠左，则更新最优
            if best_rank is None or r < best_rank or (r == best_rank and pos is not None and i < pos):
                best_rank, pos = r, i
        # best_rank/pos为None，代表没有可以合并的pair，跳出while循环
        if best_rank is None or pos is None:
            break
        # 获取要合并的pair
        p = (seq[pos], seq[pos + 1])
        # 根据pair查到合并后的new_id
        new_id = pair_to_new[p]
        # 切片替换：pos位置两个元素替换成new_id，生成新序列
        seq = seq[:pos] + [new_id] + seq[pos + 2 :]
    # 返回反复合并完成后的token id序列
    return seq

# ============================================================
# 函数作用：单chunk BPE合并实现，按merges列表顺序逐条执行全部合并规则
# 阶段：【推理阶段】
# 输入：ids：单个预分词chunk原始字节id列表；merges：训练产出合并规则
# 输出：List[int]，BPE合并完成后的子词token id序列
# ============================================================
def encode_piece_sequential(ids: List[int], merges: List[Tuple[int, int, int]]) -> List[int]:
    # 拷贝输入序列，不改动外部传入的ids
    seq = ids[:]
    # 按merges训练生成的顺序，逐条取出合并规则
    for left, right, new_id in merges:
        # 调用merge函数：在整个seq上把所有(left,right)替换成new_id，得到新seq
        seq = merge(seq, (left, right), new_id)
    # 全部规则应用完毕，返回结果
    return seq

# ============================================================
# 函数作用：完整文本BPE编码对外入口：预分词→逐chunk转字节id→逐chunkBPE合并→拼接结果
# 阶段：【推理阶段】
# 输入：text：待编码原始字符串；merges：训练得到合并规则；pattern：GPT2预分词正则
# 输出：List[int]，整篇文本编码后的完整token id序列
# ============================================================
def bpe_encode(text: str, merges: List[Tuple[int, int, int]], pattern: re.Pattern = GPT2_SPLIT_PATTERN) -> List[int]:
    # 初始化空列表保存整篇文本所有token id
    out: List[int] = []
    # 对输入文本做GPT‑2预分词，遍历每一个chunk片段
    for piece in pretokenize_gpt2(text):
        # piece字符串转字节id列表；调用encode_piece_sequential做BPE合并；extend把结果追加到out
        out.extend(encode_piece_sequential(bytes_to_ids(piece), merges))
    # 返回整篇文本的token id序列
    return out

# ============================================================
# 函数作用：BPE解码，token id序列还原回原始字符串
# 阶段：【推理阶段】
# 输入：ids：待解码token id序列；vocab：词表字典 key=token_id value=bytes
# 输出：str，还原得到字符串；非法字节自动替换为�不抛异常
# ============================================================
def bpe_decode(ids: Sequence[int], vocab: Dict[int, bytes]) -> str:
    # 1.遍历ids，每个id查表vocab拿到对应的bytes对象
    # 2.b"".join把所有小bytes拼接成一整个大bytes
    # 3.decode("utf‑8")转为字符串；errors="replace"遇到非法字节不会崩溃，替换为�符号
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

# ============================================================
# 函数作用：多进程worker子进程，统计给定一批文本行内部pair局部频次
# 阶段：【训练阶段，仅被parallel_count_pairs调用】
# 输入：lines：分给当前worker的一批文本行；pattern：预分词正则
# 输出：Dict[Tuple[int,int], int]，本分片pair局部计数字典
# ============================================================
def worker_count_pairs(lines: List[str], pattern: re.Pattern) -> Dict[Tuple[int, int], int]:
    # 本worker分片内pair的局部计数字典
    local: Dict[Tuple[int, int], int] = {}
    # 遍历分配给本进程的每一行文本
    for line in lines:
        # 正则预分词，遍历该行切分出的每一个chunk
        for m in pattern.finditer(line):
            # chunk字符串转为字节id列表
            ids = bytes_to_ids(m.group(0))
            # 遍历chunk全部相邻id对
            for i in range(len(ids) - 1):
                # 取出pair
                p = (ids[i], ids[i + 1])
                # 局部计数+1
                local[p] = local.get(p, 0) + 1
    # 返回本分片统计结果，交给主进程汇总
    return local

# ============================================================
# 函数作用：多进程并行统计全语料pair频次；切分语料分片给worker，汇总各进程计数
# 阶段：【训练阶段】
# 输入：corpus_lines：全部训练语料按行组成list；num_workers：进程数量；pattern：预分词正则
# 输出：Dict[Tuple[int,int], int]，全语料汇总pair全局频次字典
# ============================================================
def parallel_count_pairs(
    corpus_lines: List[str], num_workers: int = 4, pattern: re.Pattern = GPT2_SPLIT_PATTERN
) -> Dict[Tuple[int, int], int]:
    # 如果进程数≤1，不走多进程开销，直接单线程执行
    if num_workers <= 1:
        return worker_count_pairs(corpus_lines, pattern)
    # 计算每个分片分配多少行；max(1, ...)防止总行数小于worker数时分片大小为0
    chunk_size = max(1, len(corpus_lines) // num_workers)
    # 将全部语料切分成多个分片shard，每个shard交给一个worker
    shards = [corpus_lines[i : i + chunk_size] for i in range(0, len(corpus_lines), chunk_size)]
    # 创建进程池，with退出时自动关闭进程池，释放资源
    with mp.Pool(num_workers) as pool:
        # starmap把每个shard和pattern传入worker_count_pairs，并行执行，收集各个worker返回结果
        parts = pool.starmap(worker_count_pairs, [(s, pattern) for s in shards])
    # 主进程全局总pair计数字典
    total: Dict[Tuple[int, int], int] = {}
    # 遍历每一个worker返回的局部统计字典
    for d in parts:
        # 遍历该worker的每一组(pair,频次)
        for k, v in d.items():
            # 不同worker相同pair的计数累加，得到全局总频次
            total[k] = total.get(k, 0) + v
    # 返回汇总完成的全局pair频次字典
    return total

# ============================================================
# 脚本主入口演示：训练BPE模型，再编码解码，断言校验无损还原
# 阶段：【演示脚本，训练+推理链路完整跑通做验证】
# 输入：脚本内置sample测试字符串，无外部入参
# 输出：控制台打印token ids与OK；断言失败抛出AssertionError，无函数返回值
# ============================================================
if __name__ == "__main__":
    # 测试样例文本，包含重复英文、中文，覆盖ASCII与UTF‑8多字节字符
    sample = "hello hello hello 你好"
    # 执行BPE训练，做10轮合并，得到词表vocab、合并规则merges
    vocab, merges = train_bpe(sample, num_merges=10)
    # 使用训练好的merges对sample文本做推理编码，得到token id序列
    ids = bpe_encode(sample, merges)
    # 解码id序列还原字符串；断言：解码结果必须等于原始sample，不等直接抛AssertionError
    assert bpe_decode(ids, vocab) == sample
    # 在控制台打印编码得到的token ids
    print("token ids:", ids)
    # 断言没有报错，打印OK，代表整套链路运行正常
    print("OK")

```

**说明**：`encode_piece_sequential` 按 `merges` 顺序逐条合并；`encode_piece_by_rank` 为等价实现。`parallel_count_pairs` 用于大语料**分块统计 pair**，主进程合并 Counter 后再选 `best_pair` 并与单进程训练逻辑对齐。

---

## 六、性能优化技巧

1. **避免全量重计**：合并后只更新受影响片段的局部统计。
2. **并行预分词与并行统计**：`parallel_count_pairs`；merge 步骤保持全局一致顺序。
3. **哈希表**：`(a,b) -> rank` 与 `(a,b) -> new_id`。
4. **原生实现**：tiktoken / Rust 后端。

**复杂度（朴素实现）**

- $M$：合并条数（`num_merges`），即训练做了多少轮 merge。
- $T$：训练语料所有片段的总长度（按字节/token 计）。
- $L$：待编码序列的长度（按字节/token 计）。

**训练：$O(M \cdot T)$**

- 每轮：扫一遍全部片段，统计相邻对频次 → $O(T)$；再扫一遍应用 merge → $O(T)$。
- 共 $M$ 轮，所以 $O(M \cdot T)$。

**编码：$O(L \cdot M)$**

- 按训练顺序逐条应用 merge，每条规则扫一遍当前序列 → $O(L)$。
- 共 $M$ 条规则，所以 $O(L \cdot M)$。

**为什么叫“朴素”**

- 每轮/每条规则都全量重扫。
- 优化版：训练用优先队列 + 增量更新；编码用 `pair → rank` 表选择当前可合并 pair。rank 表只减少规则查询成本，不会自动变成单次扫描；本节逐次重扫的 rank 版最坏仍为 $O(L^2)$。链表配合优先队列的编码在常见假设下可达 $O(L\log L)$（不含 rank 表预处理）。
- 教学代码为直观，用朴素版。

**一句话记**
训练是“$M$ 轮 × 全语料 $T$”，编码是“$M$ 条规则 × 全序列 $L$”。

---

## 七、常见现象解释

### 7.1 为什么数字会被拆成多个 token？

数字的切分同时受预分词正则与合并词表影响。本节 GPT-2 正则将连续数字作为片段，再按 BPE 合并；其他 tokenizer 可能先限制每个数字片段的位数。`12345 → 12 / 34 / 5` 只是可能的示例，不是固定规则。

### 7.2 Why same word tokenizes differently with/without leading space

GPT-2 风格正则把「可选空格 + 词」绑在一起，`hello` 与 `  hello  ` 是不同片段。

### 7.3 Why Chinese uses more tokens than English（UTF-8 encoding）

**中文 token 为什么比英文多（UTF-8）**

1. **字节起点不同**：汉字 UTF-8 多为 3 字节，英文 ASCII 多为 1 字节，同样一个字/字母，中文占的字节数就多。
2. **语料偏置**：训练语料英文占多数，高频字节对偏英文，英文被大量合并成短 token；中文对 rank 靠后、合并机会少，更多以字节形式保留。

两条叠加 → 中文 token 更多。

---

## 八、BPE vs WordPiece vs SentencePiece

| Feature | BPE | WordPiece | SentencePiece |
|---------|-----|-----------|----------------|
| **合并准则** | 相邻对 **频次** 最高 | 似然类目标（依实现） | 可配 **BPE 或 Unigram** |
| **空白与多语** | 依赖预分词 | 子词 + `##` 等 | 空白可编码，**不依赖英文空格** |
| **初始单元** | 字节级 BPE 为 **256 字节** | 通常从字符等基本符号开始 | 通常为 Unicode 字符，可选 byte fallback；算法可为 BPE 或 Unigram |
| **典型模型** | GPT、Llama 等 | BERT 系 | T5、多语模型等 |

**补充**：SentencePiece 是**工具/流程**；内部可跑 BPE 或 Unigram。WordPiece 与 BPE 的**目标函数**不同，面试常考。

---

## 九、面试高频题（10 道主问题 + 2 道追问，均附详细答案）

### 1. BPE 分词器的训练流程是什么？

**答**：预分词 → 片段转 UTF-8 字节 → 加权统计相邻对 → 选频次最高对（平局字典序）→ 生成新 ID 并全局应用合并 → 重复至目标词表。推理时用相同预分词与 `merges`。

### 2. 字节级 BPE 相比字符级 BPE 有什么优势？

**答**：256 单元固定；任意 UTF-8 文本可表示；实现简单、生态一致。

### 3. BPE 的时间复杂度是多少？如何优化？

**朴素复杂度**
- 训练：$O(M \cdot T)$
  - $M$：merge 条数；$T$：语料总长度。
  - 每轮全量重扫统计 + 应用合并，共 $M$ 轮。
- 编码：$O(L \cdot M)$
  - $L$：待编码序列长度。
  - 逐条应用 merge，每条扫一遍序列，共 $M$ 条。

**优化手段**
1. **增量统计**：只更新受合并影响的局部 pair 频次，不每轮全量重扫。具体复杂度取决于合并次数、索引与优先队列实现，不能统一保证 $O(T\log T)$。
2. **并行 map-reduce**：分片统计 pair 频次，再合并结果（代码里的 `parallel_count_pairs`）。
3. **`merge_ranks` 哈希**：`pair → rank` 建表，编码时 $O(1)$ 查优先级，避免线性找规则。
4. **原生代码**：热点用 C/C++/Rust 实现（如 HuggingFace `tokenizers`），减少 Python 开销。

**一句话记**
朴素是“每轮全量重扫”，优化靠“增量 + 并行 + 哈希 + 原生”。

### 4. 为什么中文在英文分词器中消耗更多 token？

**答**：UTF-8 三字节汉字 + 语料偏英文导致合并机会少。

### 5. BPE、WordPiece、SentencePiece 的区别？

**答**：见第八节；核心在**准则**与**是否依赖空格预分词**。

### 6. 预分词（pre-tokenization）的作用是什么？

**答**：限制合并不跨边界、稳定统计、与英文词边界对齐；须训练/推理一致。

### 7. 分词器的词表大小如何选择？

**答**：嵌入与 softmax 成本 vs 序列长度；常见几十 k；结合实验与压缩率。

### 8. 如何处理未见过的特殊字符？

**答**：字节级下变为 UTF-8 字节，通常无需 `<unk>`。

### 9. BPE 合并时如何处理平局（tie-breaking）？

**答**：固定规则（如 pair 字典序）保证可复现。

### 10. 分词器对模型性能有什么影响？

**答**：影响序列长度、稀有词/数字/代码切分、多语公平性及训练-推理一致性。

### 11.（追问）为何必须按训练得到的 merge 顺序编码？

**答**：顺序定义唯一确定性切分；乱序会改变 ID 分布。

### 12.（追问）多进程能否每块单独训练一套 BPE？

**答**：不能得到统一词表；应**全局汇总**统计再统一 merge。

---

## 十、本节小结

- 分词器是 LLM 的**入口**：`text ↔ token IDs`。
- BPE **迭代合并最频相邻对**；字节级简单稳健。
- **训练**保存 `merges` 与 `vocab`；**推理**预分词与合并顺序一致。

---

## 十一、下一步预告

**Lesson 03：Transformer 架构** 将衔接 token 嵌入与 Decoder-only：注意力、残差、层归一化与因果掩码。

---

## 附录：与课程其它文档的衔接

| 文档 | 说明 |
|------|------|
| [00-课程总览与学习路线](00-课程总览与学习路线.md) | CS336 全局路线图 |
| Assignment 1（Basics） | 常含 BPE + Transformer + 训练循环；实现须与作业说明中的 **tie-break、预分词** 完全一致 |

**结语**：把**初始化—预分词—统计—合并—编解码**闭环跑通一次，面试会轻松很多。

---

## 附录：自测练习题（选做）

1. 将第五节代码保存为 `bpe_tutorial.py`，运行 `python bpe_tutorial.py`，确认输出 `OK`。
2. 对同一短句比较 `encode_piece_sequential` 与 `encode_piece_by_rank` 的输出是否一致（同一片段、同一 `merges`）。
3. 用 `tiktoken.get_encoding("cl100k_base")` 对中英混合句 `encode`，观察 token 数与「UTF-8 字节数」的比值，并用本节第七部分解释。
4. 说明为何 `parallel_count_pairs` 只加速**统计**，而**选 merge 与应用 merge** 仍须在全局一致顺序下进行。
5. 口述：WordPiece 与 BPE 的**目标函数**差异，各举一个在工业界的代表模型。

---

## 十二、BPE 核心算法分步复盘（白板）

1. **初始化**：256 字节 ID。  
2. **预分词**：GPT-2 正则切片段。  
3. **统计**：相邻对加权频次。  
4. **合并**：最大频；平局固定规则。  
5. **新 ID**：递增写入 `merges`。  
6. **应用**：全语料替换 `(a,b)→new`。  
7. **迭代**至目标词表。  

---

## 十三、面试十大高频题（合并速查）

| # | 问题 | 要点 |
|---|------|------|
| 1 | 训练流程？ | 见第十二节 |
| 2 | 字节级原因？ | 256、无字符 OOV |
| 3 | 中文 token？ | UTF-8 多字节 + 语料 |
| 4 | 复杂度？ | 朴素 $O(MT)$ 量级 |
| 5 | 优化？ | 并行统计、哈希、C++ |
| 6 | vs WordPiece/Unigram？ | 频次 / 似然 / LM |
| 7 | 预分词？ | 限制范围、空格附着 |
| 8 | UNK？ | 字节级通常无 |
| 9 | 词表大小？ | 压缩率与参数折中 |
| 10 | merge 顺序？ | 推理须与训练一致 |

---
