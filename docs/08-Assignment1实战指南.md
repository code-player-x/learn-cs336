# Lesson 08：Assignment 1 实战指南

> Stanford CS336：Language Modeling from Scratch — 面试导向「从零到可训练 Transformer LM」全链路整合

---

## 一、标题与定位

本节是 **CS336 基础篇（Assignment 1 / Basics）** 的实战总览：把 **BPE 分词器、Decoder-only Transformer 语言模型（含 RoPE、MHA、RMSNorm、SwiGLU）、交叉熵损失、手写 AdamW、带学习率调度的训练循环、Top-p 文本生成** 收束为一条可执行路径，并对照单元测试与小型训练建立「实现—调试—复盘」闭环。

**学完应能**：向面试官 **逐步展开** 你的 LLM 实现；清楚 **设计取舍、超参选择与踩坑经历**；用 **STAR** 讲清项目背景与个人贡献。

**预计时间**：精读本文约 2～3 小时；若已克隆官方作业仓库，端到端实现与调通测试约 **3～7 天**（视基础与每日投入而定）。

**成功标准（学习视角）**：`uv run pytest` 全绿；在 toy 数据上 loss 随步数下降；`generate` 能输出比均匀随机略连贯的续写（不要求 ChatGPT 级别）。

---

## 二、核心概念（零基础友好）

### 2.1 这条作业在解决什么问题？

语言模型在给定前文的情况下，为**下一个 token** 在词表上分配概率分布。Assignment 1 要求你**不依赖** `torch.optim.AdamW` 等高层封装（以课程 PDF 为准），从零拼出：

**文本 → 整数序列 → 嵌入向量 → 多层因果 Transformer → 词表 logits → 交叉熵 → 反向传播 → AdamW 更新**。

### 2.2 为什么需要 BPE？

字符级序列太长，词表级分词对未登录词不友好。**字节级 BPE** 在「子词」与「字节」之间折中：既能表示任意 UTF-8 文本，又能通过 merge 得到高频片段，控制词表大小 $V$。

### 2.3 为什么是 Decoder-only Transformer？

自回归语言建模只需「看见当前位置及以前」，因此用 **因果自注意力**（causal mask），不需要 Encoder 的双向注意力。每层通常是：**归一化 → 自注意力（+ 残差）→ 归一化 → FFN（+ 残差）**，具体是 Pre-LN 还是 Post-LN **以作业与测试为准**。

### 2.4 RoPE 一句话

**旋转位置编码（RoPE）** 把位置信息编码进 $Q,K$ 的二维子空间中，通过旋转实现相对位置关系；现代 LLM 常用 RoPE 替代可学习绝对位置嵌入。实现时要注意：**旋转施加在 head 维度的正确子空间上**，且与 **因果 mask** 分工明确（RoPE 管位置，mask 管「不能看未来」）。

### 2.5 RMSNorm / SwiGLU 一句话

- **RMSNorm**：用均方根归一化，比 LayerNorm 略省参数，常见于 LLaMA 系。
- **SwiGLU FFN**：$\mathrm{SwiGLU}(x) = (\mathrm{Swish}(xW_1) \odot xW_2) W_3$（形状以作业定义为准），表达能力与 GELU-MLP 不同，是当前大模型常用 FFN 形态。

### 2.6 交叉熵在做什么？

对每个位置，模型输出 $V$ 维 logits，与「真实下一个 token」做 **多分类交叉熵**。语言建模通常把 `(B, T, V)` 与右移一位的 `labels` 对齐后 **展平** 成 `(B*(T-1), V)` 与 `(B*(T-1),)` 再计算（忽略 padding 位置时用 `ignore_index`）。

### 2.7 AdamW 与「手写」的意义

**AdamW** 把权重衰减**解耦**在参数更新上，而不是混进梯度里的 L2。手写一遍是为确认你理解 `m,v`、偏差修正、`ε`、以及 `param_groups`（例如 bias 不衰减）。

### 2.8 学习率调度

常见组合：**warmup**（步数或比例）+ **cosine decay** 或 **linear decay**。调度对象通常是 **当前 step 的有效学习率** $\eta_t$，再代入 AdamW 更新式。

### 2.9 Top-p 采样

从最高概率的 token 开始累加概率，直到超过阈值 $p$，再在该集合内按重归一化概率采样；可避免长尾噪声，比纯 greedy 更自然。

---

## 三、Assignment 1 总览：从零训练 Transformer LM

### 3.1 你将完成什么

- 实现 **全部关键组件**，使模型在真实或 toy 语料上**可前向、可反传、可更新**。
- 用 **`pytest`** 保证分词器、模型、优化器与训练逻辑与课程规范一致。
- 形成可讲述的 **端到端故事**：数据如何进模型、loss 如何算、生成如何做。

### 3.2 组件清单（逐项自检）

| # | 组件 | 要点 |
|---|------|------|
| 1 | **BPE Tokenizer** | 预分词（常为 GPT-2 风格正则）、字节映射、pair 统计、迭代 merge、`encode` / `decode`、特殊 token 与 tie-break |
| 2 | **Transformer LM** | Token embedding；**RoPE** 与 **MHA**（多头、因果 mask）；**RMSNorm**；**SwiGLU FFN**；最终 **lm_head**（$D \to V$） |
| 3 | **Cross-entropy loss** | 时间维 **shift**、展平、`ignore_index` 处理 padding |
| 4 | **AdamW** | $m_t,v_t$、偏差修正、**解耦** `weight_decay`、`param_groups` |
| 5 | **Training loop** | `zero_grad` → forward → loss → `backward` →（可选 `clip_grad_norm_`）→ `step`；**LR schedule** |
| 6 | **Text generation** | 自回归逐 token；**top-p**（nucleus）采样；`eval` + `torch.no_grad()` |

具体 API 名称、是否要求 **weight tying**、词表索引范围、特殊 token 列表，**以官方 PDF 与测试为准**。

### 3.3 与前置课程的关系

| 前置课 | 本节如何用到 |
|--------|----------------|
| [Lesson 02 BPE](02-BPE分词器原理与实现.md) | 预分词、字节、merge、encode/decode |
| [Lesson 03 Transformer](03-Transformer架构详解.md) | Decoder-only 堆叠、残差与归一化顺序 |
| [Lesson 04 RoPE/MHA](04-多头注意力与RoPE.md) | 因果注意力、RoPE 施加维度 |
| [Lesson 05 RMSNorm/SwiGLU](05-RMSNorm-SwiGLU-GQA.md) | 现代 LLM 子层 |
| [Lesson 06 AdamW](06-AdamW优化器实现.md) | 矩估计、偏差修正、解耦权重衰减 |
| [Lesson 07 训练与采样](07-训练循环与损失函数.md) | CE、调度、Top-p |

---

## 四、端到端代码走读（串联所有组件）

下面用 **记号**：batch $B$，序列长度 $T$，宽度 $D$，层数 $L$，头数 $H$，词表 $V$，头维 $d_\text{head}=D/H$（需整除）。

### 4.1 数据进入模型之前

1. 原始字符串 `text`。
2. `ids = tokenizer.encode(text)` → `List[int]`，长度约与字节/子词数相关。
3. 构造训练 batch：`input_ids` 形状 `(B, T)`，`dtype=torch.long`，$\max(\text{ids}) < V$。
4. `input_ids = input_ids.to(device)`，`model = model.to(device)`。

### 4.2 前向（Transformer LM）

1. **Embedding**：`x = embed(input_ids)` → `(B, T, D)`。
2. **RoPE**：在注意力内部对 $Q,K$ 按位置旋转（实现细节见作业；注意 **不要** 把 RoPE 当成因果 mask 的替代品）。
3. **L 个 Decoder block**（示意）：  
   `x = x + attn(norm(x))`；`x = x + ffn(norm(x))`（Pre-LN 写法为例）。
4. **Causal MHA**：注意力 logits 为 `(B, H, T, T)`（或等价形状），对 $j>i$ 的位置加 mask，softmax 后与未来无关。
5. **输出头**：`logits = lm_head(norm(x))` → `(B, T, V)`。

### 4.3 损失

```python
# 示意：无 padding 的最简对齐
logits = model(input_ids)   # (B, T, V)
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, V),
    input_ids[:, 1:].reshape(-1),
)
```

若有 padding，对 `labels` 置 `-100`（或作业规定值）并在 `cross_entropy(..., ignore_index=...)` 中忽略。

### 4.4 反向与优化

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 推荐保留
optimizer.step()
# 下一步前：scheduler.step() 或按 step 更新 lr（依实现而定）
```

### 4.5 生成（Top-p）

1. `model.eval()`，`torch.no_grad()`。
2. 从 prompt 得到 `input_ids`，循环：forward 取最后一个位置 logits → 可选温度缩放 → **top-p 过滤与重归一化** → `torch.multinomial` 采样下一个 id → 拼接到序列直到 `max_new_tokens` 或 EOS。

---

## 五、项目结构与文件组织

### 5.1 推荐目录布局（与社区常见作业仓库兼容）

官方仓库命名可能为 `assignment1-basics`、`cs336_basics` 等；下面为**思路示例**（包名以你克隆版本为准）。

```
assignment1/
├── cs336_basics/                    # 可 import 的包名（示例）
│   ├── __init__.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── bpe.py                   # 训练、merges、encode/decode
│   │   └── regex.py                 # GPT-2 预分词（若要求独立文件）
│   ├── model/
│   │   ├── __init__.py
│   │   ├── transformer.py         # LM：Embedding、Blocks、lm_head
│   │   ├── attention.py             # 因果 MHA + RoPE
│   │   └── modules.py               # RMSNorm、SwiGLU 等
│   ├── optim/
│   │   └── adamw.py                 # 手写 AdamW，不 import torch.optim
│   └── train/
│       ├── loop.py                  # 训练循环、调度器
│       ├── data.py                  # Dataset / DataLoader
│       └── generate.py              # 采样（可选独立）
├── scripts/
│   └── train.py                     # 入口：解析参数、启动训练
├── tests/
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_adamw.py
├── pyproject.toml
└── README.md
```

### 5.2 模块依赖方向

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  tokenizer  │     │    model     │     │   AdamW     │
│  str↔ids    │     │  nn.Module   │     │  参数更新    │
└──────┬──────┘     └──────┬───────┘     └──────▲──────┘
       │                 │                     │
       │    input_ids    │    logits/loss      │ step()
       └────────────────┴─────────────────────┘
                          train/loop.py
```

- **Tokenizer** 不依赖 `torch.nn`：只负责 `str ↔ List[int]`，便于单独测试。
- **Model** 只依赖张量与模块约定。
- **Optimizer** 依赖 `param.grad`；在 `model.to(device)` **之后** 构造，避免 state 设备错误。

### 5.3 与本仓库 `code/` 的对应关系

本学习项目中的参考实现可对照阅读：

- `code/tokenizer/` — BPE
- `code/model/` — Transformer LM
- `code/training/` — 循环与优化器

**官方 `tests/` 与 PDF 为最高准则**；本地 `code/` 用于类比结构，不要假设 API 完全一致。

---

## 六、测试策略：单元测试、梯度检查、形状验证

### 6.1 单元测试（pytest）

- **Tokenizer**：固定小语料训练 BPE；检查 `decode(encode(text))` 在允许规则下与原文一致；merge 顺序与 tie-break 与参考一致。
- **模型**：固定种子与极小 `B,T,D,L,H`，对 `output shape`、`因果性`（未来位置不应影响过去输出）做检验。
- **AdamW**：若干步后参数应变化；`weight_decay` 仅作用于应衰减的组；与 `torch.optim.AdamW` 在简单网络上数值接近（若作业允许对拍）。

### 6.2 梯度检查

- 对关键模块使用 **有限差分** 或 `torch.autograd.gradcheck`（在 double、极小输入上）验证自定义算子/重组逻辑（若作业要求）。
- 训练一步后检查 **`param.grad is not None`** 且非全零（排除被冻结参数）。

### 6.3 形状验证

- 在 `forward` 关键处 `assert` 或一次性打印：`embed (B,T,D)`、`attn (B,H,T,T)`、`logits (B,T,V)`。
- **`cross_entropy`**：`C` 必须在最后一维；否则先 `permute` / `view`。

### 6.4 过拟合单 batch

- 取 `B=1`、重复同一段文本，训练数十～数百步，**loss 应明显下降**——证明「数据—标签—loss—反传」闭环正确。

### 6.5 运行测试（uv）

```bash
uv sync                                  # 首次安装依赖
uv run pytest                            # 全部测试
uv run pytest -x tests/test_tokenizer.py # 单文件，遇错即停
uv run pytest -k "bpe"                   # 按名称子串筛选
```

若无 `uv`，可用 `pytest` 或 `python -m pytest`。

**习惯**：改 tokenizer 只跑 `test_tokenizer`；改模型只跑 `test_model`；全绿后再集成。

---

## 七、常见 Bug 与调试技巧

### 7.1 形状不匹配（Shape mismatch）

| 现象 | 常见原因 |
|------|-----------|
| matmul 维度错误 | $QK^\top$ 中 head 维与 `d_head` 混淆；`transpose` 写错 |
| `cross_entropy` 报错 | logits 与 labels 长度差 1；`V` 不在最后一维 |
| attention 广播失败 | 未 reshape 为 `(B, H, T, d)`；mask 长度不是 `T` |

**方法**：固定 `B=1`、小 `T`，逐步打印 `tensor.shape`。

### 7.2 因果 mask 未正确施加

- **症状**：验证集或生成时「偷看未来」，loss 异常低但不泛化；或 attention 权重在非因果位置非零。
- **处理**：显式构造 `(T,T)` 上三角 mask；softmax 前将禁止位置设为 `-inf` 或 `torch.finfo(dtype).min`；检查 **半精度** 下是否出现全 `-inf` 行导致 NaN。

### 7.3 RoPE 施加在错误维度

- **症状**：位置不变性异常、长序列 ppl 崩、与参考实现对拍失败。
- **处理**：对照论文/讲义，确认旋转作用于 **每个 head 内** 的成对维度；`cos/sin` 缓存与 `position` 对齐；不要与 embedding 加性位置编码混用除非作业要求。

### 7.4 BPE merge 顺序与平局（tie-break）

- **症状**：encode 结果与官方不一致、测试偶发失败。
- **处理**：**全局**选最高频 pair；平局按 PDF（常见 **字典序**）打破；**推理**严格按训练得到的 **merge 列表顺序**应用；预分词正则与字节映射与训练一致。

### 7.5 数值精度问题

- **症状**：loss NaN、Inf、训练几步后崩溃。
- **处理**：降低 LR；`clip_grad_norm_`；检查 RMSNorm 的 `eps`；混合精度时用 `GradScaler`；检查 masked softmax 数值稳定性。

### 7.6 其他高频问题

- **设备不一致**：`Expected all tensors on same device` → 数据、`model`、optimizer state 同设备。
- **假内存泄漏**：列表里累积未 `detach()` 的 loss；每步用 `loss.item()` 记日志。
- **评估/生成**：忘记 `model.eval()` 与 `torch.no_grad()`。

---

## 八、训练配置：模型大小、batch、序列长度、学习率

以下为 **自学 toy / 小语料** 的常用起点；真实作业以 PDF 与机器显存为准。

| 项 | Toy / 调试建议 | 说明 |
|----|----------------|------|
| $D$（d_model） | 128～384 | 先保证能过拟合小数据 |
| $L$（层数） | 2～6 | 深模型更难调，先浅后深 |
| $H$（头数） | $D$ 整除 $d_\text{head}$，如 4～8 | 与 RoPE 实现一起测 |
| $T$（序列长度） | 128～512 | 显存 $\propto B \cdot T^2$（注意力） |
| $B$（batch） | 从 1～8 起 | OOM 则减 $B$ 或梯度累积 |
| 学习率 $\eta$ | $1\mathrm{e}{-4}$～$3\mathrm{e}{-4}$ 量级试探 | 配合 warmup |
| weight decay $\lambda$ | $0.01$～$0.1$（常见范围） | bias/LayerNorm 常不衰减 |
| 调度 | warmup + cosine | warmup 步数占总步数 1%～10% |

**面试表述**：说明你如何 **先小模型过拟合** 再放大；如何看 **train/val loss** 与 **梯度范数**。

---

## 九、如何运行：uv、pytest、训练脚本

### 9.1 环境（uv）

```bash
cd /path/to/assignment1
uv sync
uv run python scripts/train.py --config configs/toy.yaml   # 示例，以仓库为准
```

### 9.2 测试

```bash
uv run pytest
uv run pytest tests/test_model.py -v
```

### 9.3 训练脚本通常做什么

- 解析 YAML/CLI：数据路径、词表大小、模型维度、训练步数、设备。
- 构建 `Dataset` / `DataLoader`。
- 初始化 `model`、`optimizer`、`lr_scheduler`。
- 循环：取 batch → forward → loss → backward → clip → step → 日志（loss、lr、可选 grad norm）。

---

## 十、性能基准与预期结果（非官方保证）

以下仅为判断「是否离谱」的**粗参考**；真实曲线依赖词表、数据、超参与种子。

| 观察项 | 粗参考 |
|--------|--------|
| 随机初始化、未训练 | loss 常接近 $\ln V$（自然对数） |
| Toy 过拟合 | 数十～数百步内 loss 明显下降 |
| 小语料真实训练 | 验证 loss 可能波动；需调 LR 与正则 |
| 生成质量 | 极小模型以「连贯子串、复述训练片段」为目标即可 |

**面试表述**：强调 **loss 曲线、梯度范数、token 级准确率**，而不是「像不像 ChatGPT」。

---

## 十一、面试中如何描述本作业（STAR 格式预览）

STAR 是 **Situation（情境）— Task（任务）— Action（行动）— Result（结果）**。

**示例骨架（请替换为你的真实数据与仓库名）**：

- **S**：在 CS336 课程中，需要在不依赖 PyTorch 自带 AdamW 的前提下，从零实现 BPE、因果 Transformer LM、优化器与训练管线，并通过官方单元测试。
- **T**：交付可训练、可复现的最小语言模型，并能在 toy 数据上过拟合验证实现正确性。
- **A**：按模块拆分 tokenizer/model/optim；先单测后集成；对 RoPE 与因果 mask 做形状与对拍检查；使用 warmup+cosine 与学习率分组；用 top-p 做生成调试。
- **R**：`pytest` 全部通过；toy 训练 loss 从约 $\ln V$ 降至明显更低；能清晰向面试官画出数据流与公式。

---

## 十二、完整整合示例（最小可运行伪代码）

下面将 **encode → 模型 → CE → AdamW → 一步更新** 串在一起（变量名示意，**不可直接当作某学期官方 API**）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设已实现：tokenizer, TransformerLM, AdamW, lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_batch = ["hello world", "cs336 assignment"]
input_ids = torch.tensor(
    [tokenizer.encode(t) for t in text_batch],
    dtype=torch.long,
    device=device,
)  # 需 padding 时应用 pad 与 attention_mask，labels 用 ignore_index

model = TransformerLM(vocab_size=V, d_model=D, n_layers=L, n_heads=H).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=...)  # 或自定义

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(input_ids)  # (B, T, V)

shift_logits = logits[:, :-1, :].reshape(-1, V)
shift_labels = input_ids[:, 1:].reshape(-1)
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```

生成侧（概念）：

```python
model.eval()
with torch.no_grad():
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        probs = top_p_filter_softmax(logits, p=0.9, temperature=0.8)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
text = tokenizer.decode(ids[0].tolist())
```

---

## 十三、面试要点速览（答题角度）

1. **数据流**：能口述从字符串到 logits 的每一步形状变化。
2. **因果性**：causal mask 与自回归训练目标一致；推理时无标签，靠采样扩展序列。
3. **BPE 与模型**：tokenizer 只影响离散 ID；嵌入矩阵行数等于 $V$；特殊 token ID 固定且文档化。
4. **优化**：AdamW 解耦衰减；调度为何需要 warmup；何时梯度裁剪。
5. **对比框架**：你实现的是「教学最小闭环」；HuggingFace 提供工程化、算子融合、分布式与生态；各有利弊见下文高频题详解。

---

## 十四、面试高频题（10+ 题详解）

### Q1：请描述你从零实现语言模型的过程

**参考答案**：我按数据流把任务拆成四块：**分词器、模型、损失与优化、训练与生成**。首先实现 **字节级 BPE**：用课程规定的预分词正则把文本切成片段，在片段内统计相邻字节对，迭代 merge 扩展词表，并严格处理 **平局规则**，保证训练与推理同一套 merge 顺序；`encode` 得到 ID 序列，`decode` 查词表拼回字节再 UTF-8 解码。接着实现 **Decoder-only Transformer**：token embedding、多层 block，每层包含 **RMSNorm**、**多头因果自注意力**（对 $Q,K$ 施加 **RoPE**）、残差与 **SwiGLU FFN**；注意力里用 **causal mask** 禁止看未来位置；最后 **lm_head** 映射到词表 logits。训练时对 logits 与 **右移一位** 的 `input_ids` 做 **交叉熵**。优化器使用 **手写的 AdamW**（含偏差修正与解耦权重衰减），训练循环里配合 **学习率调度**（如 warmup+cosine），并记录 loss。验证无误后用 **top-p** 做文本生成调试。整个过程以 **`pytest`** 与 toy 过拟合实验锁定正确性。

### Q2：实现过程中遇到的最大挑战是什么？

**参考答案**（请结合真实经历改写）：我遇到的最大挑战是 **多组件耦合时的错误定位**——例如 BPE 的 merge 顺序与 tie-break 有一处不一致，会导致 encode 结果偏移，进而让模型输入分布与测试期望不符；另一类是 **RoPE 与多头 reshape** 的维度顺序错误，表现为 loss 不降或数值不稳定。我的做法是：**冻结其他模块**，用最小输入单独验证 tokenizer；模型侧用 **固定种子、B=1、小 T** 打印中间张量形状，并对照讲义检查 RoPE 与 mask 的广播维度；必要时与参考实现或 `torch` 内置算子做小规模数值对拍。通过 **分层调试**，最终让单测与过拟合实验都通过。

### Q3：如何验证每个组件的正确性？

**参考答案**：**Tokenizer**：小语料训练、与官方样例 encode/decode 一致；边界字符串与特殊 token。**注意力与 RoPE**：形状检查、因果性测试（未来 token 不应影响当前输出）、与已知实现对比。**损失**：手算微型样例（$B=1,T=2$）核对 CE。**AdamW**：单步更新可解析的简单二次函数或对照 `torch.optim.AdamW`（若允许）。**端到端**：单 batch 过拟合、全量 `pytest`、观察 loss 是否从约 $\ln V$ 下降。

### Q4：你的模型有多少参数？训练了多少数据？

**参考答案**：这是开放性问题，请填真实数字。示例：**参数量**可按 $\approx 2 V D$（嵌入+输出头，若未 tying）+ Transformer 主体（每层 attention/FFN）估算；口头可说明「约 **X M** 参数」。**数据量**说明语料来源（如 toy 复制语料 / 小型维基子集 / 课程提供 shard）、**大致 token 数或文档数**，以及训练 **步数与总 token 数（tokens = batch × T × steps）**。面试官关注的是你是否清楚 **规模量级** 与 **实验可复现**，而非背诵精确个位数。

### Q5：训练过程中的 loss 曲线是怎样的？

**参考答案**：**健康情况**：经过 warmup 后，train loss **整体下行**，可能有噪声；若划分验证集，val loss 先降后可能略升（轻微过拟合）。**异常**：全程平坦在 $\ln V$ 附近 → 可能未学习（LR 太小、标签错、mask 错）；突然变 NaN → LR 过大、数值问题或未裁剪梯度。**我如何描述**：我会准备一张真实截图或口述「前 N 步从 A 降到 B」，并提到曾用 **梯度范数** 与 **学习率** 辅助判断。

### Q6：你如何调优超参数？

**参考答案**：我先保证 **正确性**，再调参。顺序上：**(1)** 固定小模型与小数据，确认能过拟合；**(2)** 调 **学习率** 与 **warmup**（常用网格或二分）；**(3)** 调 **weight decay**、**dropout**（若实现）；**(4)** 再增大 $D,L,T$ 或数据。**batch 与序列长度**受显存约束，必要时 **梯度累积**。记录每次实验的 `lr, wd, batch, T, steps` 与曲线，避免「凭感觉改多处」。

### Q7：BPE 分词器和 Transformer 模型如何连接？

**参考答案**：连接点是 **离散 token ID**。BPE 输出 `List[int]`，每个整数在 $[0, V-1]$（或作业规定范围）；模型中的 **`nn.Embedding(V, D)`** 把这些 ID 映射为向量。模型 **不** 直接处理字符串。特殊 token（如 EOS）在 encode 时插入，词表大小需与嵌入与 **lm_head** 输出维一致。训练数据管道负责 batching 与 padding，并在 labels 里标记忽略位置。

### Q8：你的实现和 HuggingFace Transformers 有什么区别？

**参考答案**：**目标不同**：我的作业实现聚焦 **教学闭环与规范一致性**（手写 AdamW、可测试的最小模块）；HuggingFace 是 **工业级库**，提供海量模型配置、**融合算子**、**分布式**、**checkpoint 生态** 与工具链。**实现层面**：HF 的 LLaMA/Mistral 等实现包含 **KV Cache**、**FlashAttention**、并行与数值细节；我的 A1 版本通常更直白、层数少、以通过测试与可解释为先。**面试价值**：我能讲清 **我实现的子集** 与 **工业版增强点** 的对应关系，而不是声称「等价于 HF」。

### Q9：从零实现 vs 使用框架，各自的优缺点？

**参考答案**：**从零实现优点**：理解每个张量、公式与边界条件；面试能白板推导；调试时有心理模型。**缺点**：耗时长、易出细节 bug、性能未必最优。**框架优点**：快速实验、GPU 优化与生态；**缺点**：若只会调 API，遇到训练异常可能不知根因。最佳实践是：**A1 类作业吃透原理**，工作中用框架并 **能读源码与定位问题**。

### Q10：这个项目中你学到了什么？

**参考答案**：我学到了 **语言建模的完整数据流** 与 **现代 LLM 基础组件**（RoPE、RMSNorm、SwiGLU、因果注意力）；学会了 **用测试驱动开发** 拆分问题；积累了 **形状、设备、数值稳定性** 的调试方法；并对 **优化器与学习率调度** 有了可量化描述的经验。这些对后续 **系统优化（FlashAttention、DDP）** 与 **对齐训练** 都是前置基础。

### Q11：请解释 Top-p 采样与 greedy 的差异

**参考答案**：**Greedy** 每步取 argmax，容易重复、缺乏多样性。**Top-p（nucleus）** 只在累积概率达到 $p$ 的最小集合内采样，兼顾质量与多样性；常配合 **temperature** 缩放 logits。训练仍用真实标签的 CE；采样只影响 **推理**。

### Q12：手写 AdamW 最容易漏掉什么？

**参考答案**：**(1)** **偏差修正**里要用当前 **step**；**(2)** **权重衰减** 是加在参数上的解耦项，不要当成经典 Adam 的 L2 梯度；**(3)** `exp_avg` / `exp_avg_sq` 与参数 **同设备同 dtype 策略**；**(4)** `param_groups` 里 **bias 不衰减** 等分组；**(5)** `zero_grad(set_to_none=True)` 的习惯。

---

## 十五、练习建议（自测清单）

1. **白板**：画出 `(B,T)` 从 embedding 到 logits 的形状变化，并标出 causal mask 作用位置。
2. **手算**：$T=3$、$V=5$ 的假 logits，写出一个 batch 的 CE 计算。
3. **代码**：实现 `top_p_filter` 纯 NumPy/torch 小函数，对随机 logits 跑通。
4. **排查**：故意关掉 causal mask，观察 loss 是否「好得不正常」。
5. **口语**：用 90 秒英文版 walkthrough，录音自我纠正。

---

## 十六、导航与延伸阅读

| 链接 | 内容 |
|------|------|
| [Lesson 07 训练与采样](07-训练循环与损失函数.md) | CE、Top-p、困惑度 |
| [Lesson 09 GPU 与内存](09-GPU架构与内存层级.md) | 进入 Assignment 2 系统篇 |
| [README 参考实现](https://github.com/Melody-Zhou/stanford-cs336-spring2025-assignments) | 社区作业结构参考 |
| [Stanford CS336 官网](https://stanford-cs336.github.io/spring2025/) | 课程主页 |

**下一课**：[Lesson 09：GPU 架构与内存层级](09-GPU架构与内存层级.md) — 为 FlashAttention 与分布式训练打基础。

---

## 附录：提交前自检清单

- [ ] `uv run pytest` 全部通过  
- [ ] 固定种子下关键输出可复现  
- [ ] `encode`/`decode` 与作业样例一致  
- [ ] 因果 mask 在 `(T,T)` 上正确  
- [ ] RoPE 施加维度与讲义一致  
- [ ] AdamW 含偏差修正与解耦权重衰减  
- [ ] 训练循环含 LR 调度（若作业要求）  
- [ ] 生成使用 `eval` + `no_grad`，并实现 top-p（若作业要求）  
- [ ] toy 训练 loss 下降趋势合理  
- [ ] README 含安装、测试、最小训练命令  

---

**结语**：Assignment 1 的目标不是「造 ChatGPT」，而是让你拥有一套 **可向面试官白板展开的实现**。对照官方 PDF 逐项勾选本文与附录清单，你会为后续 FlashAttention、DDP 等系统主题打下扎实接口与调试基础。
