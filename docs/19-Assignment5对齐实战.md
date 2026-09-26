# Lesson 19：Assignment 5 对齐实战（数学推理：SFT + GRPO）

> **Stanford CS336**：Language Modeling from Scratch — 面试导向学习指南（第 19 节）

**先修**：[Lesson 17：SFT 有监督微调](./17-SFT有监督微调.md)、[Lesson 18：RLHF / DPO / GRPO 对齐技术](./18-RLHF-DPO-GRPO对齐技术.md)。

**面试热度**：★★★★☆（对齐 / 应用算法 / 推理增强岗高频；常与「SFT → RL → 评估」链路绑定）

---

## 标题（Title）

**本节主题**：**Assignment 5：Alignment** —— 在**数学推理**任务上，完成 **SFT（监督微调）→ GRPO（组相对策略优化）** 的完整训练闭环，并可选实现 **DPO** 进行**安全偏好对齐**。

**版本边界**：本指南以 [2025 年官方 Assignment 5 主讲义](https://github.com/stanford-cs336/assignment5-alignment/blob/eb2c562e05802d3c9ee4c4ec2eec43b104d77e9e/cs336_spring2025_assignment5_alignment.pdf) 为基准：主线还包含 **zero-shot baseline 与 Expert Iteration**（采样、验证、保留正确答案再 SFT）。该版本 GRPO **不要求 reference KL 项**，并比较是否除组标准差、不同长度归一化及 on/off-policy 更新。以下 KL 与 chat-template 示例属于扩展，不能当作所有年份作业的必做规格；实际评测应使用讲义指定模型、prompt 和奖励函数。

**你在简历/面试里的一句话**：在数学基座上比较 zero-shot、SFT、Expert Iteration 与 GRPO，用可验证奖励和组内优势优化，并按固定协议报告实际验证结果；KL 和偏好 DPO 是按版本选择的扩展，不能把未做的实验写成经历。

**与前后课关系**：第 17 课讲 SFT 通用范式，第 18 课讲 RLHF/DPO/GRPO 理论；本课把二者**落到作业级实现与调试**，下一课（推理优化与部署）延续「训好模型之后如何快、稳地服务」。

---

## 概念讲解（Concepts）

### 1. Assignment 5 总览：SFT + RL 面向数学推理

| 模块 | 你在练什么 | 面试官想听到的关键词 |
|------|-------------|------------------------|
| **Part 1：SFT** | 指令-回答数据、**loss masking**、训练循环、数学基准评测 | instruction tuning、只监督 assistant、GSM8K/MATH |
| **Expert Iteration** | 用规则筛选正确 rollout，再以 SFT 迭代更新 | verified self-training、采样与过滤 |
| **Part 2：GRPO** | 多解采样、规则奖励、组优势、token 级策略梯度/clip；KL 可选 | relative advantage、rule-based reward、old policy log-prob |
| **Optional：DPO** | 偏好对、Bradley-Terry 隐式奖励、β | preference data、helpfulness vs safety |
| **集成评测** | SFT-only vs SFT+GRPO、CoT 质量 | pass@k、maj@k、长度与格式 |

**课程叙事**：预训练模型擅长「续写」，未必擅长「按指令一步步解数学题并给出可检查答案」。Assignment 5 用 **SFT** 建立**格式与行为先验**，再用 **GRPO** 在**同一题目多次采样**中做**相对比较**，配合 **稀疏但可复现** 的**答案正确性奖励**，在工程上比完整 RLHF（RM+PPO）更轻量，又比纯 SFT 更能**针对评测目标**塑形。

**目标**：模型不仅能背题型，更能在**可解析的最终答案**（如 `\boxed{}`）上提升准确率，同时控制**幻觉**、**格式崩坏**与**对奖励规则的投机（reward hacking）**。

---

### 2. Part 1：SFT 实现要点

#### 2.1 加载预训练模型与分词器

- **基座**：课程通常提供较小规模 LM（或指定开源权重），需与 **tokenizer 配套**（词表、特殊 token、chat template）。
- **设备与精度**：训练侧常用 **BF16**；若作业允许 **LoRA**，需在加载后挂载适配器并仅更新可训练参数。
- **一致性**：`model.eval()` / `model.train()` 切换、**梯度检查点**、**FlashAttention** 等优化若开启，SFT 与后续 GRPO **forward 路径**应对齐，避免 log prob 与生成不一致。

#### 2.2 指令数据准备（Instruction Data）

典型对话结构（与 ChatML / Alpaca 等模板兼容）：

- **System（可选）**：你是数学助手；要求逐步推理；最终答案放在 `\boxed{}`。
- **User**：题目（可含 LaTeX）。
- **Assistant**：**Chain-of-Thought（CoT）** + **最终答案**。

数据工程检查表：

| 维度 | 说明 |
|------|------|
| **可解析性** | 金标答案可被规则提取（`\boxed{}`、`####` 行等），与**评测脚本**一致 |
| **难度分布** | 覆盖由易到难；避免全为 OOD 导致梯度噪声过大 |
| **格式一致** | 固定「推理 / 结论」分隔方式，降低 RL 阶段奖励设计复杂度 |
| **去重与泄漏** | 训练/验证/测试严格划分；避免基准题直接进训练集（若作业禁止） |

#### 2.3 Loss Masking：仅在 assistant token 上监督

**定义**：将多轮对话拼成单条 `input_ids` 后，**仅对 assistant 所对应的 token 位置**计算下一词交叉熵；**system / user** 以及 **assistant 之前的所有前缀**在 `labels` 上标为 **忽略**（常见为 `-100`，与 PyTorch `CrossEntropyLoss(ignore_index=-100)` 对齐）。

形式化：设掩码 $m_t \in \{0,1\}$，在 assistant 区间为 1：

$$
\mathcal{L}_{\text{SFT}} = - \frac{1}{\sum_t m_t} \sum_{t} m_t \log p_\theta(x_t \mid x_{<t})
$$

**常见错误**：误监督 user 内容 → 模型被训练成「复述题目」；**错位**：`logits` 与 `labels` 未按「预测下一 token」对齐；**模板差异**：`apply_chat_template` 与手写拼接不一致导致 mask 偏移。

#### 2.4 SFT 训练循环（逻辑）

1. 按 batch 读取对话，构建 `input_ids` / `labels`（含 masking）。
2. **Forward** 得 `logits`，计算 **shifted CE**（与 `ignore_index`）。
3. **Backward**、梯度裁剪、优化器步进；记录 **loss、学习率、吞吐**。
4. 按步或按 epoch 在**固定验证集**上算 loss 或 **小型 held-out EM**（若作业提供脚本）。

超参经验起点（需按算力与模型规模校准）：

| 超参 | 常见范围 | 备注 |
|------|----------|------|
| 学习率 | $10^{-5}$～$5\times10^{-5}$（全参） | 大模型常更小；LoRA 可略大 |
| 有效 batch | 梯度累积拉大 | 影响稳定性与泛化 |
| 序列长度 | 2k～8k | 数学题 + CoT 需要足够上下文 |
| Epoch | 1～3 | 小数据多 epoch 易过拟合格式 |
| Warmup + cosine | 常用 | 与第 7 课训练循环叙事一致 |

#### 2.5 数学基准评测（Evaluation）

| 基准 | 含义 | 常用指标 |
|------|------|----------|
| **GSM8K** | 小学数学应用题 | 最终答案 EM |
| **MATH** | 竞赛级 | 分难度 EM；可报 pass@N |
| **AIME 等** | 更难 | 样本少、方差大 |

**关键**：答案解析与奖励口径应一致；SFT/RL 对照必须使用固定评测模板、温度和 token 上限。训练采样可用不同温度探索，不要求与评测温度相同，但必须记录并处理对应的行为策略概率。

---

### 3. Part 2：GRPO 面向数学推理

#### 3.1 每题多条解（Multiple Samples per Problem）

对同一题目 $q$，从当前策略 $\pi_\theta$ **独立采样** $G$ 条完整解答 $\{y^{(i)}\}_{i=1}^G$（可固定 temperature、top-p）。

- **G 过小**：组内方差估计差，优势噪声大。
- **G 过大**：生成与反向成本线性上升。

实践常从 **G ∈ [4, 16]** 起步，在固定「每步算力预算」下与 **学习率、batch 内题目数** 联调。

#### 3.2 规则奖励：答案正确性验证

主信号常为 **0/1**：

```text
r = 1.0  if normalize(extract_answer(pred)) == normalize(gold)
r = 0.0  otherwise
```

`extract_answer` 与 `normalize`（分数、小数、LaTeX 等价化）应与 **评测脚本共用**，避免训练奖励与测试 EM **定义不一致**。

可选 **塑形**（慎用）：格式分、长度惩罚、重复惩罚。塑形越复杂，**投机空间**越大，越需要监控 **长度分布** 与 **人工抽样**。

#### 3.3 组优势（Group Advantage）

对组内奖励 $\{r_i\}_{i=1}^G$：

**去均值**：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j
$$

**标准化**（更常见）：

$$
A_i = \frac{r_i - \mu}{\sigma + \epsilon},\quad \mu=\frac{1}{G}\sum_j r_j,\ \sigma^2=\frac{1}{G}\sum_j (r_j-\mu)^2
$$

**退化情况**：全组同分时 $\sigma=0$，减均值后优势恰为零，加 $\epsilon$ 可防除零，策略梯度项为零。若另有 KL、熵正则或 weight decay，整体参数更新仍可能非零。跳过整个组会改变这些项的权重，应明确约定，不能把“近零方差”一律视为无效。

**直觉**：在同一难度题目内做 **相对比较**，缓和「难题普遍低分、简单题普遍高分」带来的 **跨题尺度** 问题，与 **稀疏终端奖励** 搭配时尤其重要。

#### 3.4 策略梯度与 KL 约束

以每条回答按有效长度平均的 GRPO-Clip 为例，令 $\rho_{i,t}=\pi_\theta(y_t^{(i)}|q,y_{<t}^{(i)})/\pi_{\mathrm{old}}(y_t^{(i)}|q,y_{<t}^{(i)})$：

$$
J_{\mathrm{clip}}=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\left(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\delta,1+\delta)A_i\right)
$$

这里是 **token 级 ratio**，不是整条序列 log-prob 相减后指数化。clip 约束目标的改进激励，不保证概率比硬性落在区间内。课程还比较 REINFORCE 与不同长度归一化；统一分母会改变训练权重，需与实验定义一致。

**可选扩展**：若使用相对冻结参考模型的 KL 正则，可最小化：

$$
\mathcal{L}_{\text{total}} = -J_{\mathrm{clip}} + \beta_{\mathrm{KL}}\mathbb{E}[\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})]
$$

KL 可缓解偏离参考，但不保证能力、安全或正确性；2025 年主讲义实验省略此项。旧策略用于 importance ratio，reference 用于可选 KL，二者不能混用。若采样使用 temperature/top-p，行为分布不再是原始 softmax；教学时可先用温度 1、top-p 1，并在真实实现中核对概率口径。

#### 3.5 GRPO 训练循环（逻辑）

1. 采样一批题目 $\{q\}$。
2. 对每个 $q$ 生成 $G$ 条 $\{y^{(i)}\}$，计算 **规则奖励** $r_i$。
3. 组内算优势 $A_i$，明确是否标准化；全同分组的策略梯度为零，不默认删除其所有正则项。
4. 对选中 token 计算 **策略损失**（+ **KL 项**）；**backward**。
5. 周期性保存 checkpoint；监控 **reward 均值、KL、生成长度**。

**工程要点**：保存 **rollout 时的旧策略 log prob** 用于 ratio；**参考模型**前向尽量 **no_grad**；分布式时注意 **同一题目 G 条** 的聚合与梯度同步。

---

### 4. 可选部分：DPO 与安全对齐

#### 4.1 安全数据集准备

构造偏好三元组 $(q, y_w, y_l)$：在同一 **用户请求** 下，$y_w$ **更安全/合规**，$y_l$ **更危险或更迎合恶意目标**。可与数学数据**分阶段**或**混合**训练，注意 **遗忘** 与 **拒答过度**。

#### 4.2 DPO 损失（实现视角）

在隐式奖励参数化下（参见第 18 课），最大化偏好对数似然，典型形式为：

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(q,y_w,y_l)}\Big[\log \sigma\Big(\beta \big(\Delta_w - \Delta_l\big)\Big)\Big]
$$

其中 $\Delta$ 是整条 completion 的序列 log-ratio，序列 log-prob 为有效 token 的和。DPO 的 $\beta$ 对应推导中的 KL 系数，固定奖励下较大值使最优策略更受参考约束；有限步偏好训练还受梯度尺度/学习率影响，不能简单推断越大越偏离。

实现检查：**仅对 completion 部分**累加 log prob；**padding** 与 **mask** 一致；**参考模型**冻结。

---

### 5. 预期结果与评估（Expected Results）

下列为**定性预期**（具体数值以作业说明与随机种子为准）：

| 阶段 | 常见现象 |
|------|----------|
| **SFT** | 指令跟随明显改善，输出格式稳定，基线 **EM** 相对预训练有提升 |
| **SFT + GRPO** | 在奖励与评测一致时，**测试 EM** 或 **pass@k** 常进一步提升；若奖励有漏洞，可能出现 **训练集 reward 涨、测试 EM 不涨** |
| **负面信号** | 平均生成长度异常上升/下降、重复模式、KL 持续飙升 |

报告结果时建议同时给出：**EM**、**pass@k**、**maj@k**（若作业要求）、**平均长度**、**KL 曲线**、**GPU 时间**，并固定 **评测协议** 做 **SFT vs SFT+GRPO** 对照表。

---

### 6. RL 训练调试专题：奖励坍塌与 KL 监控

#### 6.1 奖励坍塌（Reward Collapse）

**表现**：曲线显示 **mean reward 趋近常数**（例如长期在 0 附近），或 **方差趋零**。

**可能原因与排查**：

| 原因 | 排查 |
|------|------|
| 解析器与金标不一致 | 单元测试 `extract_answer` / `normalize` |
| 采样温度过低，多样性不足 | 提高 temperature 或 top-p |
| 题目过难，几乎全错 | 检查数据难度与 SFT 质量 |
| 全组同分 | 优势应恰为零；检查是否错误除零，以及 KL/weight decay 是否仍更新 |

#### 6.2 KL 散度监控

**若启用 KL 扩展**：记录相对冻结 reference 的 KL，注明估计器、token/序列平均及行为分布；未启用时不要求驻留 reference。下表仅适用于启用该正则的实验。

| 现象 | 可能解读 | 调参方向 |
|------|----------|----------|
| KL **持续上升** | 策略偏离参考过快 | 增大 **β**、减小 **RL 学习率**、加强 **clip** |
| KL **接近 0** | 更新过弱或实现 bug（如 ref 未冻结却与 policy 混用） | 检查 **ref forward**、学习率 |
| **reward 升而 KL 爆** | 典型「能力-稳定性」权衡 | 优先 **稳住 KL**，再微调 **β** |

建议与 **验证集 perplexity 相对 SFT**（若可算）或 **小型通用任务** 一并观察，避免 **数学分涨、通用能力掉** 未被察觉。

#### 6.3 其他高频问题

- **指标与训练不一致**：统一解析器与模板。
- **长度爆炸**：长度惩罚、`max_new_tokens`、检查是否在奖励中无意鼓励长输出。
- **分布式下优势算错**：确认 **同一 prompt 的 G 条** 在同一进程组内聚合后再反传。

---

### 7. 面试中如何呈现 Assignment 5

用 **问题 → 方法 → 指标 → 复盘** 控制在 **90～120 秒**：

1. **问题**：基座在数学指令与可解析答案上不足。
2. **方法**：SFT 与 Expert Iteration 做冷启动/自训练；GRPO 用组采样、规则奖励与组优势；明确是否启用 KL，偏好 DPO 可选。
3. **指标**：GSM8K/MATH EM、pass@k、SFT vs SFT+GRPO。
4. **复盘**：一次真实的 reward 或可选 KL 异常与定位方法（解析器、采样概率口径、mask、超参）。

---

## 代码走读（Code）

以下为实现**思路**的伪代码骨架，**函数名与文件路径以课程官方仓库为准**。

### 1. SFT：加载、批处理、Masking、训练步

```python
import torch
import torch.nn.functional as F

def build_sft_batch(tokenizer, conversations, max_length: int):
    """
    conversations: List[ List[{role, content}] ]，多轮对话。
    返回 padding 后的 input_ids 与 labels；labels 在非 assistant 段为 -100。
    """
    # 教学模板必须支持 {% generation %}；不能逐消息模板化后再拼接。
    # 实际课程若用 r1_zero prompt，应按其 prompt/response 边界构造 mask。
    features = []
    for conv in conversations:
        enc = tokenizer.apply_chat_template(
            conv, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True, max_length=max_length, add_generation_prompt=False,
        )
        ids = enc["input_ids"]
        mask = enc.get("assistant_masks")
        if mask is None or len(mask) != len(ids):
            raise ValueError("unsupported assistant mask")
        labels = [tid if m else -100 for tid, m in zip(ids, mask)]
        if not any(tid != -100 for tid in labels[1:]):
            raise ValueError("empty supervision after truncation/shift")
        features.append({"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels})
    # tokenizer.pad 本身不负责按 -100 补齐 labels，交给显式标签 collator。
    from transformers import DataCollatorForSeq2Seq
    return DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)(features)


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V)；labels: (B, T)，-100 忽略。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if not (shift_labels != -100).any():
        raise ValueError("no supervised tokens")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# 训练循环骨架（单卡示意）
def sft_train_step(model, batch, optimizer, max_grad_norm: float = 1.0):
    model.train()
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    # 若模型未内置 loss，则用手写 sft_loss(out.logits, batch["labels"])
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()
```

**检查点**：`labels` 与「预测下一 token」**错位**；`ignore_index` 与模板 **逐 token 对齐**；多轮时 **每一段 assistant** 是否都应被监督（作业约定为准）。

---

### 2. GRPO：多解、奖励、优势、裁剪与 KL（示意）

```python
import statistics
import torch.nn as nn

def extract_answer(text: str) -> str | None:
    """与评测脚本共享：从模型输出中解析最终答案。"""
    ...

def normalize(ans: str) -> str:
    """数值/符号/LaTeX 等价归一化。"""
    ...

def rule_reward(pred: str, gold: str) -> float:
    p = extract_answer(pred)
    if p is None:
        return 0.0
    return 1.0 if normalize(p) == normalize(gold) else 0.0


def group_advantages(rewards: list[float], eps: float = 1e-5) -> tuple[list[float], bool]:
    """
    返回 (advantages, skip)。
    skip=True 仅表示全同分、策略梯度为零；不表示应跳过 KL/其他正则。
    """
    if len(rewards) < 2 or eps <= 0:
        raise ValueError("need at least two rewards and positive eps")
    mu = statistics.mean(rewards)
    sigma = statistics.pstdev(rewards)
    if sigma == 0:
        return [0.0] * len(rewards), True
    adv = [(r - mu) / (sigma + eps) for r in rewards]
    return adv, False


def token_log_probs(model, input_ids, attention_mask, labels_for_completion):
    """
    返回 token log-prob 与 response mask，均为 (B,T-1)，供 GRPO token ratio。
    DPO 的序列 log-prob 另由 (logp * mask).sum(-1) 得到，不取长度平均。
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    labels = labels_for_completion[:, 1:]
    mask = labels != -100
    safe_labels = labels.masked_fill(~mask, 0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return logp, mask


def grpo_loss_term(logp_new, logp_old, advantage, response_mask, clip_eps: float = 0.2):
    """logp: (B,T)，advantage: (B,)，response_mask 为 0/1，按有效长度平均。"""
    mask = response_mask.bool()
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("empty response")
    # 必须在 exp 之前屏蔽 padding；最后才乘 mask 无法消除 inf/NaN。
    log_ratio = (logp_new - logp_old.detach()).masked_fill(~mask, 0.0)
    ratio = torch.exp(log_ratio)
    adv = advantage.detach().unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    token_loss = -torch.minimum(unclipped, clipped)
    return (token_loss.masked_fill(~mask, 0.0).sum(-1) / counts).mean()


# 单题多采样逻辑位置：generate G 次 -> 算 reward -> advantages -> 反传
```

**检查点**：**旧策略** log prob 用于 ratio；**参考模型**仅用于 KL；**generate 与 log prob 路径**使用同一套 attention mask 与 special tokens。

**数值边界**：上例在指数运算前屏蔽无效 token，避免 padding 的异常 log-ratio 污染损失与梯度；有效 token 的极端 log-ratio 仍可能溢出，PPO-style clip 并不保证所有 ratio 都有界。真实训练需监控非有限值、精度与更新幅度，不能把屏蔽 padding 当作所有 NaN 的修复。

---

### 3. 可选：DPO batch 与损失

```python
import torch.nn.functional as F

def dpo_loss(
    policy_logp_chosen,
    policy_logp_rejected,
    ref_logp_chosen,
    ref_logp_rejected,
    beta: float = 0.1,
):
    logits = beta * (
        (policy_logp_chosen - ref_logp_chosen)
        - (policy_logp_rejected - ref_logp_rejected)
    )
    return -F.logsigmoid(logits).mean()
```

**检查点**：`logp_*` 为**整条 completion** 的聚合；batch 维平均；**参考模型**无梯度。

---

### 端到端串联（End-to-End Walkthrough）

1. **环境**：安装依赖、对齐 **CUDA / PyTorch**、能跑通 **预训练权重加载** 与 **单次 forward**。
2. **数据**：准备数学指令 JSON/JSONL；划分 train/val；**打印一条** 经 `apply_chat_template` 后的 token 与 mask，确认 **assistant 段** 正确。
3. **SFT**：实现 **masking + CE**；跑若干 step 后 **loss 下降**；在 val 上跑 **官方评测脚本** 得 **SFT 基线 EM**。
4. **GRPO**：按讲义从基座/SFT/EI checkpoint 初始化 policy，缓存旧策略 log-prob；实现 G 次 generate → reward → advantage → loss，记录 reward、长度、熵与梯度；仅在 KL 扩展中额外使用 reference。
5. **对照**：同一评测协议下 **SFT vs SFT+GRPO**；保存 **最佳 checkpoint** 与 **超参表**。
6. **可选 DPO**：构造安全偏好对；在 ref 上跑 **DPO**；小样本测 **拒答** 与 **数学 EM** 是否掉点。

---

## 面试要点（Interview Points）

| 主题 | 一句话 |
|------|--------|
| **为何先 SFT** | 冷启动策略、稳定格式、缩小 RL 探索空间，并提供 $\pi_{\text{ref}}$。 |
| **GRPO 与 PPO** | GRPO 用**组内基线**处理稀疏奖励；未必省掉 clip，但弱化 **价值网络** 依赖叙事。 |
| **KL** | 可选 reference 正则；2025 主线省略，不保证消除遗忘或投机。 |
| **DPO** | 离线偏好优化，无显式 RM rollout；数据需覆盖目标行为。 |
| **评测** | 解析/奖励口径一致；对照实验固定评测模板与采样协议，训练探索温度可以不同。 |

### STAR 话术模板

- **S（情境）**：课程要求完成数学推理上的对齐 pipeline：SFT + GRPO，可选 DPO。
- **T（任务）**：提升 **EM / pass@k**，并保证可复现评测。
- **A（行动）**：构造指令-CoT 数据与 response mask；比较 SFT/EI/GRPO，明确组优势、长度归一化和可选 KL 的配置；只陈述实际做过的扩展。
- **R（结果）**：对照表汇报 **SFT vs SFT+GRPO**；复盘一次 **reward/KL** 问题与修复。

### 面试高频题（10+ 详解）

**Q1：你是如何实现数学推理的 RL 训练的？**  
**答**：比较 zero-shot、SFT 与 EI；GRPO 每题采 G 条，用规则奖励、组内去均值/可选标准差与 token 级策略梯度或 clip。缓存旧策略 log-prob，统一解析/评测口径。2025 主线不加 reference KL；若做扩展，另报告配置与消融。

**Q2：组优势怎么算？全错一组怎么办？**  
**答**：原版标准化为 $A_i=(r_i-\mu)/(\sigma+\epsilon)$，课程也比较仅去均值。全同分时 $A_i=0$，策略梯度为零，但 KL/weight decay 等仍可产生更新；是否跳过整个组要明确目标，不能一概而论。

**Q3：奖励如何设计？**  
**答**：主信号 **正确性**；辅助项谨慎；与 **EM** 一致；监控 **长度与投机**。

**Q4：SFT 与 SFT+GRPO 预期差异？**  
**答**：SFT 稳格式；GRPO 在奖励对齐评测时常 **涨 EM / pass@k**；需防 **分布过拟合** 与 **hack**。

**Q5：训练中有哪些挑战？**  
**答**：**KL 暴涨**、**reward 不变**、解析错误、长度过长；对应调 **β、LR、温度、max tokens** 与 **代码审查**。

**Q6：如何评估推理提升？**  
**答**：**EM、pass@k、maj@k**；分层难度；抽样检查 **CoT 是否跳步**；报告 **长度与 KL**。

**Q7：DPO 安全对齐实现注意什么？**  
**答**：**completion mask**、**β**、**ref 冻结**、与数学能力 **混合比例** 防遗忘。

**Q8：RL 超参经验？**  
**答**：RL LR 常 **低于 SFT**；扫 **β、G、clip**；固定 **验证协议** 选 checkpoint。

**Q9：CoT 怎么训？**  
**答**：SFT **显式监督** 中间步骤；GRPO 多为 **终端奖励**，CoT 间接被塑造；需 **模板一致**。

**Q10：项目体现什么能力？**  
**答**：对齐 **全流程**、**奖励与评测一致**、**方差控制（组优势）**、**KL 稳定训练**。

**Q11：G 与吞吐？**  
**答**：每题 **G 次生成** 成本高；调参时可在相近 **采样预算** 下比较 **验证 EM**。

**Q12：GRPO 一定需要 $\pi_{\text{ref}}$ 的 KL 吗？**  
**答**：不一定。KL 可限制相对参考的漂移，但也增加模型与前向开销，不保证安全/能力保留。2025 作业主线明确省略；是否启用应按任务验证和消融判断。

---

## 练习题（Practice）

1. **Masking**：若把 system token 也计入 loss，优化目标偏到哪里？如何用最小单元测试发现？
2. **优势**：某题 G 条全错，标准化后梯度是否应近似为 0？你的实现是否 **skip**？
3. **KL**：β 过大与过小各表现为何？你最先改 **β** 还是 **RL LR**？
4. **DPO**：β 增大时，策略更贴近还是更远离 $\pi_{\text{ref}}$？对安全数据意味着什么？
5. **评测**：pass@32 升而 maj@1 降，可能说明什么（多样性 vs 一致性）？
6. **解析**：若 `normalize` 把 `1/2` 与 `0.5` 判不等，训练与测试会怎样分叉？
7. **分布式**：同一题 G 条 rollouts 若跨卡拆分，优势应在何处聚合？

---

## 导航（Navigation）

| 上一节 | 下一节 |
|--------|--------|
| [← Lesson 18：RLHF / DPO / GRPO 对齐技术](./18-RLHF-DPO-GRPO对齐技术.md) | [Lesson 20：推理优化与模型部署 →](./20-推理优化与模型部署.md) |

---

**延伸阅读**：[DeepSeekMath（GRPO）](https://arxiv.org/html/2402.03300v3)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/html/2305.18290v2)。作业细节以官方对应年份讲义与代码为准。

**文档版本**：Lesson 19 — Assignment 5 对齐实战（面试导向）。
