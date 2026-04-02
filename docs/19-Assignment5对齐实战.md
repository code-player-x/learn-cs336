# Lesson 19：Assignment 5 对齐实战（数学推理：SFT + GRPO）

> **Stanford CS336**：Language Modeling from Scratch — 面试导向学习指南（第 19 节）

**先修**：[Lesson 17：SFT 有监督微调](./17-SFT有监督微调.md)、[Lesson 18：RLHF / DPO / GRPO 对齐技术](./18-RLHF-DPO-GRPO对齐技术.md)。

**面试热度**：★★★★☆（对齐 / 应用算法 / 推理增强岗高频；常与「SFT → RL → 评估」链路绑定）

---

## 标题（Title）

**本节主题**：**Assignment 5：Alignment** —— 在**数学推理**任务上，完成 **SFT（监督微调）→ GRPO（组相对策略优化）** 的完整训练闭环，并可选实现 **DPO** 进行**安全偏好对齐**。

**你在简历/面试里的一句话**：在指令跟随基座上，用**可验证规则奖励**与**组内相对优势**做 RL，提升 GSM8K/MATH 等指标，并用 **KL 到 SFT 参考模型** 抑制策略漂移；可选用 **DPO** 在偏好数据上强化拒答有害请求的能力。

**与前后课关系**：第 17 课讲 SFT 通用范式，第 18 课讲 RLHF/DPO/GRPO 理论；本课把二者**落到作业级实现与调试**，下一课（推理优化与部署）延续「训好模型之后如何快、稳地服务」。

---

## 概念讲解（Concepts）

### 1. Assignment 5 总览：SFT + RL 面向数学推理

| 模块 | 你在练什么 | 面试官想听到的关键词 |
|------|-------------|------------------------|
| **Part 1：SFT** | 指令-回答数据、**loss masking**、训练循环、数学基准评测 | instruction tuning、只监督 assistant、GSM8K/MATH |
| **Part 2：GRPO** | **多解采样**、**规则奖励**、**组优势**、**KL 约束**、策略梯度 | relative advantage、rule-based reward、reference model |
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

形式化：设掩码 \(m_t \in \{0,1\}\)，在 assistant 区间为 1：

\[
\mathcal{L}_{\text{SFT}} = - \frac{1}{\sum_t m_t} \sum_{t} m_t \log p_\theta(x_t \mid x_{<t})
\]

**常见错误**：误监督 user 内容 → 模型被训练成「复述题目」；**错位**：`logits` 与 `labels` 未按「预测下一 token」对齐；**模板差异**：`apply_chat_template` 与手写拼接不一致导致 mask 偏移。

#### 2.4 SFT 训练循环（逻辑）

1. 按 batch 读取对话，构建 `input_ids` / `labels`（含 masking）。
2. **Forward** 得 `logits`，计算 **shifted CE**（与 `ignore_index`）。
3. **Backward**、梯度裁剪、优化器步进；记录 **loss、学习率、吞吐**。
4. 按步或按 epoch 在**固定验证集**上算 loss 或 **小型 held-out EM**（若作业提供脚本）。

超参经验起点（需按算力与模型规模校准）：

| 超参 | 常见范围 | 备注 |
|------|----------|------|
| 学习率 | \(10^{-5}\)～\(5\times10^{-5}\)（全参） | 大模型常更小；LoRA 可略大 |
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

**关键**：评测时的 **prompt 模板**、**temperature**、**max tokens**、**答案解析函数** 与训练/奖励侧 **必须同源**，否则出现「训练涨分、评测无效」的假结论。

---

### 3. Part 2：GRPO 面向数学推理

#### 3.1 每题多条解（Multiple Samples per Problem）

对同一题目 \(q\)，从当前策略 \(\pi_\theta\) **独立采样** \(G\) 条完整解答 \(\{y^{(i)}\}_{i=1}^G\)（可固定 temperature、top-p）。

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

对组内奖励 \(\{r_i\}_{i=1}^G\)：

**去均值**：

\[
A_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j
\]

**标准化**（更常见）：

\[
A_i = \frac{r_i - \mu}{\sigma + \epsilon},\quad \mu=\frac{1}{G}\sum_j r_j,\ \sigma^2=\frac{1}{G}\sum_j (r_j-\mu)^2
\]

**退化情况**：全组 **同分**（全对或全错）时 \(\sigma \approx 0\)。实务应 **跳过该题的策略梯度** 或 **不反传**，避免除零或零梯度噪声步被误放大。

**直觉**：在同一难度题目内做 **相对比较**，缓和「难题普遍低分、简单题普遍高分」带来的 **跨题尺度** 问题，与 **稀疏终端奖励** 搭配时尤其重要。

#### 3.4 策略梯度与 KL 约束

对每条采样序列，最大化加权对数似然（可 token 级聚合）：

\[
J \approx \mathbb{E}\Big[\sum_{t} A_i \cdot f\big(\log \pi_\theta(y_t^{(i)}\mid q, y_{<t}^{(i)})\big)\Big]
\]

实务中常配合 **PPO 式 clip**：用 **旧策略** \(\pi_{\theta_{\text{old}}}\) 的采样计算 **importance ratio** \(r_t=\pi_\theta/\pi_{\theta_{\text{old}}}\)，并对目标做 **clip**，限制单次更新幅度。

**KL 到参考模型** \(\pi_{\text{ref}}\)（通常为 **冻结的 SFT 模型**）：

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PG}} + \beta \cdot \mathbb{E}[\text{KL}(\pi_\theta \| \pi_{\text{ref}})]
\]

作用：**锚定**语言能力与格式先验，减轻 **为刷正确率而胡编**、**模式坍缩**（例如极短输出、重复 `\boxed{1}`）。

#### 3.5 GRPO 训练循环（逻辑）

1. 采样一批题目 \(\{q\}\)。
2. 对每个 \(q\) 生成 \(G\) 条 \(\{y^{(i)}\}\)，计算 **规则奖励** \(r_i\)。
3. 组内算 **优势** \(A_i\)；过滤 \(\sigma\approx 0\) 的组（按实现约定）。
4. 对选中 token 计算 **策略损失**（+ **KL 项**）；**backward**。
5. 周期性保存 checkpoint；监控 **reward 均值、KL、生成长度**。

**工程要点**：保存 **rollout 时的旧策略 log prob** 用于 ratio；**参考模型**前向尽量 **no_grad**；分布式时注意 **同一题目 G 条** 的聚合与梯度同步。

---

### 4. 可选部分：DPO 与安全对齐

#### 4.1 安全数据集准备

构造偏好三元组 \((q, y_w, y_l)\)：在同一 **用户请求** 下，\(y_w\) **更安全/合规**，\(y_l\) **更危险或更迎合恶意目标**。可与数学数据**分阶段**或**混合**训练，注意 **遗忘** 与 **拒答过度**。

#### 4.2 DPO 损失（实现视角）

在隐式奖励参数化下（参见第 18 课），最大化偏好对数似然，典型形式为：

\[
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(q,y_w,y_l)}\Big[\log \sigma\Big(\beta \big(\Delta_w - \Delta_l\big)\Big)\Big]
\]

其中 \(\Delta\) 为 \(\log \frac{\pi_\theta(y|q)}{\pi_{\text{ref}}(y|q)}\) 在整条 completion 上的聚合（常取 **序列 log prob 之和**）。**β** 控制与 \(\pi_{\text{ref}}\) 的偏离强度：β 大 → 更强调偏好对比，但需防 **训练不稳定**。

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
| 优势全为零仍反传 | 确认实现中 **跳过 \(\sigma\approx 0\)** |

#### 6.2 KL 散度监控

**应记录的标量**：batch 内 **近似 KL(\(\pi_\theta\|\pi_{\text{ref}}\))**（token 平均或序列平均，与作业定义一致）。

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
2. **方法**：SFT 做格式与冷启动；GRPO 用 **组采样 + 规则奖励 + 组内优势 + KL**；可选 DPO 做安全偏好。
3. **指标**：GSM8K/MATH EM、pass@k、SFT vs SFT+GRPO。
4. **复盘**：一次真实的 **KL 或 reward** 异常与如何定位（解析器 / 超参 / 跳过零方差组）。

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
    batch_input_ids, batch_labels = [], []

    for conv in conversations:
        ids, lab = [], []
        for turn in conv:
            # 具体 API 以 tokenizer.apply_chat_template 为准；此处为概念示意
            segment = tokenizer.apply_chat_template(
                [turn], tokenize=True, add_generation_prompt=(turn["role"] == "user")
            )
            seg_ids = segment["input_ids"]
            if turn["role"] == "assistant":
                ids.extend(seg_ids)
                lab.extend(seg_ids)
            else:
                ids.extend(seg_ids)
                lab.extend([-100] * len(seg_ids))

        ids, lab = ids[:max_length], lab[:max_length]
        batch_input_ids.append(ids)
        batch_labels.append(lab)

    batch = tokenizer.pad(
        {"input_ids": batch_input_ids, "labels": batch_labels},
        padding=True,
        return_tensors="pt",
    )
    return batch


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, T, V)；labels: (B, T)，-100 忽略。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
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
    若组内方差过小（全同分），返回 skip=True，建议本组不反传。
    """
    mu = statistics.mean(rewards)
    sigma = statistics.pstdev(rewards)
    if sigma < eps:
        return [0.0] * len(rewards), True
    adv = [(r - mu) / (sigma + eps) for r in rewards]
    return adv, False


def seq_log_probs(model, input_ids, attention_mask, labels_for_completion) -> torch.Tensor:
    """
    对 completion token 求和或求平均 log pi（与作业定义一致）。
    labels_for_completion: 仅 completion 位置非 -100，用于 mask。
    """
    ...


def grpo_loss_term(logp_new, logp_old, advantage, clip_eps: float = 0.2):
    """PPO 风格标量示意：在序列级聚合 ratio 时需与课程公式一致。"""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.min(unclipped, clipped)


# 单题多采样逻辑位置：generate G 次 -> 算 reward -> advantages -> 反传
```

**检查点**：**旧策略** log prob 用于 ratio；**参考模型**仅用于 KL；**generate 与 log prob 路径**使用同一套 attention mask 与 special tokens。

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
4. **GRPO**：从 **SFT checkpoint** 初始化 policy 与 ref；实现 **G 次 generate → reward → advantage → loss**；记录 **reward 均值、KL、长度**。
5. **对照**：同一评测协议下 **SFT vs SFT+GRPO**；保存 **最佳 checkpoint** 与 **超参表**。
6. **可选 DPO**：构造安全偏好对；在 ref 上跑 **DPO**；小样本测 **拒答** 与 **数学 EM** 是否掉点。

---

## 面试要点（Interview Points）

| 主题 | 一句话 |
|------|--------|
| **为何先 SFT** | 冷启动策略、稳定格式、缩小 RL 探索空间，并提供 \(\pi_{\text{ref}}\)。 |
| **GRPO 与 PPO** | GRPO 用**组内基线**处理稀疏奖励；未必省掉 clip，但弱化 **价值网络** 依赖叙事。 |
| **KL** | 锚定 SFT，减轻遗忘与胡编。 |
| **DPO** | 离线偏好优化，无显式 RM rollout；数据需覆盖目标行为。 |
| **评测** | 解析器、模板、采样与训练侧必须一致。 |

### STAR 话术模板

- **S（情境）**：课程要求完成数学推理上的对齐 pipeline：SFT + GRPO，可选 DPO。
- **T（任务）**：提升 **EM / pass@k**，并保证可复现评测。
- **A（行动）**：构造指令-CoT 数据；实现 **assistant-only loss**；GRPO **组采样、规则奖励、组优势、KL**；可选 **偏好对与 DPO**。
- **R（结果）**：对照表汇报 **SFT vs SFT+GRPO**；复盘一次 **reward/KL** 问题与修复。

### 面试高频题（10+ 详解）

**Q1：你是如何实现数学推理的 RL 训练的？**  
**答**：先 **SFT** 学会指令格式与 CoT；奖励用 **规则验证**（解析后与金标比对）得稀疏 0/1；优化用 **GRPO**：每题采样 **G 条**，**组内标准化优势**，**策略梯度 +（常）PPO clip**，并对 **\(\pi_{\text{ref}}\)（SFT）** 加 **KL**。解析与评测 **同源**，并缓存 **旧策略 log prob**。

**Q2：组优势怎么算？全错一组怎么办？**  
**答**：\(A_i=(r_i-\mu)/(\sigma+\epsilon)\)。若 **全同分**，\(\sigma\approx 0\)，应 **跳过梯度** 或 **不更新**，避免无效步。

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

**Q12：为何需要 \(\pi_{\text{ref}}\) 的 KL？**  
**答**：RL **最大化奖励** 易 **偏离数据分布**；KL 正则保留 **LM 先验**，抑制 **遗忘与投机**。

---

## 练习题（Practice）

1. **Masking**：若把 system token 也计入 loss，优化目标偏到哪里？如何用最小单元测试发现？
2. **优势**：某题 G 条全错，标准化后梯度是否应近似为 0？你的实现是否 **skip**？
3. **KL**：β 过大与过小各表现为何？你最先改 **β** 还是 **RL LR**？
4. **DPO**：β 增大时，策略更贴近还是更远离 \(\pi_{\text{ref}}\)？对安全数据意味着什么？
5. **评测**：pass@32 升而 maj@1 降，可能说明什么（多样性 vs 一致性）？
6. **解析**：若 `normalize` 把 `1/2` 与 `0.5` 判不等，训练与测试会怎样分叉？
7. **分布式**：同一题 G 条 rollouts 若跨卡拆分，优势应在何处聚合？

---

## 导航（Navigation）

| 上一节 | 下一节 |
|--------|--------|
| [← Lesson 18：RLHF / DPO / GRPO 对齐技术](./18-RLHF-DPO-GRPO对齐技术.md) | [Lesson 20：推理优化与模型部署 →](./20-推理优化与模型部署.md) |

---

**延伸阅读**：DeepMind GRPO；OpenAI InstructGPT；Rafailov et al. DPO。作业细节以 **CS336 官方 Assignment 5 说明与代码框架** 为准。

**文档版本**：Lesson 19 — Assignment 5 对齐实战（面试导向）。
