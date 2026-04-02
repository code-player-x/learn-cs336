# Lesson 18：RLHF · DPO · GRPO 对齐技术

> **CS336 面试导向学习指南** — 从人类反馈强化学习（RLHF）到直接偏好优化（DPO）与组相对策略优化（GRPO）：把「会说话的基座」变成 **有用、无害、诚实（HHH）** 的助手。

---

## 一、概念（Concepts）

### 1.1 为什么要对齐：HHH 与「预训练目标 ≠ 人类目标」

仅靠大规模预训练（下一词预测，NTP）得到的模型，优化的是 **语料分布上的似然**，并不直接优化产品与社会所期望的行为：

| 维度 | 英文 | 含义（面试可展开） |
|------|------|---------------------|
| **有用** | Helpful | 遵循指令、完成任务、信息密度高、减少无效啰嗦 |
| **无害** | Harmless | 拒绝恶意请求、降低有害输出、提高对越狱提示的鲁棒性 |
| **诚实** | Honest | 在不确定时表达不确定、减少编造（幻觉）、引用与事实一致 |

这三项合称 **HHH**。**对齐（alignment）** 的目标，是把模型行为从「像互联网语料」拉向 **更符合人类价值观与使用规范**。常见技术路径包括：

- **SFT**：用示范数据教会指令遵循与对话格式（见 [Lesson 17](./17-SFT有监督微调.md)）。
- **偏好学习**：RLHF、DPO、迭代偏好优化等，用排序或成对比较细调行为。
- **规则 / 宪法**：Constitutional AI 等，用原则约束自评与改写。
- **可验证奖励 RL（RLVR）**：数学、代码等任务上用 **执行结果** 作奖励，常与 GRPO 类组采样结合。

**一句话**：预训练学「统计上的像」；对齐学「人类觉得该像什么样」。

---

### 1.2 RLHF 三阶段 Pipeline（标准叙事）

**RLHF（Reinforcement Learning from Human Feedback）** 在工业界与论文（如 InstructGPT）中常被描述为三步（实现细节因团队而异，面试按此框架答即可）：

| 步骤 | 名称 | 作用 |
|------|------|------|
| **1** | **SFT 模型作起点** | 用高质量指令–回答数据微调基座，得到「会听话、会对话格式」的初始策略；该 checkpoint 常同时作为后续 RL 的 **初始策略** 与 **参考模型 \(\pi_{\mathrm{ref}}\)** 的来源（参考模型多 **冻结** 或极慢更新） |
| **2** | **奖励模型（RM）训练** | 收集人类偏好数据 \((x, y_w, y_l)\)，用 **Bradley–Terry（BT）** 等配对模型学习标量 \(r_\phi(x,y)\)，近似人类排序 |
| **3** | **PPO 等策略优化** | 以 RM 为奖励信号优化 \(\pi_\theta\)，并加 **KL 到 \(\pi_{\mathrm{ref}}\)**，在「刷分」与「别偏离 SFT 太远」之间折中 |

直觉：**SFT** 教格式与基本服从；**RM** 定义「什么叫更好」；**RL** 把「更好」变成可优化目标。

---

### 1.3 步骤 1：SFT 模型作为起点

给定上下文 \(x\)（单轮指令或多轮对话），策略 \(\pi_\theta(y\mid x)\) 在 SFT 阶段通过 **负对数似然**（常对 assistant 段 mask 后计算）模仿示范。完成后得到 **SFT 模型**：

- 作为 **PPO 的初始策略**，避免从随机策略冷启动；
- 初始化或拷贝出 **\(\pi_{\mathrm{ref}}\)**，用于后续 KL 惩罚，锚定「可接受行为」邻域。

**面试要点**：SFT 无法区分「两个都不错但人类更喜欢 A」这类细粒度偏好，因此需要偏好数据 + RM 或 DPO 类直接偏好目标。

---

### 1.4 步骤 2：奖励模型与人类偏好、Bradley–Terry

#### 偏好数据形态

典型为 **\((x, y_w, y_l)\)**：同一 prompt \(x\) 下，**chosen** \(y_w\) 与 **rejected** \(y_l\)。来源可包括：人类并排标注、排序多条候选、或 **AI 反馈（RLAIF）** 生成的合成偏好对。

#### Bradley–Terry 模型

将「\(y_w\) 优于 \(y_l\)」的概率写成与 **隐式效用差** 相关的 logistic 形式。若用可学习标量奖励 \(r_\phi(x,y)\) 近似人类效用，常见写法为：

\[
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)
\]

训练时最大化该模型下的对数似然，等价于让 **chosen 的奖励高于 rejected**。RM 常为与策略同族的 **Transformer**：输入 \((x,y)\) 拼接，取末 token 隐状态经线性层输出 **标量奖励**。

**工程注意**：RM 易出现 **长度偏置**（更长回答分更高）；需长度归一、截断或数据构造控制；奖励 **数值尺度** 需与后续 PPO 的超参（如 advantage 归一化）匹配。

---

### 1.5 步骤 3：PPO 与 RLHF 目标

#### PPO 在 LM 中的角色

将文本生成视为序列决策：每步选 token；**RM** 常在 **完整回答** 后给出终端奖励（可叠加逐步 KL 惩罚）。**价值网络 \(V_\psi\)** 估计从某前缀出发的期望回报，用于 **GAE** 等 **优势函数** \(A_t\)，降低策略梯度方差。

#### 裁剪目标（Clipped Surrogate）

用重要性比 \(r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}\) 利用旧策略样本更新，并对目标 **clip**，限制单次更新幅度：

\[
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]
\]

直觉：**别把策略一步改太狠**，否则分布剧变、训练易崩。

#### KL 惩罚：贴近 SFT 参考模型

目标中常加入 **\(\beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\mathrm{ref}})\)**（或等价约束），使优化后的策略 **不要偏离 SFT 参考太远**：

- RM 只是人类偏好的 **近似**，在未见区域可能被 **过度优化**；
- 无 KL 时，策略可能找到 **RM 盲点**（reward hacking），对人类很糟但对 RM 分高。

KL 起到 **信任域**：在参考模型附近的「安全邻域」内提升期望奖励。

#### PPO + RLHF 的典型挑战

| 挑战 | 说明 |
|------|------|
| **训练不稳定** | 奖励尺度、优势归一化、学习率、clip 系数、熵 bonus 需联合调节；策略与价值网络估计滞后于分布漂移 |
| **Reward hacking** | 模型利用 RM 漏洞刷分（冗长、固定讨好句式、格式技巧），与人类真实偏好背离 |
| **算力与显存：四模型** | 经典实现需同时维护 **(1) 策略 \(\pi_\theta\)**、**(2) 旧策略 / rollout 缓存用于 ratio**、**(3) 参考模型 \(\pi_{\mathrm{ref}}\)**、**(4) 奖励模型 \(r_\phi\)**；若使用 **价值网络**，则再加 **critic**。业界常说「四个大模型」量级开销，指 **多路前向** 与优化器状态叠加，对显存与吞吐压力极大 |

---

### 1.6 DPO（Direct Preference Optimization）

#### 关键洞见：隐式奖励模型

在 BT 偏好假设与一定最优性条件下，可将 **隐式奖励** 与策略、参考策略通过 **配分函数** 重参数化，从而 **不必显式训练 RM**，也 **不必在线 RL rollout**。

#### DPO 损失（交叉熵形式在偏好对上）

设 \(\sigma\) 为 logistic，\(\beta>0\) 控制偏离参考模型的强度。DPO 常写为：

\[
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta\left(
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)\right)\right]
\]

**直觉分项**：

- \(\log \pi_\theta(y_w\mid x) - \log \pi_\theta(y_l\mid x)\)：提高 chosen、压低 rejected 的似然；
- 减去 \(\log \pi_{\mathrm{ref}}\)：**相对参考模型归一化**，避免把两边概率一起抬高；
- \(\beta\)：越大越强调偏好对比，过大可能不稳定或损害通用行为。

从形式上看，这是对 **偏好对** 的 **负对数似然（交叉熵）** 风格目标，实现上类似监督学习，**稳定且简单**。

#### DPO 的优点

- **不需要单独 RM**；
- **不需要 PPO 式 rollout、价值网络**；
- **离线数据**上直接更新，工程链路短、复现性好。

#### DPO 的局限

- 依赖 **离线偏好分布**，对训练后策略新错误的覆盖可能不足；
- **模式坍缩** 风险：过度强化某些「chosen」模式；
- 数据质量仍是上限。

#### DPO vs RLHF（PPO+RM）对照

| 维度 | RLHF（PPO+RM） | DPO |
|------|----------------|-----|
| RM | 显式训练 | 隐式（含在损失里） |
| RL | 需要采样与优势估计 | 通常不需要 |
| 价值网络 | 常用 | 不需要 |
| 稳定性 | 调参难 | 相对稳 |
| 在线探索 | 可设计在线管线 | 典型为离线；也有迭代 DPO |
| 典型风险 | 系统复杂、reward hack | 分布偏移、模式单一 |

---

### 1.7 GRPO（Group Relative Policy Optimization）

#### 与 DeepSeek-R1 等推理增强路线

**GRPO** 在公开讨论中常与 **DeepSeek-R1** 等 **长思维链 + 强化学习** 路线一起出现：对 **数学 / 代码** 等 **可验证任务**，用 **规则或执行反馈** 作奖励，通过 **同一 prompt 下多次采样** 构造 **组内基线**，从而 **弱化经典价值网络**。

#### 不需要（或弱化）Critic / Value Model

经典 PPO 用 \(V_\psi(s)\) 作 baseline 降方差。GRPO 对同一 \(x\) 采样 \(G\) 个回答 \(\{y^{(i)}\}_{i=1}^G\)，得奖励 \(R_i\)，用 **组内均值** 构造优势，例如：

\[
A_i = R_i - \frac{1}{G}\sum_{j=1}^G R_j
\]

再代入策略梯度或 **PPO-style clip** 更新。这样用 **统计基线** 替代 **对所有状态学习一个全局 value**，在 **终端稀疏奖励**、**可验证结果** 场景尤其自然。

#### 规则奖励与可验证任务

- **数学**：答案是否与标准解一致（符号化、数值容差、`\\boxed{}` 解析等）；
- **代码**：单元测试、隐藏测例、编译是否通过。

这类 **Outcome Reward** 比纯神经 RM 更难被「空话」欺骗，适合 **推理链** 与 **RLVR** 叙事。

#### GRPO vs PPO

| 维度 | 经典 PPO（RLHF） | GRPO |
|------|------------------|------|
| Baseline | 学习的 \(V_\psi\) 为主 | 组内均值等 **相对基线** |
| 奖励 | 常为学得 RM | 常为 **可验证 / 规则** |
| 采样 | rollout | **同 prompt 组采样** |
| 适用 | 开放域偏好 | 数学、代码等 **对错清晰** 任务 |

---

### 1.8 其他对齐方法（简表）

| 方法 | 核心思想 | 备注 |
|------|----------|------|
| **RLAIF** | 用 **强模型**（或专用评判模型）代替人类生成偏好对，再走 RM+RL 或 DPO | 降标注成本；偏见会 **从教师模型传递** |
| **Constitutional AI** | 用 **宪法式原则** 引导模型 **自评、改写**，可再经 RLHF/DPO 强化 | 减少部分人工标注；原则仍需人设计 |
| **Rejection Sampling（拒绝采样）** | 从策略采多个候选，用 RM 或规则 **选最优**，可仅做 SFT 微调或作偏好数据构造 | 简单但 **推理时多倍算力**；适合中等规模提质 |

---

### 1.9 安全对齐：红队与安全 RLHF

- **红队（Red-teaming）**：有组织地 **模拟攻击者**（越狱提示、诱导有害输出、隐私套取等），发现模型漏洞，再 **回流数据与策略**（SFT、偏好、策略约束）。是 **评测—迭代** 闭环的关键环节，不能仅靠静态基准分数。
- **Safety RLHF**：在通用 RLHF 流程中，将 **安全相关偏好**（拒绝恶意请求、降低毒性）显式纳入 **RM 训练数据** 或 **奖励 shaping**，使 PPO/DPO 目标与安全指标一致。常与 **内容审核分类器**、**策略约束**、**宪法** 组合使用。

**面试一句**：安全不是「训一次 RM 就结束」，而是 **持续对抗评测 + 数据飞轮**。

---

### 1.10 CS336 Assignment 5 中的对齐组件（与课程叙事对齐）

Stanford **CS336 Assignment 5（Alignment）** 在常见大纲中把抽象对齐技术落到 **可复现 pipeline**，与本课概念对应关系可记为：

1. **SFT 子模块**：在 **数学推理** 等任务上，用指令数据教会 **格式与指令遵循**（如可解析答案、模板）；为后续 RL 提供 **稳定策略起点** 与 **参考模型**。
2. **GRPO 子模块**：在 **可验证奖励**（如判题、执行结果）下做 **组内相对优势** 优化，体会 **无需单独 value model** 的 RL 形态，与经典 **PPO+RM** 对照。
3. **可选 DPO 子模块**：用 **安全相关偏好对** 做 **直接偏好优化**，理解 **隐式奖励** 与 **KL 隐含在 log-ratio** 中的实现细节。

**作业层面一句话**：在指令跟随基座上，用 **GRPO + 规则奖励** 强化推理；可选 **DPO** 做安全偏好对齐。具体函数名与检查点以 **当年官方仓库 / PDF** 为准；动手路线见 [Lesson 19](./19-Assignment5对齐实战.md)。

---

## 二、代码（Code）

下列为 **教学级伪代码**，重在 API 形状与概念对应；真实框架（TRL、OpenRLHF、Verl、课程仓库等）在 mask、分布式、旧策略缓存上会有更多细节。

### 2.1 奖励模型：Bradley–Terry / Pairwise Logistic

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_rm_loss(rm: nn.Module, x_tokens, y_w_tokens, y_l_tokens) -> torch.Tensor:
    """rm(x, y) -> 标量 reward，batch 维 (B,)"""
    r_w = rm(x_tokens, y_w_tokens)
    r_l = rm(x_tokens, y_l_tokens)
    return -F.logsigmoid(r_w - r_l).mean()
```

**要点**：注意 padding mask、长度偏置、奖励尺度与后续 PPO 归一化一致。

### 2.2 PPO：Clipped Surrogate 与 KL（示意）

```python
def clipped_surrogate_ratio(ratio, advantage, eps=0.2):
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    return torch.minimum(unclipped, clipped).mean()

def kl_penalty_per_sequence(logp_theta, logp_ref):
    # 常对有效 token 求和或按长度归一，再与 beta 相乘并入总目标
    return (logp_theta - logp_ref).sum(dim=-1).mean()
```

**要点**：语言模型需正确处理 **因果 logits**、**旧策略采样**、**GAE** 与 **KL 估计器**；显存上需规划 **policy / ref / RM / (value)** 多路前向。

### 2.3 DPO：直接偏好损失

```python
def dpo_loss(pi_theta, pi_ref, x, y_w, y_l, beta: float):
    def seq_logprob(policy, x_, y_):
        # 对非 pad token 的 log pi(y|x) 求和，形状 (B,)
        return policy.logprob_sum_conditional(x_, y_)

    logp_w_theta = seq_logprob(pi_theta, x, y_w)
    logp_l_theta = seq_logprob(pi_theta, x, y_l)
    logp_w_ref = seq_logprob(pi_ref, x, y_w)
    logp_l_ref = seq_logprob(pi_ref, x, y_l)

    inside = beta * ((logp_w_theta - logp_w_ref) - (logp_l_theta - logp_l_ref))
    return -F.logsigmoid(inside).mean()
```

**要点**：\(\pi_{\mathrm{ref}}\) 通常 **冻结**；\(\pi_\theta\) 由参考初始化；\(\beta\) 与 batch 构造影响极大。

### 2.4 GRPO：组内相对优势

```python
def group_relative_advantages(rewards_group: torch.Tensor) -> torch.Tensor:
    """rewards_group: (G,) 同一 prompt 的 G 条轨迹标量奖励"""
    return rewards_group - rewards_group.mean()

# 后续将 advantages 接入策略梯度或 PPO-style clip（依课程实现而定）
```

**要点**：\(G\) 增大可降低方差但增加采样算力；奖励需在同一评判标准下可比。

---

## 三、面试要点（Interview points）

### 3.1 一句话速记

- **RLHF**：SFT → RM（BT pairwise）→ PPO + KL 锚定 \(\pi_{\mathrm{ref}}\)。
- **RM**：学 \(r_\phi(x,y_w) > r_\phi(x,y_l)\)；推理时标量奖励驱动 RL。
- **PPO**：clip 限步长；KL 抑制偏离与 reward hacking；**多模型前向** 推高算力与显存。
- **DPO**：隐式奖励；对比 **log-ratio**；无显式 RM、无典型 RL 循环。
- **GRPO**：**组采样** + **组均值基线** + **可验证奖励**；推理任务友好。
- **在线 / 离线**：偏好数据是否随当前策略持续刷新。
- **对齐税**：对齐后部分通用能力或广度可能下降。
- **RLVR**：可验证奖励减轻主观 RM 偏差，适合代码与数学。

### 3.2 高频对比（白板级）

| 主题 | 答法骨架 |
|------|----------|
| BT 与 RM | BT 给出 \(P(\text{win})=\sigma(r_w-r_l)\)；RM 学 \(r_\phi\) 逼近人类效用 |
| PPO clip | ratio 超出 \([1-\epsilon,1+\epsilon]\) 时被截断，防止一步更新过大 |
| DPO 各项 | 相对参考的 log-ratio 差；\(\beta\) 控制偏离参考的强度 |
| GRPO baseline | 组内减均值 ≈ 控制 prompt 难度差异的相对排序信号 |
| 四模型成本 | policy、ref、RM、（value）；多路前向 + 优化器状态 |

---

## 四、面试深度问答（12+ 题详解）

### Q1：RLHF 的三个步骤分别解决什么问题？

**答**：（1）**SFT**：把预训练模型变成 **遵循指令、会对话格式** 的策略，并提供 **RL 起点** 与 **参考模型** 初值。（2）**奖励模型**：把 **人类偏好** 压缩成 **可微对比信号** \(r_\phi(x,y_w) > r_\phi(x,y_l)\)，供 RL 使用。（3）**PPO（+KL）**：在 RM 标量奖励下 **提升策略**，同时用 KL **限制与 SFT 的偏离**，缓解 RM 近似误差带来的 **过度优化**。

---

### Q2：奖励模型如何训练？Bradley–Terry 起什么作用？

**答**：数据为 **\((x,y_w,y_l)\)**。BT 假设 \(P(y_w \succ y_l\mid x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\)。训练最小化 **负对数似然** \(-\log \sigma(r_w-r_l)\)，使被人类选中的回答得分更高。BT 提供了 **配对比较** 与 **标量奖励** 之间的概率桥梁，便于用 **二元交叉熵** 训练 RM。

---

### Q3：PPO 的 clip 目标在优化什么？ratio 过大或过小会怎样？

**答**：在 **重要性采样** 下用旧策略数据更新新策略，clip 限制 **\(r_t=\pi_\theta/\pi_{\mathrm{old}}\)** 偏离 1 的程度。**ratio 过大**：更新步长过大，策略剧变、价值估计失效、训练不稳定。**ratio 过小**：有效梯度被截断，更新保守。clip 在 **步长与稳定性** 间折中。

---

### Q4：RLHF 里 KL 惩罚的目标是什么？和「贴近 SFT」有什么关系？

**答**：KL 约束 \(\pi_\theta\) 接近 **\(\pi_{\mathrm{ref}}\)**（常为 SFT）。RM 不能覆盖所有行为；无 KL 时策略可能 **利用 RM 漏洞** 得高分但输出 **有害或无用**。**贴近 SFT** 即保留预训练+SFT 已学的 **有用能力与语言质量**，在 **信任域** 内优化偏好。

---

### Q5：为什么说经典 RLHF+PPO「贵」？「四个模型」指什么？

**答**：一次训练步往往涉及：**当前策略** 前向/反向、**旧策略** 存 logits 或重算以算 ratio、**参考模型** 前向算 KL、**奖励模型** 前向算回报；若使用 **价值网络**，再叠加 critic。显存与算力接近 **多份大模型** 同时驻留或频繁切换，故常称 **四模型量级** 开销（具体是否含 value 依实现而定，面试讲清 **多路前向** 即可）。

---

### Q6：DPO 的核心洞见是什么？为什么不需要显式 RM？

**答**：在 BT 与某些正则化假设下，**最优策略与隐式奖励** 可写成仅依赖 **\(\pi_\theta\)** 与 **\(\pi_{\mathrm{ref}}\)** 的 **闭式关系**，从而偏好似然可直接对策略参数优化，**RM 被消去** 或 **隐含在 log-ratio 中**。实现上是对偏好对的 **sigmoid 交叉熵**，无需单独训练 \(r_\phi\)。

---

### Q7：写出 DPO 损失并解释 \(\beta\)。

**答**：\(\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}[\log\sigma(\beta(\Delta_w-\Delta_l))]\)，其中 \(\Delta_y=\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}\)。**\(\beta\)** 控制 **偏好对比强度** 与 **偏离参考的程度**：\(\beta\) 大则更强调「chosen 相对 rejected 的边际」，但过大可能训练不稳或 **过拟合偏好数据**。

---

### Q8：DPO 与 RLHF（PPO+RM）如何选择？

**答**：**DPO** 适合 **静态偏好数据**、希望 **快速迭代、系统简单**、团队 RL 工程经验有限。**RLHF+PPO** 适合需要 **在线采样、探索式纠错、复杂奖励 shaping** 的场景，但承担 **调参与不稳定性**。许多产品先 **DPO 上线**，再视需要叠 RLHF 或混合。

---

### Q9：GRPO 与 PPO 的本质区别是什么？

**答**：**PPO** 是通用 **on-policy** 优化框架，RLHF 中常配 **学得 RM + value**。**GRPO** 强调 **同一 prompt 组内多条样本**，用 **组内相对排名/减均值** 作优势，常配 **可验证奖励**，从而 **弱化或避免单独 value model**。二者可共享 **clip** 等稳定技巧，但 **基线来源与奖励类型** 不同。

---

### Q10：GRPO 为什么可以不需要 Value Model？

**答**：Critic 用于估计 **状态值** 以降低方差。GRPO 在同一 \(x\) 下采 \(G\) 条轨迹，用 **组平均奖励** 作 **逐样本 baseline**，优势近似 \(R_i-\bar R\)，在 **终端奖励、同题可比** 的设置下提供 **零成本（无额外网络）的方差缩减**。这不等于所有任务都不需要 value，而是 **任务结构使组基线足够有效**。

---

### Q11：什么是 reward hacking？如何缓解？

**答**：策略找到 **奖励函数的捷径**（冗长、套话、刷格式）以得高分，但 **人类不满意**。缓解：**KL 到参考**、**奖励工程**（长度归一、多样性约束）、**红队与数据迭代**、**可验证奖励（RLVR）**、**多 RM 集成** 等。

---

### Q12：在线与离线偏好学习有何区别？

**答**：**离线**：固定数据集上训练（典型 DPO），成本低、可复现；风险是 **分布偏移**。**在线**：训练时用当前策略生成候选再标注，信号对准 **当前错误**；成本高、系统复杂。比喻：**离线如刷题库，在线如边考边改错**。

---

### Q13：对齐税（alignment tax）是什么？如何观察与缓解？

**答**：为获得 **更安全、更听话**，在 **其他能力**（如创意、部分知识问答）上 **性能下降**。原因：KL、偏好数据偏向保守、目标与预训练不一致等。缓解：**预训练数据混合回放**、**多任务偏好**、**评测驱动调 \(\beta\) 与数据配比**。

---

### Q14：RLAIF 与 Rejection Sampling 各是什么？各有什么代价？

**答**：**RLAIF** 用 **AI 生成偏好** 替代部分人类标注，再走 RM/RL 或 DPO；**代价** 是教师模型的 **偏见与盲点** 会传递。**Rejection Sampling** 对同一 prompt **多采样**，用 RM/规则 **取最优**；**代价** 是 **推理与训练时采样倍数** 的算力开销，但实现简单、易与 SFT 结合。

---

## 五、练习（Practice）

1. 写出 BT 假设下的 pairwise logistic 损失，并说明与 **二元分类交叉熵** 的联系。
2. 推导：将 DPO 公式展开为仅含 \(\log\pi_\theta\) 与 \(\log\pi_{\mathrm{ref}}\) 的差，并标注 \(\beta\) 出现位置。
3. 手算：同一数学题 4 个样本奖励为 \([1,0,0,1]\)，求组内优势向量。
4. 解释 PPO 中若 **删除 clip**、仅保留 KL，训练可能出现什么现象？
5. 举两个 **reward hacking** 例子，并各给一条 **非神经网络** 缓解手段。
6. 对比 **RLAIF** 与 **人类标注** 在 **成本、偏差、适用场景** 三维上的差异。
7. 设计一个最小实验：固定 SFT 模型，比较 **仅 SFT** vs **SFT+DPO** 在安全评测集上的拒答率与有用性（需定义评测协议）。
8. 说明 **红队**  findings 如何回流到 **下一轮 RM 数据** 或 **DPO 偏好对**（流程图级描述即可）。

---

## 六、导航（Navigation）

| 上一课 | 下一课 |
|--------|--------|
| [Lesson 17：SFT 有监督微调](./17-SFT有监督微调.md) | [Lesson 19：Assignment 5 对齐实战](./19-Assignment5对齐实战.md) |

**相关链接**：[训练循环与损失函数](./07-训练循环与损失函数.md)、[课程总览](./00-课程总览与学习路线.md)。

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
| \(x\) | prompt / 上下文 |
| \(y_w, y_l\) | chosen / rejected |
| \(r_\phi\) | 奖励模型 |
| \(\pi_\theta\) | 当前策略 |
| \(\pi_{\mathrm{ref}}\) | 参考策略（常冻结） |
| \(\beta\) | DPO 温度系数或 RL 中 KL 系数（语境依章节） |
| \(\sigma\) | logistic 函数 |
| \(\mathrm{KL}\) | Kullback–Leibler 散度 |
| \(G\) | 组采样条数 |

---

> **学习建议**：先确保 [Lesson 17](./17-SFT有监督微调.md) 中 **mask 与参考模型角色** 清晰，再对照本课 **BT → PPO → DPO → GRPO** 串成一条线；动手请完成 [Assignment 5 实战](./19-Assignment5对齐实战.md) 中的损失与解析器，把公式跑通。

*文档版本：CS336 面试导向 · Lesson 18 · 对齐技术总览；作业细节以官方当年说明为准。*
