# 端到端 RAG 教学案例：文档、检索、引用与评测

> 返回[总索引](README.md)。本例补齐[最小门禁示例](最小验收示例.md)跳过的文档处理和检索环节，默认不调用模型。它是小型教学管线，不是生产知识库、通用问答基准或权限服务。

## 一、运行入口与边界

仓库根目录执行，只需要 Python 标准库；默认离线、只输出 JSON，不写文件、不需要 GPU 或 API key：

```bash
python -m code.basic_study.rag_example --split dev
python -m code.basic_study.rag_example --split heldout
python -m unittest discover -s tests -p 'test_rag_example.py' -v
```

[代码](../../code/basic_study/rag_example.py)包含完整步骤；[测试](../../tests/test_rag_example.py)覆盖 BM25 手算、权限过滤、非法引用、HTTP 协议与失败计分。默认生成器名为 `extractive-baseline`：它选择一段原文作为答案，**不是 LLM**，也不具备一般语义理解能力。

## 二、数据与版本合同

使用自建虚构商店语料，不代表真实商家的退货、配送或员工政策。输入为 [manifest](../../code/basic_study/rag_data/manifest.json)与四份纯段落文件：[退货](../../code/basic_study/rag_data/corpus/refund.md)、[配送](../../code/basic_study/rag_data/corpus/shipping.md)、[客服](../../code/basic_study/rag_data/corpus/service.md)、[内部](../../code/basic_study/rag_data/corpus/internal.md)。

- 解析器只按空行分段，不支持一般 Markdown 的标题、表格、PDF/OCR。每段一个事实，形成 12 个 chunk。
- chunk ID 为文档 ID 加段序号，例如 `shipping:3`；这里依赖固定快照，增删/重排段落须更新版本与 gold，不能当作生产稳定 ID 方案。
- `readers` 由可信 fixture 给定。客户可读公开政策，员工另可读内部文档。每次查询先筛权限，再算 BM25 统计、排序与生成上下文；本例没有缓存。
- 角色不是客户端可自行声明的授权。离线报告会包含员工测试结果，只能用作测试材料；生产身份、租户、ACL 撤销、日志隔离需另外实现。

[开发集](../../code/basic_study/rag_data/dev.json)有 3 个退货问题；[固定教学留出集](../../code/basic_study/rag_data/heldout.json)有 8 个配送/客服/权限/多证据问题。两组 case ID 和 gold chunk 不重叠；问题与 gold 不参与建索引，gold 不发送给生成器。知识库是开放书证据，允许检索，不是测试标签泄漏。

这是公开、人工设计的小样本：只能示范集合隔离，不证明泛化。查看留出失败后若据此调参，该集合就成为开发材料；下一轮正式报告应另建未使用过的分组留出集。

## 三、实际执行链路

1. 读文件与元数据，按段落切片，生成来源 ID、版本与可读角色。
2. 用可信角色过滤 chunk；中文用重叠二元字符片段，ASCII 用字母/数字词项。这是教学分词，不等于 BPE 或专业中文分词。
3. BM25 检索 Top-4，记录分数和来源。`k1=1.5`、`b=0.75` 是示例配置，不是所有业务的最优值。
4. 离线生成器取最高分段落：问题词项覆盖率至少 0.5 才返回，否则拒答。问句停用词和阈值都是脆弱启发式，**不构成语义相关性或无答案检测保证**。
5. 校验输出 schema，复用已有引用门禁：来源必须属于本次检索证据，版本一致，quote 必须等于完整原段落；任一非法引用拒绝整个候选。
6. 独立对照 gold，分别统计检索召回、证据集合匹配、运行错误与失败样本。引用门禁通过不等于回答完整或正确。

**BM25 核对式**

对去重后的查询词项 $q$，授权语料含 $N$ 个 chunk，$f(t,D)$ 为词频、$n_t$ 为包含词项的 chunk 数：

$$
\operatorname{score}(q,D)=\sum_{t\in q}\log\!\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right)
\frac{f(t,D)(k_1+1)}{f(t,D)+k_1\left(1-b+b\frac{|D|}{\operatorname{avgdl}}\right)}.
$$

词频为零的项贡献零；空查询、无授权语料或零平均长度返回空检索。IDF 和参数含义可核对 [Lucene BM25Similarity](https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)；本例参数与分词不等于 Lucene 的默认配置。

## 四、结果怎么读

报告包含 fixture 版本、规范化 chunk 内容/权限的 SHA-256、集合、生成器、配置、逐题来源/分数/答案/错误。正式交付还应记录代码提交、Python 版本；模型调用另记录服务/模型版本、采样配置和用量，不能仅凭模型名称复现实验。

| 字段 | 定义与限制 |
| --- | --- |
| `retrieval_recall_at_k` | 有 gold 证据的题，逐题召回比例再取均值；拒答题无 gold，值为 null，不纳入召回均值 |
| `gold_evidence_set_accuracy` | 所有题中，状态与通过门禁的来源集合同时匹配 gold 的比例；拒答题也计分，运行错误算失败。不是一般答案语义准确率 |
| `gate_rejected` | 输出声称回答，但来源/版本/完整原文不合法，被引用门禁拦截；不表示发现了所有幻觉 |
| `failure` / `errors` | 区分证据未召回、选择/拒答问题与生成运行错误；错误不从分母删除，不输出原始错误里的凭证 |

当前 v1 离线运行：开发集 3/3；固定留出集召回均值 1.0，证据集合匹配 7/8，运行错误为 0。这些结果由运行计算，不是预填模型成绩；数据或实现变化后应重新运行。

**必须保留的失败例**：`two_sources` 同时问配送时长和客服在线时间。检索已找齐两段，但单段摘录只答了一个子问题，引用合法却回答不完整。因此不能把“引用通过率”当作任务成功率，也不应为得到 100% 而删掉这题。

可比较一个不使用证据、总是拒答的简单基线：

```python
from code.basic_study.rag_example import evaluate

baseline = evaluate(generator=lambda question, hits: {"status": "refused", "citations": []},
                    generator_name="always-refuse")
print(baseline["gold_evidence_set_accuracy"])  # v1 固定留出集为 2/8
```

后续只在开发材料上逐项尝试更好的分词、改写、rerank 或多证据选择，固定预算做消融；重新建留出集后报告收益和新增失败，不默认引入向量数据库或 Agent。

## 五、可选模型接入与未验证项

只有同时显式传入 `--endpoint` 和 `--model` 才会发网络请求。接口是非流式 chat-completions JSON：请求含 `model`/`messages`，响应读 `choices[0].message.content`，并解析为上述引用 JSON。协议依据见 [Chat API Reference](https://developers.openai.com/api/reference/resources/chat)；不保证每个兼容服务或模型支持相同选项。

例如你已自行启动兼容的本地模型服务，可将模型占位名称替换成该服务支持的真实名称：

```bash
python -m code.basic_study.rag_example --split dev \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model MODEL_NAME
```

本例不启动模型服务，也不自动选择模型、下载权重或购买 API。外部服务使用 HTTPS；若需鉴权，从安全环境注入 `RAG_API_KEY`，不把真实 key 写进代码/示例命令/报告。显式接入会把问题与授权片段发往所选服务，应先核实数据授权、保留政策和费用。

HTTP 调用使用 30 秒超时、1 MiB 响应上限，不跟随重定向、不自动重试；网络、响应格式或 schema 错误记录为 `error`，程序以非零状态结束，不伪装成正常拒答。模型只能选完整原段落，不支持自由改写/语义蕴含检查；协议测试通过也不证明模型遵守指令。

已验证：离线管线、固定样本与失败路径、本地 mock HTTP 收发/错误/重定向合同。**未验证**：真实 LLM 答案质量、实际模型服务兼容性与用量、生产身份/租户/并发、增量索引与缓存失效、PDF/表格解析、攻击防护和大规模性能。完整项目仍按[实践清单](实践与验收清单.md)验收。
