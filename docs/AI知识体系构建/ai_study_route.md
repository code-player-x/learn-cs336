# 大模型工程师知识图谱
## 一、基础能力(Fundamentals)
- 数学基础
    - 线性代数：矩阵、向量、特征值
    - 概率统计：贝叶斯、分布、期望
    - 微积分：梯度、链式法则
- 编程能力
    - Python
    - PyTorch / TensorFlow
    - NumPy / Pandas
- 计算机基础
    - 操作系统(Linux)
    - 网络：HTTP、API
    - 数据结构与算法

## 二、模型原理(Model Core)
- Transformer架构
    - Self‑Attention
    - Multi‑Head‑Attention
    - Positional Encoding位置编码
- 语言模型谱系
    - GPT(生成式)
    - BERT(编码器)
    - T5(编解码)
- Tokenizer分词机制
    - BPE / SentencePiece
    - Token成本与上下文窗口关系

## 三、训练体系(Training Pipeline)
- 预训练
    - 大规模语料
    - 自监督学习
- 微调技术
    - SFT有监督微调
    - RLHF
    - DPO
    - GRPO
- 数据工程
    - 数据集清洗
    - 数据标注
    - 去重与质量控制

## 四、推理与优化(Inference & Optimization)
- 推理策略
    - Temperature、Top‑P、Top‑K
    - Beam Search
- 性能优化
    - KV‑Cache
    - Continuous Batching连续批处理
    - Speculative Decoding投机解码
- 模型压缩
    - 量化：INT8、INT4、GPTQ、AWQ

## 五、工程架构(Engineering System)
- 推理运行时
    - vLLM
    - TensorRT‑LLM
    - ONNX Runtime
    - Ollama / Llama.cpp
- 部署方式
    - 云API
    - 私有化部署
    - 边缘部署
- 服务架构
    - FastAPI / Flask
    - 负载均衡
    - 可观测性监控

## 六、RAG系统(核心能力)
- 基本流程：用户查询 → 检索 → 拼接上下文 → 生成回答
- 核心组件
    - Embedding向量模型
    - 向量数据库
    - 检索算法：向量检索 / BM25
    - Chunk分块：大小、overlap重叠
    - Rerank重排
- 优化策略
    - 多查询改写

## 七、Prompt工程
- 基础技巧
    - 角色设定
    - Few‑shot
    - 输出格式控制
- 高级技巧
    - CoT思维链
    - ReAct
    - Prompt模版化

## 八、Agent系统
- 核心能力
    - 任务拆解
    - 工具调用
    - 多步推理
- 关键组件
    - Memory记忆
    - Planner规划器
    - Executor执行器
- 常用框架
    - LangChain
    - LlamaIndex
    - AutoGPT
    - CrewAI

## 九、评估体系(Evaluation)
- 自动评估
    - Perplexity
    - BLEU / ROUGE
    - LLM‑as‑Judge
- Benchmark测试集
- 人工评估
    - 准确性
    - 可读性
    - 安全性

## 十、安全与对齐
- 风险类型
    - 幻觉(Hallucination)
    - 偏见(Bias)
    - 越界输出(Jailbreak)
- 防护策略
    - Prompt层防护
    - 输出审查

## 十一、性能指标
- Latency延迟
- Throughput吞吐量
- GPU显存占用
- Token/s

## 十二、成长路径
- 初级：会调用API、搭建RAG、Agent原型
- 中级：模型部署、性能调优、生产落地
- 高级：架构设计、复杂多Agent、模型微调优化