# 📈 Financial QA Quality Assessment: NLP & BERT Fine-tuning Pipeline
**基于多维特征与大模型微调的金融互动问答质量自动分级系统**

## 📖 1. 项目概况 (Project Overview)
在证券市场中，投资者与上市公司存在严重的信息不对称，且监管互动平台常充斥大量“套话”与“废话”。
本项目摒弃了传统的纯人工审核或简单的文本长度过滤，采用 **“小样本回归归纳规则 -> 海量数据弱监督打标 -> 深度学习微调泛化”** 的全栈工程管线。旨在自动化识别并抽取高质量的董秘问答，为后续的金融研报或大模型 RAG（检索增强生成）知识库提供纯净的数据源。

## ⚙️ 2. 核心架构与代码模块 (Pipeline Architecture)
项目代码严格按照数据流向解耦为核心执行模块：

### 核心数据流转图
```mermaid
graph TD
    A([110万原始 A 股问答数据]) --> B[01_数据清洗与过滤]
    
    subgraph 弱监督打标阶段 Weak Supervision
    B --> C{02_特征工程提取}
    C --> D[信息熵 42%]
    C --> E[相关度 23%]
    C --> F[数字密度 15%]
    C --> G[时效性 13%]
    C --> H[长度惩罚 -6%]
    D & E & F & G & H --> I(计算综合质量得分 Quality Score)
    I --> J[全局分位数截断: Top30% / Bottom30%]
    end
    
    J --> K([10 万条高置信三分类数据集])
    
    subgraph 模型演进阶段 Model Evolution
    K --> L[03_Baseline: TF-IDF + 逻辑回归]
    K --> M[06_Tokenize: HuggingFace Dataset 映射]
    M --> N[07_LLM: BERT-base-chinese 大模型微调]
    end
    
    L -.-> O((基准准确率: 55.4%))
    N -.-> P((微调准确率: 88.2%))
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style O fill:#fdd,stroke:#333
    style P fill:#dfd,stroke:#333,stroke-width:2px
