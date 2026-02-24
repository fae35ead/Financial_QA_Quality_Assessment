Financial QA Quality Assessment: NLP & BERT Fine-tuning Pipeline
**基于多维特征与大模型微调的金融互动问答质量自动分级系统**

1. 项目概况
在证券市场中，投资者与上市公司存在严重的信息不对称，且监管互动平台常充斥大量“套话”与“废话”。
本项目摒弃了传统的纯人工审核或简单的文本长度过滤，采用 **“小样本回归归纳规则 -> 海量数据弱监督打标 -> 深度学习微调泛化”** 的全栈工程管线。旨在自动化识别并抽取高质量的董秘问答，为后续的金融研报或大模型 RAG（检索增强生成）知识库提供纯净的数据源。

2. 核心架构与代码模块
项目代码严格按照数据流向解耦为 7 个核心执行模块：

（1）弱监督打标的“先验量化”
`01_train_weight.py`：摆脱主观臆断，在 50 条人工标注的黄金样本上训练多元线性回归模型，倒推并量化了 6 大文本特征的权重贡献度。

```markdown
### 特征权重多元回归分析结果
基于 50 条人工盲评样本的 `Linear Regression` 倒推权重，系统排除了占比仅 1.6% 的无用情感特征，最终确立了如下打分策略：

```text
█ 信息熵 (Entropy)       : 42.0%  [正相关]
█ 相关性 (Relevance)     : 23.0%  [正相关]
█ 数字密度 (Num Density) : 15.0%  [正相关]
█ 时效性 (Time)          : 13.0%  [正相关]
█ 长度惩罚 (Length)      : -6.0%  [负相关]

`02_main.py` & `05_combine_data.py`：根据回归分析结果，提取各年份的文本特征并进行 Min-Max 归一化。基于公式 `质量分 = 0.42*信息熵 + 0.23*相关性 + 0.15*数字密度 + 0.13*时效性 - 0.06*长度惩罚` 计算全局得分。随后采用全局分位数（Top 30% 为标签2，Bottom 30% 为标签0）完成海量数据的自动化打标。
`04_spot_check.py`：提供抽样盲测接口，确保弱监督标签在语义层面符合人类直觉。

（2）模型演进与评测
`03_baseline_TF-IDF.py`：将问题与回答拼接为上下文，构建传统机器学习基线。使用 `TfidfVectorizer(max_features=20000)` 提取词频特征，结合 `LogisticRegression(solver='saga')` 训练快速分类器。
`06_bert_data_process.py`：将 Pandas DataFrame 转化为 HuggingFace Dataset 原生格式，应用 `bert-base-chinese` 分词器，采用 `max_length=256` 进行固定长度截断与填充。
`07_bert_train.py`：加载预训练分类头 (`num_labels=3`)，开启 FP16 混合精度与按 Epoch 评估的早停策略，完成 BERT 模型的终极微调。

### 核心数据流转图
```mermaid
graph TD
    A([110万原始 A 股问答数据]) --> B[01_数据清洗与过滤]
    
    subgraph 弱监督打标阶段 (Weak Supervision)
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
    
    subgraph 模型演进阶段 (Model Evolution)
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

3. 性能评估
*(注：本评估采用 8:1:1 严格分层抽样拆分测试集，确保各质量类别的先验分布一致。)*

|         模型架构 (Model)      |          核心超参数 / 策略            | 测试集准确率 (Accuracy)   |                          特性分析                       |
| :--------------------------- | :------------------------------------| :------------------------| :------------------------------------------------------|
| **TF-IDF + LR (Baseline)**   | `max_features=20000`, `max_iter=2000`|       **55.45%**         |         丢失词序与上下文逻辑，极易受高频套话干扰。         |
|   **BERT-base-chinese**      |       `lr=2e-5`, `batch=16`, `FP16`  |       **88.23%**         | 具备深度上下文注意力机制，精准拟合了复杂的业务质量判断逻辑。 |

4. 局限性与工程反思
作为一个追求客观与严谨的工程项目，本系统目前存在以下已知局限性，这也是未来模型迭代的核心方向：
1. **回归先验集的规模瓶颈**：当前弱监督规则的权重配置（如信息熵占 42%），仅来源于 50 条人工盲评样本的回归计算。样本量级过小，其回归系数在百万级统计分布中可能存在局部过拟合风险，未来需将黄金校验集扩充至 1000+。
2. **循环验证**：BERT 的高准确率很大程度上证明了其对 `质量得分公式` 这一规则系统的完美拟合。未来应脱离规则体系，引入大语言模型 (如 GPT-4) 作为 Judge 进行 Out-of-Distribution 泛化能力测试。
3. **基准对比的非对称性与算力 ROI 评估**：当前管线仅采用了纯文本的 `TF-IDF + LR` 作为基线模型，BERT 凭借其深层注意力机制对其形成降维打击是符合预期的。但在真实的工业落地中，大模型的推理延迟和显存开销极高。未来更具指导意义的消融实验应是：将前期提取的高价值手工特征（信息熵、相关度等）直接喂给 XGBoost 或 LightGBM 等梯度提升树模型构建“强基线 (Strong Baseline)”。若树模型能在极低算力开销下达到相近的准确率（如 85%+），则需从业务视角重新评估部署亿级参数大模型的性价比与必要性。

5. 快速复现 (Quick Start)
本仓库遵循模块化解耦，请确保在根目录下存在 `functions.py`，并按照以下顺序执行脚本：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成基线模型与分层切分数据集
python 03_baseline_TF-IDF.py

# 3. 数据流转化与张量化
python 06_bert_data_process.py

# 4. 启动 GPU 炼丹炉微调模型
python 07_bert_train.py