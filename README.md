# 📈 A股互动易问答质量评估与逃避战术智能识别系统
> **基于 LCPPN 架构与知识蒸馏的工业级金融 NLP 情报提炼管线**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-⚡-green.svg)
![Transformers](https://img.shields.io/badge/Transformers-🤗-yellow.svg)

## 🌟 项目简介

在高度复杂的 A 股金融市场中，上市公司管理层在互动易等平台的问答质量是极具价值的量化情绪信号。然而，真实的金融文本充斥着海量的“客套废话”，且高管在面对尖锐问题时常采用“战略性模糊”、“外部归因”等逃避战术。

本项目旨在构建一套端到端（End-to-End）的高精度文本情报提炼引擎。项目彻底摒弃了传统的扁平化分类与浅层 TF-IDF 特征，创新性地引入了：
* **前置实体防线**：基于 `FlashText` 的金融实体极速拦截，彻底根除“闭世界假设”的噪音污染。
* **半监督知识蒸馏**：利用大模型（LLM）构建金标准，训练 1 亿参数的 Teacher 模型输出“软标签（Soft Labels）”，成功蒸馏出速度翻倍、Macro-F1 高达 0.77+ 的轻量级 Student 模型。
* **LCPPN 层级路由架构**：(Local Classifier Per Parent Node) 动态路由分类树，将极其复杂的意图识别解耦为 Root 宏观路由与多个 Sub-node 领域专家分类器。
* **Train-Val 绝对隔离的少数类增强**：运用 LLM 对极其稀缺的战术样本（如“外部归因”）进行同义重写与场景泛化，在绝对不污染验证集的前提下抹平极端长尾分布。

## 🏗️ 核心架构 (LCPPN)

系统采用层级递进的漏斗式架构，精准剥离逃避战术并提炼财务信号：

1. **Level 0 (网关门卫)**: `FlashText` + 信息熵过滤，拦截无金融实体、纯客套的低质闲聊。
2. **Level 1 (根节点分类)**: 判定回答的语用学立场，分流至 `Direct (直接)`、`Intermediate (避重就轻)`、`Evasive (打太极)`。
3. **Level 2 (子节点专家)**:
   - **财务质量分支 (Direct)**: 5 分类业务归因（资本运作、产能规划、技术研发等）。
   - **逃避战术分支 (Evasive)**: 4 分类心理归因（转移话题、战略性模糊、外部归因、推迟回答）。

## 📂 目录结构

```text
├── app/                        # 线上推理服务 (FastAPI + TailwindCSS前端)
│   ├── main.py                 # VRAM多路复用与LCPPN路由核心接口
│   └── index.html              # 瀑布流雷达诊断前端界面
├── data/                       # 数据集与字典 (注: 实际数据因隐私/体积未上传)
│   ├── others/                 # THUOCL、baostock 等金融专属实体字典与停用词
│   └── processed/              # 清洗与 LLM 标注后的高质量流转数据
├── models/                     # 训练完成的模型权重存放目录 (需自行下载或训练)
├── notebooks/                  # 数据分析与实验记录 (Jupyter Notebook)
│   ├── 00_Rejected_Cases_Analysis.ipynb
│   ├── 01_Model_Evaluation.ipynb    # [WIP] 学生模型性能与泛化能力评估
│   └── 02_Bad_Case_Analysis.ipynb   # [WIP] LCPPN 错题本与主动学习回流分析
├── src/                        # 核心 NLP 训练管线 (Pipeline)
│   ├── 00_Data_Preprocess.py     # 基础数据清洗与对齐
│   ├── 01_Entities_Filter.py     # FlashText 极速实体白名单/黑名单双重过滤
│   ├── 02_Entropy_Calculated.py  # 信息增益与 Jaccard 相似度降噪
│   ├── 03_LLM_Labeling.py        # 零样本 CoT 高质量金标准构建
│   ├── 04~05_Teacher_*.py        # Teacher (RoBERTa-Cross-Encoder) 训练与离线打标
│   ├── 06_Student_Model_Distillation.py # Student (DistilBERT) 软标签蒸馏
│   ├── 07~08_LCPPN_Subnodes_*.py # LCPPN 子节点数据分流与加权微调
│   └── 09_Inference_Pineline.py  # 离线端到端联调测试脚手架
├── utils/                      # 工具脚本 (抽样、LLM 数据增强隔离等)
└── requirements.txt            # 项目依赖
