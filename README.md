# A股问答对智能分析与监管合规轻量化平台

## 项目简介

这是一个面向 **A 股互动问答场景** 的轻量级智能分析产品原型，目标是把现有的金融 NLP 研究项目进一步包装成具备产品思维和业务落地能力的解决方案，用于展示在 **监管与合规场景** 下的实际应用价值。

项目基于现有的 **A 股问答对财务实体提取与董秘逃避战术鉴别系统** 演进而来，核心能力包括：

- 对输入的问答对进行两层识别：
  - **第一层**：判断董秘回答是“直接回答”还是“逃避回答”
  - **第二层**：
    - 若为直接回答，则执行 **财务实体提取**，输出结构化业务信息
    - 若为逃避回答，则执行 **逃避战术识别**，输出战术分类与置信度分布
- 对 **低置信度样本** 启动数据飞轮：
  - 默认进入人工复核队列
  - 可选使用 Agent 对低置信度样本进行复核建议
  - 复核结果沉淀为高质量样本，用于后续微调与迭代

这个项目不是面向真实商业上线的 SaaS 产品，而是一个 **面向实习招聘、面试展示与产品能力训练** 的完整产品化项目：既保留了模型与算法深度，也体现了从“模型工程”走向“业务产品”的思考能力。

---

## 解决什么问题

### 1. 行业内缺少垂直金融问答解析能力
通用 NLP 模型或大模型在复杂金融语境、修辞性逃避、董秘话术等场景下效果有限，难以低成本规模化处理海量问答对。

### 2. 人工处理效率低
监管人员或数据分析人员面对全市场每日新增问答时，依赖人工逐条筛选的效率较低，难以及时发现高风险样本或提取结构化情报。

### 3. 长尾类别识别弱
现有模型在第一层分类表现较好，但第二层某些细分类别仍存在明显长尾问题，需要通过低置信度复核与持续微调形成数据飞轮。

---

## 目标用户

### 主用户群 A：监管与合规端
- 交易所一线监管人员
- 证监局审查员
- 上市公司董秘办内审人员

### 次用户群 B：量化投研端
- 量化私募研究员
- 券商资管数据挖掘人员
- 第三方金融终端数据加工方

当前版本优先服务 **监管端场景**。

---

## 核心功能

### 1. 单条问答分析
支持用户直接粘贴一组问答文本，系统自动输出：
- 第一层分类结果（直接回答 / 逃避回答）
- 第一层置信度
- 第二层结果：
  - 财务实体提取结果
  - 或逃避战术概率分布

### 2. 批量 Excel / CSV 上传分析
支持批量上传问答对数据，适合监管排查与历史样本回放。

### 3. 实体与意图高亮
对文本中的关键财务实体进行高亮展示，并在顶部输出业务标签或高风险标签。

### 4. 置信度雷达图
对于逃避样本，展示各类逃避战术的概率分布雷达图，帮助快速理解模型判断。

### 5. 低置信度数据飞轮
当模型对第一层或第二层判断置信度不足时：
- 样本自动进入待复核队列
- 由人工确认或修正结果
- 可选让 Agent 给出复核建议
- 高质量复核结果进入训练集，用于模型持续优化

### 6. 私有化部署
系统设计支持私有化部署，适合监管与合规场景的使用要求。

---

## 当前模型能力概览

当前项目已有一套完整的模型训练与部署流程，包括：

1. 原始问答数据去噪
2. 金融实体识别与无关回答过滤
3. 使用 LLM 打标 1000 条数据训练 Teacher 模型（RoBERTa）
4. Teacher 对 10 万条数据打软标签，蒸馏 Student 模型（DistilBERT）
5. 针对少数类做 LLM 扩充，并基于 EvasionBench 规则生成：
   - 财务实体分类器
   - 逃避战术分类器
6. 最终完成模型推理部署

### 当前效果
- 第一层分类：整体指标在 **80%+**
- 第二层分类：整体约 **60% 左右**
  - 部分类别可达 **80%+**
  - 部分长尾类别仅约 **30%**
- 推理延迟较低，已支持 FastAPI 原型部署

---

## 技术栈

### 当前项目已使用
- Python 3.10+
- PyTorch
- Hugging Face Transformers
- FastAPI
- FlashText
- Jupyter Notebook

### 产品化推荐技术栈
#### 前端
- React
- TypeScript
- Ant Design
- ECharts
- React Query / Zustand

#### 后端
- FastAPI
- Celery
- Redis
- PostgreSQL
- MinIO（或兼容 S3 的对象存储）

#### 模型与训练
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- MLflow

#### 部署与运维
- Docker
- Docker Compose
- Kubernetes（可选）
- Prometheus + Grafana

---

## 项目结构（当前仓库）

```text
Financial_QA_Quality_Assessment/
├── app/                        # FastAPI 服务与前端原型页面
│   ├── api/                    # 路由层（含 review/annotate）
│   ├── core/                   # 配置、数据库、Celery基础能力
│   ├── models/                 # Pydantic 请求响应模型
│   ├── services/               # 推理、批量任务、复核、Agent服务
│   ├── tasks/                  # Celery 异步任务
│   ├── main.py                 # 应用装配入口
│   ├── index.html              # 分析页面（兼容 /api/evaluate）
│   └── review.html             # 低置信度复核工作台
├── data/                       # 数据集、字典与处理结果
├── models/                     # 模型权重目录
├── notebooks/                  # 实验分析与评估记录
├── src/                        # 核心训练与推理脚本
├── utils/                      # 工具函数
└── requirements.txt            # Python 依赖
```

---

## 如何运行

### 1. 创建环境

```bash
conda create -n qa_env python=3.10
conda activate qa_env
pip install -r requirements.txt
```

可选：复制一份环境变量模板并按本机路径调整模型目录。

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

### 2. 准备模型权重

如果仓库中没有现成模型权重，需要按顺序运行 `src/` 目录下的脚本完成训练与蒸馏：

```bash
python src/00_Data_Preprocess.py
python src/01_Entities_Filter.py
python src/02_Entropy_Calculated.py
python src/03_LLM_Labeling.py
python src/04_Teacher_Model_Train.py
python src/05_Teacher_Inference.py
python src/06_Student_Model_Distillation.py
python src/07_LLM_Subnode_Labeling.py
python utils/data_expand.py
python src/08_LCPPN_Subnodes_Train.py
```

### 3. 启动在线推理服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 启动异步任务（阶段C）

```bash
celery -A app.core.celery_app.celery_app worker --loglevel=info
```

若本机未部署 PostgreSQL / Redis，可先通过 Docker 启动基础服务：

```bash
docker compose up -d postgres redis
```

### 5. 打开前端页面

直接打开：

```text
app/index.html
app/review.html
```

如果后续接入 React 前端，可通过 Vite 或 Nginx 单独启动前端服务。

### 6. 核心 API（阶段C）

- `POST /infer`：单条问答推理
- `POST /api/evaluate`：与原型前端兼容的单条推理接口
- `POST /batch_infer`：创建批量分析任务
- `GET /jobs/{job_id}`：查询批任务状态与结果
- `GET /review/queue`：分页拉取待复核样本队列
- `GET /review/{sample_id}`：查看样本详情、模型输出、Agent建议、人工记录
- `POST /review/{sample_id}/agent-suggestion`：人工触发 Agent 建议（异步）
- `POST /annotate`：人工确认/修改并回流训练集

---

## 推荐的产品化运行方式（目标形态）

后续产品化版本建议采用以下服务组合：

- `frontend`：React + Ant Design
- `api`：FastAPI
- `worker`：Celery
- `db`：PostgreSQL
- `cache`：Redis
- `storage`：MinIO
- `model-service`：模型推理服务

推荐通过 Docker Compose 一键启动。

---

## 用户使用流程

### 场景 1：单条问答分析
1. 用户输入公司、时间和问答对文本
2. 系统执行两层识别
3. 前端展示：
   - 是否逃避
   - 置信度
   - 财务实体高亮或逃避战术雷达图
4. 若置信度低，则进入待复核队列

### 场景 2：监管批量排查
1. 用户上传 Excel / CSV 文件
2. 系统异步批量分析
3. 返回结果表格与高风险样本筛选
4. 对低置信度样本进行人工复核
5. 复核结果沉淀为训练数据

---

## Roadmap

### 已完成
- 金融问答数据清洗与降噪
- 金融实体过滤
- LLM 辅助打标
- Teacher-Student 蒸馏
- LCPPN 分层分类架构
- FastAPI 原型部署
- 阶段C（收缩版）数据飞轮闭环：低置信度入队、Agent建议辅助、人工复核回流
- PostgreSQL + Redis + Celery 技术栈接入（复核链路）

### 计划中
- 全量 React 复核后台
- 数据飞轮与 LoRA 微调自动训练流水线
- React 产品化前端
- 批量上传与导出
- 审计日志与私有化部署脚本

---

## 项目价值

这个项目的价值不只是“做出一个模型”，而是完整展示：

- 如何把一个金融 NLP 项目抽象成真实业务场景问题
- 如何从模型工程走向产品设计
- 如何围绕低置信度样本构建数据飞轮
- 如何为监管与合规场景设计轻量级产品原型

它适合用于：
- 实习招聘作品集
- 面试项目介绍
- 展示产品思维 + 技术理解 + 业务抽象能力

---

## License

本项目用于学术研究、个人学习、面试展示与产品原型训练。
