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
支持批量上传问答对数据，适合监管排查与历史样本回放（单次上限 500 条）。

### 3. 实体与意图高亮
对文本中的关键财务实体进行高亮展示，并在顶部输出业务标签或高风险标签。

### 4. 置信度雷达图
对于直接回答、部分响应、逃避回答样本，均可展示子节点概率分布雷达图，帮助快速理解模型判断。

### 5. 低置信度数据飞轮
当模型对第一层或第二层判断置信度不足时：
- 样本自动进入待复核队列
- 阈值规则：第一层置信度 < 0.65 或第二层置信度 < 0.65
- 由人工确认或修正结果
- 可选让 Agent 给出复核建议
- 分析页支持“手动加入待复核队列”，用于高置信但疑似误判样本
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

如果你使用 PyCharm 或本地静态服务，页面端口可能是随机的（如 `127.0.0.1:3611`）。
建议在 `.env` 配置：

```env
QA_ALLOW_ORIGIN_REGEX=^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$
```

否则浏览器对 `POST /api/evaluate` 的预检请求（`OPTIONS`）可能返回 `400 Disallowed CORS origin`，前端表现为 `Failed to fetch`。

复核入队阈值可通过以下环境变量调整（支持 `0~1` 概率或百分比）：

```env
QA_REVIEW_ROOT_THRESHOLD=0.65
QA_REVIEW_SUB_THRESHOLD=0.65
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

可选：先基于现有问答语料自动发现候选词条（人工审核后再补充到 `data/others/custom_entities.txt`）。

```bash
python utils/discover_candidate_terms.py --max-rows 200000 --top-k 500 --min-count 20
```

可选：运行候选词人工审核脚本，逐条输入 `y / n / s`，并自动回流词表。

```bash
python utils/review_candidate_terms.py --input data/others/candidate_entities.auto.tsv
```

脚本输出与回流行为：
- 重写 `data/others/custom_entities.reviewed.txt`（本次审核选择 `y` 的词）
- 重写 `data/others/custom_stopwords.reviewed.txt`（本次审核选择 `s` 的词）
- 仅追加新增词到 `data/others/custom_entities.txt`
- 仅追加新增词到 `data/others/custom_stopwords.txt`（不存在则创建）

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

如果仅做前端联调且本机未启动 PostgreSQL，可在 `.env` 中保留
`QA_DATABASE_FALLBACK_TO_SQLITE=true`（默认即为 true），
后端在 PostgreSQL 不可用时会自动回退到 `data/processed/stage_c.db`。
若不想打印完整数据库异常细节，可设置 `QA_DATABASE_FALLBACK_VERBOSE=false`（默认值）。

### 5. 打开前端页面

#### 过渡静态页（兼容保留）

```text
app/index.html
app/review.html
```

#### React + TS 产品化前端（阶段D）

```bash
cd frontend
npm install
npm run dev
```

默认地址：`http://127.0.0.1:5173`（或 Vite 输出地址）。

### 6. 核心 API（阶段D）

- `POST /infer`：单条问答推理
- `POST /api/evaluate`：与原型前端兼容的单条推理接口（响应含 `sample_id` 与复核状态）
- `POST /batch_infer`：创建批量分析任务
- `GET /jobs/{job_id}`：查询批任务状态与结果
- `GET /jobs/{job_id}/export`：导出批任务结果 CSV
- `GET /review/queue`：分页拉取待复核样本队列
- `GET /review/{sample_id}`：查看样本详情、模型输出、Agent建议、人工记录
- `POST /review/{sample_id}/enqueue`：手动把当前样本加入待复核队列
- `POST /review/{sample_id}/agent-suggestion`：人工触发 Agent 建议（异步）
- `POST /annotate`：人工确认/修改并回流训练集

`/infer` 与 `/api/evaluate` 当前响应中已包含：
- `root_probabilities`：根节点全概率分布
- `sub_probabilities`：子节点全概率分布
- `entity_hits`：实体命中（含 `text/start/end/source_text`）

### 7. Dify 工作流 I/O 契约（复核建议）

为保证 `AgentService` 与 Dify 工作流稳定对接，建议按以下契约配置：

- 输入变量（Workflow `inputs`）：
  - `question`：投资者提问文本
  - `answer`：董秘回答文本
  - `model_result`：模型初判（默认以 JSON 字符串传入，兼容 Dify `text-input`）
- 输出内容（建议）：
  - 返回一个 JSON 对象，字段为 `root_label`、`sub_label`、`confidence`、`reason`
  - `root_label` 仅允许：`Direct (直接响应)` / `Intermediate (避重就轻)` / `Evasive (打太极)`
  - `confidence` 为 0~1 浮点数
  - 可选补充 `evidence` 字段，存放模型判定证据片段

后端默认按以下路径依次读取 Dify 输出：`answer`、`output_text`、`data.answer`、`data.outputs.result` 等。
若你的工作流输出字段不同，可用环境变量 `QA_DIFY_OUTPUT_PATH` 指定（例如 `data.outputs.result`）。
若你的工作流把 `model_result` 改成对象输入，可将 `QA_DIFY_MODEL_RESULT_AS_JSON=false`。
若你希望控制 Agent 建议等待时长，可设置：
- `QA_DIFY_CONNECT_TIMEOUT`（默认 5 秒）
- `QA_DIFY_WRITE_TIMEOUT`（默认 10 秒）
- `QA_DIFY_READ_TIMEOUT`（默认 20 秒，超时后自动回退为模型结论）

兼容说明（后端自动归一）：
- `root_label` 若输出 `Direct/Intermediate/Evasive/Fully Evasive`，会自动映射到系统标准标签。
- `sub_label` 在 `Evasive` 下若输出 `以定期报告为准` 等同义表述，会自动映射为 `推迟回答`。

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
   - 财务实体高亮或子节点概率雷达图
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
- 阶段D 前端升级：React + TypeScript 三页面（分析页、批量任务页、待复核页）
- 阶段D 可解释结果：实体高亮、概率分布与子节点雷达图（Direct/Intermediate/Evasive）
- 阶段D 批任务能力：任务进度轮询与 `GET /jobs/{job_id}/export` CSV 导出

### 计划中
- 数据飞轮与 LoRA 微调自动训练流水线
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

## 阶段E运行手册（Docker Compose）

阶段E主入口锁定为 `api + worker + redis + postgres`，不包含前端服务。

### 1) 启动

```bash
docker compose up --build -d postgres redis api worker
docker compose ps
```

### 2) 健康检查

```bash
curl http://127.0.0.1:8000/api/health
```

预期返回 `status=ok`，并可看到模型初始化状态与任务仓库状态。

### 3) 阶段E关键配置约束

- Compose 场景强制 `QA_DATABASE_FALLBACK_TO_SQLITE=false`
- `api/worker` 仅在依赖服务 healthy 后启动
- `api` 使用非 `--reload` 启动模式
- `api/worker` 不允许运行时 `pip install`

### 4) 常见故障排查

1. `api` 启动失败且提示数据库不可用  
   检查 `postgres` 容器是否 healthy；确认 `QA_DATABASE_URL` 指向 `postgres:5432`。
2. 触发 Agent 建议任务后长期 pending  
   检查 `worker` 日志与 `redis` 状态：`docker compose logs worker redis --tail=200`。
3. 前端 `Failed to fetch`  
   检查 CORS 配置（`QA_ALLOW_ORIGIN_REGEX`）以及后端接口地址是否指向 `http://127.0.0.1:8000`。

### 5) 演示流程建议

1. 分析页输入问答，触发模型推理。  
2. 低置信度样本自动入队，或手动加入待复核队列。  
3. 在复核页触发 Agent 建议并轮询 `/jobs/{job_id}`。  
4. 查看复核详情并完成导出。
