# 金融问答对实体识别与逃避战术检测系统

## 项目简介
### 项目背景
本项目聚焦上市公司投资者问答场景，目标是将原有研究型模型能力升级为可演示、可复核、可迭代的数据产品原型。  
核心方向是围绕**金融问答**文本，完成**财务实体识别**与**逃避战术检测**，并通过**低置信度复核**形成**数据飞轮**。

### 核心问题
- 问答文本中既有业务事实，也有大量模板化/策略性回复，纯规则难以稳定识别。
- 二层细分类（尤其逃避战术细类）存在长尾问题，单次训练难以覆盖。
- 仅有推理接口不足以支持持续迭代，需要将“模型判断 -> 复核 -> 回流训练集”闭环化。

### 当前解决方案
- 在线推理：`FastAPI` 提供单条/批量接口，输出两层分类、概率分布、实体命中信息。
- 复核闭环：将低置信度样本自动入队，支持人工触发 `Dify Agent 辅助复核`，人工确认后回流训练语料。
- 产品化升级：新增 `React + TypeScript` 前端工作台，覆盖分析页、批任务页、复核页。

## 核心能力
### 问答对预处理 / 金融相关过滤
- [已完成] 离线数据处理脚本链路（`src/00` ~ `src/02`）用于数据清洗、实体相关过滤、信息熵筛选。
- [已完成] 在线推理前置实体门卫：基于金融词典（`THUOCL_caijing`、`baostock_entities`、`custom_entities`）与最小长度规则拦截低相关样本。
- [进行中] 词典扩充流程已提供工具（`utils/discover_candidate_terms.py`、`utils/review_candidate_terms.py`），以“自动发现 + 人工审核”方式持续补词。

### 两层分类/识别能力
- [已完成] 第一层分类：`Direct / Intermediate / Evasive`。
- [已完成] 第二层分支：
  - `Direct` 分支：财务业务子类识别（5类）。
  - `Evasive` 分支：逃避战术子类识别（4类）。
  - `Intermediate` 分支：固定为“无下游细分（部分响应）”。
- [已完成] 返回可解释字段：`root_probabilities`、`sub_probabilities`、`entity_hits`、`warning`。

### 推理服务
- [已完成] `POST /infer` 与兼容接口 `POST /api/evaluate`。
- [已完成] `POST /batch_infer` + `GET /jobs/{id}` + `GET /jobs/{id}/export`（CSV导出）。
- [进行中] 批任务状态当前使用内存任务仓库（`InMemoryJobStore`），服务重启后不可恢复。

### 数据飞轮闭环
- [已完成] 低置信度判定（根节点阈值、子节点阈值、长尾标签）与自动入复核队列。
- [已完成] 人工复核提交后写入 `data/processed/review_training_corpus.csv`，支持样本回流。
- [进行中] 自动化再训练/自动发布尚未打通，当前仍以离线训练脚本为主。

### Agent 辅助复核
- [已完成] 人工触发 Agent 建议：`POST /review/{sample_id}/agent-suggestion`，异步任务由 `Celery` 执行。
- [已完成] 支持 Dify 输出解析、字段映射、超时兜底、无效响应兜底。
- [进行中] 真实业务工作流 Prompt 与输出字段仍需按具体 Dify 工作流持续联调优化。

## 当前实现进度
### 已完成
- [已完成] 本地模型推理服务（两层分类 + 概率分布 + 实体命中）。
- [已完成] 单条分析、批量分析、任务轮询、CSV导出全链路接口。
- [已完成] 低置信度自动入队、手动入队、复核详情查询。
- [已完成] Dify Agent 异步建议任务链路（API -> Celery -> DB 回写）。
- [已完成] 人工确认/修正并回流训练语料。
- [已完成] React 前端三页原型（分析、批任务、复核）与基础单测。
- [已完成] Docker Compose 编排（`api + worker + redis + postgres`）。

### 进行中
- [进行中] 批量任务持久化（当前为内存态，生产韧性有限）。
- [进行中] 前端仍属产品化原型阶段，尚未接入权限体系与完整审计视图。
- [进行中] 数据飞轮“回流 -> 自动训练 -> 自动发布”仍为半自动流程。
- [进行中] Dify 工作流输出协议在不同应用下需要继续对齐。

### 下一步计划
- [规划中] 批任务改造为持久化任务队列（支持重启恢复与更细粒度监控）。
- [规划中] 增量训练流水线与模型版本管理自动化。
- [规划中] 复核审计看板、角色权限、操作留痕增强。

## 系统架构概览
系统由四个层次组成：
- 在线推理层：`InferenceService` 负责实体门卫、根节点分类、子节点分类与解释性输出。
- API 与任务层：`FastAPI` 提供业务接口；批量推理通过 `BackgroundTasks` 运行；Agent 建议通过 `Celery + Redis` 异步执行。
- 数据闭环层：`ReviewService` 将推理结果落库，执行低置信度入队、复核状态流转、训练语料回流。
- 前端工作台层：`React + Ant Design` 提供分析、批任务、复核三类页面，调用统一后端 API。

## 系统界面浏览
<img width="1898" height="2288" alt="首页分析" src="https://github.com/user-attachments/assets/6d78dbbe-e748-4d7f-a4cf-1baf84469bdc" />
<img width="1896" height="1645" alt="批量任务" src="https://github.com/user-attachments/assets/29d0e7b8-e2cc-4a3c-8c00-aedf2bffc342" />
<img width="1912" height="956" alt="人工复核" src="https://github.com/user-attachments/assets/5a9643e4-5ec6-407f-ad91-c8175de94f5b" />



## 项目目录结构
```text
.
├─ app/
│  ├─ api/                    # /infer /batch_infer /jobs /review /annotate
│  ├─ core/                   # 配置、数据库、Celery
│  ├─ models/                 # Pydantic 模型 + SQLAlchemy ORM
│  ├─ services/               # 推理、复核、Agent、批处理服务
│  ├─ tasks/                  # Celery 异步任务
│  ├─ main.py                 # FastAPI 应用入口
│  ├─ index.html              # 旧版静态分析页（兼容保留）
│  └─ review.html             # 旧版静态复核页（兼容保留）
├─ frontend/                  # React + TS 前端
│  └─ src/
│     ├─ pages/               # Analysis / BatchTasks / Review 页面
│     ├─ components/          # EntityHighlight / ProbabilityRadar
│     └─ api/client.ts        # 前端 API 封装
├─ src/                       # 离线训练与数据处理脚本（00~09）
├─ utils/                     # 词条发现、词条复核、评估与数据增强工具
├─ data/
│  ├─ raw/                    # 原始问答数据
│  ├─ processed/              # 中间产物、训练集、stage_c.db、回流语料
│  └─ others/                 # 金融词典、停用词、候选词文件
├─ tests/                     # 后端与数据工具测试
├─ docker-compose.yml
├─ Dockerfile
└─ requirements.txt
```

## 快速开始
### 环境要求
- Python `3.10+`
- Node.js `20+`（前端 Vite 8）
- Redis（Agent 异步任务需要）
- PostgreSQL（推荐；也支持 SQLite 回退）

### 安装依赖
```bash
pip install -r requirements.txt
npm install
```

### 配置 `.env`
```bash
cp .env.example .env
```

最小建议配置（按本地环境修改）：
- 模型目录：`QA_ROOT_MODEL_DIR`、`QA_DIRECT_MODEL_DIR`、`QA_EVASIVE_MODEL_DIR`
- 数据库：`QA_DATABASE_URL`
- 任务队列：`QA_CELERY_BROKER_URL`、`QA_CELERY_RESULT_BACKEND`
- 低置信度阈值：`QA_REVIEW_ROOT_THRESHOLD`、`QA_REVIEW_SUB_THRESHOLD`
- Dify：`QA_DIFY_API_URL`、`QA_DIFY_API_KEY`、`QA_DIFY_OUTPUT_PATH`

### 启动 API / Worker / 前端
```bash
# 1) 启动后端 API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2) 启动 Celery Worker（用于 Agent 建议任务）
celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1
# Linux / Docker 可使用：
# celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=prefork --concurrency=2

# 3) 启动前端
npm run dev
```

访问：
- 前端开发页：`http://127.0.0.1:5173`
- 健康检查：`http://127.0.0.1:8000/api/health`
- 说明：当未检测到可用 Celery worker 时，`/review/{sample_id}/agent-suggestion` 会自动回退为 API 进程内执行，避免任务长期处于 `pending`。

若使用 Celery Worker，请确保 Redis 可用，否则会出现
`Cannot connect to redis://localhost:6379/0 (10061)`：
```bash
# 方式1：Docker（推荐）
docker compose up -d redis

# 方式2：本地调试不启Worker（仅调接口）
# .env 中设置 QA_CELERY_TASK_ALWAYS_EAGER=true
```

说明：
- [已完成] React 前端已可用（分析/批量/复核）。
- [已完成] 旧静态页 `app/index.html`、`app/review.html` 仍可联调使用。
- [进行中] 前端仍是原型形态，非完整商用后台。

### Docker Compose（推荐演示方式）
```bash
docker compose up --build -d postgres redis api worker
docker compose ps
```

## 关键接口说明
| 方法 | 路径 | 说明 | 当前状态 |
|---|---|---|---|
| `POST` | `/infer` | 单条推理（主接口） | [已完成] |
| `POST` | `/api/evaluate` | 兼容旧前端的单条推理接口 | [已完成] |
| `POST` | `/batch_infer` | 提交批量推理任务（上限由 `QA_MAX_BATCH_ITEMS` 控制） | [已完成] |
| `GET` | `/jobs/{id}` | 查询任务状态（批任务 + Agent任务） | [已完成] |
| `GET` | `/jobs/{id}/export` | 导出批任务结果 CSV | [已完成] |
| `GET` | `/review/queue` | 获取低置信度复核队列（分页/状态/时间过滤） | [已完成] |
| `GET` | `/review/{sample_id}` | 获取样本详情（模型输出/Agent建议/人工结果） | [已完成] |
| `POST` | `/review/{sample_id}/enqueue` | 手动加入复核队列 | [已完成] |
| `POST` | `/review/{sample_id}/agent-suggestion` | 触发 Dify Agent 辅助复核任务 | [已完成] |
| `POST` | `/annotate` | 人工确认/修正并回流训练语料 | [已完成] |
| `GET` | `/api/health` | 服务健康检查 | [已完成] |

## 数据飞轮说明
1. 模型推理入库：`/infer` 或 `/batch_infer` 执行后，样本与模型结果写入 `qa_samples`、`model_outputs`，并记录 `model` 来源标注。
2. 低置信度入队：当根节点/子节点置信度低于阈值，或命中长尾标签时，`review_status` 置为 `pending_review`。
3. 复核队列消费：人工在 `/review/queue` 查看样本，进入 `/review/{sample_id}` 查看模型细节。
4. 请求 Agent 建议：人工触发 `/review/{sample_id}/agent-suggestion`，任务异步执行，结果回写为 `agent` 标注并更新状态为 `agent_suggested`。
5. 人工最终确认：通过 `/annotate` 提交最终标签，状态变更为 `confirmed` 或 `revised`。
6. 训练集回流：复核结果写入 `data/processed/review_training_corpus.csv`，用于后续离线训练迭代。

## Dify Agent 接入说明
### 环境变量
- `QA_DIFY_API_URL`：Dify 工作流调用地址（如 `/v1/workflows/run`）
- `QA_DIFY_API_KEY`：Dify API Key
- `QA_DIFY_USER`：调用 user 字段
- `QA_DIFY_OUTPUT_PATH`：自定义结果字段路径（如 `data.outputs.result`）
- `QA_DIFY_MODEL_RESULT_AS_JSON`：是否将 `model_result` 以 JSON 字符串传入
- `QA_DIFY_CONNECT_TIMEOUT` / `QA_DIFY_WRITE_TIMEOUT` / `QA_DIFY_READ_TIMEOUT`：超时控制

### 当前接入方式
- 后端在 `AgentService` 中通过 `httpx` 调用 Dify，统一解析输出并归一化 `root_label / sub_label / confidence / reason`。
- 通过 `Celery` 异步执行，避免阻塞主 API 请求。
- 内置兜底策略：
  - 未配置 Dify 时回退为模型结论建议。
  - 超时/结构异常时返回可审计的 fallback 结果并保留原始响应信息。

### 当前阶段
- [已完成] 后端调用链路、字段归一化、超时与异常兜底、任务状态轮询接口。
- [进行中] 真实业务 Dify 工作流的 prompt 细化与稳定性联调（不同工作流输出结构需按项目配置）。

## 技术栈
### 后端
- FastAPI
- SQLAlchemy
- Celery
- Redis
- PostgreSQL（可回退 SQLite）

### 前端
- React 19
- TypeScript
- Vite 8
- Ant Design
- TanStack React Query
- ECharts

### 模型/训练
- PyTorch
- Hugging Face Transformers
- FlashText
- Jieba
- Pandas / Scikit-learn

### 数据库/任务队列
- PostgreSQL（结构化数据）
- SQLite（开发回退）
- Redis + Celery（异步任务）
- FastAPI BackgroundTasks（当前批任务执行）

## 适用场景 / 项目价值
- 监管与合规团队对上市公司问答文本的快速筛查。
- 金融文本质检场景中的“逃避表述识别 + 复核回流”流程验证。
- 作品集展示“从研究型模型到产品化升级”的完整工程能力：模型推理、接口设计、前端交互、低置信度复核、数据飞轮。

## Roadmap
- [已完成] 两层分类推理服务、批量任务、复核闭环、Dify Agent 辅助复核、前端工作台原型。
- [进行中] 批任务持久化、Dify 工作流联调稳定性、复核台产品化细节。
- [规划中] 自动化再训练与模型发布、权限与审计体系、更完善的观测与运维指标。

## License
当前仓库未提供 `LICENSE` 文件。  
如需对外开源发布，建议补充明确的开源协议（如 MIT/Apache-2.0）并在根目录新增 `LICENSE`。
