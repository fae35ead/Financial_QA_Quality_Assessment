# ARCHITECHTURE：A股问答对智能分析与监管合规轻量化平台

## 1. 架构设计目标

本项目的架构设计遵循三个核心原则：

1. **主流**：优先选择社区成熟、资料丰富的技术，便于快速实现与排查问题
2. **成熟**：优先选择工程工具链完善、可维护性强的方案
3. **合适**：架构要“刚刚好”，满足当前需求并保留适度扩展空间，避免过度设计

该项目本质上是一个 **轻量级、可私有化部署的 A 股问答分析产品原型**，目标不是构建超大规模平台，而是围绕：

- 模型推理服务
- 批量问答分析
- 低置信度样本复核
- 数据飞轮
- 产品化展示

构建一套清晰、可维护、可演示的系统架构。

---

## 2. 技术栈选择及原因

## 2.1 前端技术栈

### React + TypeScript
**选择原因：**
- React 是主流前端框架，社区活跃，资料多
- TypeScript 更适合中后台和表格型产品，类型更清晰
- 适合构建分析页、列表页、复核页、管理页

### Ant Design
**选择原因：**
- 企业中后台场景成熟
- 表格、表单、弹窗、上传组件齐全
- 适合快速搭建监管分析后台

### ECharts
**选择原因：**
- 对雷达图、趋势图、分布图支持完善
- 适合展示逃避战术概率分布与风险概览

### React Query / Zustand
**选择原因：**
- React Query 适合管理请求缓存、异步状态
- Zustand 轻量，适合局部状态管理
- 比 Redux 更适合本项目体量

---

## 2.2 后端技术栈

### Python + FastAPI
**选择原因：**
- 项目已有 FastAPI 原型，迁移成本低
- Python 与现有模型训练与推理代码天然兼容
- FastAPI 文档能力强，适合快速开发 API
- 社区成熟，适合 AI 服务场景

### Celery + Redis
**选择原因：**
- 批量 Excel 分析需要异步任务
- Celery 是 Python 生态成熟方案
- Redis 可作为消息中间件与缓存层

### PostgreSQL
**选择原因：**
- 结构化数据建模能力强
- 适合存储样本、模型输出、标注结果、审计记录
- 支持 JSONB，适合保存第二层复杂结构输出
- 开源、稳定、私有化友好

### Redis
**选择原因：**
- 用于缓存热点结果
- 用于异步任务队列
- 用于待复核样本队列或状态缓存

### MinIO
**选择原因：**
- 兼容 S3，适合私有化部署
- 适合保存批量导出文件、模型权重、审计包
- 比直接依赖云厂商对象存储更适合本项目场景

---

## 2.3 模型与训练技术栈

### PyTorch
**选择原因：**
- 现有项目已使用
- 是 NLP 训练和部署主流方案
- 生态成熟

### Hugging Face Transformers
**选择原因：**
- 与 DistilBERT、RoBERTa 等模型兼容
- 训练、推理、加载模型方便
- 社区成熟

### PEFT / LoRA
**选择原因：**
- 适合小成本微调
- 适合数据飞轮驱动的周期性迭代
- 比全量微调更轻便，适合 demo 与私有化环境

### MLflow
**选择原因：**
- 方便记录模型版本、参数、指标
- 便于展示模型迭代逻辑
- 适合后续扩展模型治理能力

---

## 2.4 运维与部署技术栈

### Docker + Docker Compose
**选择原因：**
- 轻量、上手快
- 适合本地和私有化单机部署
- 非常适合面试展示场景

### Kubernetes（可选）
**选择原因：**
- 适合后续扩展
- 不是 MVP 必需，但作为扩展方向合理

### Prometheus + Grafana
**选择原因：**
- 主流监控方案
- 可监控接口耗时、任务量、低置信度样本数

---

## 3. 总体架构设计

系统整体分为六层：

1. **展示层**：React 前端页面
2. **接口层**：FastAPI 提供业务 API
3. **业务服务层**：分析、批量任务、复核与导出逻辑
4. **模型服务层**：两层推理模型与结果封装
5. **数据层**：PostgreSQL、Redis、MinIO
6. **训练与迭代层**：低置信度样本回流、LoRA 微调、模型版本管理

### 架构特点
- 前后端分离，适合中后台产品形态
- 模型服务与业务服务解耦，便于后续扩展
- 数据飞轮作为独立闭环设计，不与主链路强耦合
- 保持轻量，避免一开始引入过重的中间件体系

---

## 4. 项目目录结构

```text
Financial_QA_Quality_Assessment/
├── frontend/                         # 阶段D React + TypeScript 前端工程（Vite）
│   ├── src/App.tsx                   # 三页面路由入口（分析页/批量任务页/待复核页）
│   ├── src/pages/                    # 页面级组件
│   ├── src/components/               # 复用组件（实体高亮/概率雷达图）
│   └── src/api/client.ts             # 前端 API 请求封装
│
├── app/                              # 应用层（阶段A+B+C 已落地）
│   ├── main.py                       # FastAPI 启动入口与应用装配
│   ├── api/                          # 接口层
│   │   ├── infer.py                  # 单条推理接口（含 /api/evaluate 兼容）
│   │   ├── batch.py                  # 批量分析任务创建接口
│   │   ├── jobs.py                   # 任务状态查询接口（含Agent异步任务）
│   │   ├── review.py                 # 待复核队列/详情/Agent建议触发
│   │   └── annotate.py               # 人工复核提交接口
│   ├── services/                     # 业务服务层
│   │   ├── inference_service.py      # 两层推理服务
│   │   ├── batch_service.py          # 批任务执行服务
│   │   ├── job_service.py            # 批量任务状态存储（内存版）
│   │   ├── review_service.py         # 低置信度入队、复核、回流
│   │   └── agent_service.py          # Dify 建议调用
│   ├── models/                       # Pydantic 请求/响应模型
│   │   ├── schemas.py
│   │   └── review_db.py              # 阶段C数据库ORM模型
│   ├── core/                         # 配置与基础设施
│   │   ├── config.py
│   │   ├── database.py               # SQLAlchemy数据库连接
│   │   └── celery_app.py             # Celery实例
│   ├── tasks/                        # Celery 任务
│   │   └── review_tasks.py
│   ├── index.html                    # 分析页面
│   └── review.html                   # 复核工作台页面
│
├── src/                              # 现有训练与推理脚本
│   ├── 00_Data_Preprocess.py
│   ├── 01_Entities_Filter.py
│   ├── 02_Entropy_Calculated.py
│   ├── 03_LLM_Labeling.py
│   ├── 04_Teacher_Model_Train.py
│   ├── 05_Teacher_Inference.py
│   ├── 06_Student_Model_Distillation.py
│   ├── 07_LLM_Subnode_Labeling.py
│   ├── 08_LCPPN_Subnodes_Train.py
│   └── 09_Inference_Pineline.py
│
├── utils/                            # 评估、抽样、数据增强与候选词发现脚本
├── data/                             # 数据
├── models/                           # 模型权重
├── requirements.txt
└── README.md
```

---

## 5. 核心模块说明

## 5.1 前端模块

### 分析页
- 单条输入
- 展示分类结果
- 实体高亮
- 雷达图展示（Direct/Intermediate/Evasive 均支持）

### 批量分析页
- Excel 上传
- 批量任务进度查看
- 结果下载
- 单次上限 500 条样本

### 复核页
- 低置信度样本列表
- 结果确认与修改
- 备注记录

### 管理页（后续）
- 模型版本
- 审计记录
- 数据导出

---

## 5.2 API 模块

### `/infer`
负责单条问答推理，返回两层分析结果。（已实现）

### `/api/evaluate`
与原型前端兼容的单条推理接口。（已实现）

### `/batch_infer`
负责批量任务创建。（已实现，当前使用内存任务队列 + BackgroundTasks）

### `/jobs/{job_id}`
返回批量任务状态与结果。（已实现，覆盖批量任务与 Agent 建议任务）

批任务结果 `results[].result` 已包含前端可解释字段：
- `root_probabilities`
- `sub_probabilities`
- `entity_hits`

### `/annotate`
保存人工复核结果。（已实现）

### `/review/queue`
分页获取待复核队列。（已实现）

### `/review/{sample_id}`
获取单样本复核详情：模型结论、Agent建议、人工记录。（已实现）

### `/review/{sample_id}/enqueue`
人工将当前样本手动加入待复核队列。（已实现）

### `/review/{sample_id}/agent-suggestion`
人工触发 Agent 建议生成（异步）。（已实现）

### `/jobs/{job_id}/export`
导出批量任务结果 CSV。（已实现）

---

## 5.3 业务服务模块

### inference_service
封装两层推理逻辑：
- 第一层逃避识别
- 第二层财务实体提取或逃避战术分类
- 统一输出结构

### batch_service
负责批量任务拆分、异步执行、结果汇总。（已实现）

### job_service
负责批量任务状态存储与进度管理。（已实现，当前为内存实现，后续迁移 Redis/PostgreSQL）

### review_service
负责低置信度判定（父/子阈值 0.65）、自动/手动入队、复核详情聚合、人工提交与训练集回流。（已实现）

### agent_service
负责 Dify 建议调用与建议结果规范化。（已实现）

### export_service
负责多格式导出（Excel、审计包等）扩展能力。（规划中）

---

## 5.4 模型服务模块

### 第一层分类模型
判断：
- 直接回答
- 逃避回答

### 第二层分流模型
- 直接回答 -> 财务实体提取
- 逃避回答 -> 逃避战术识别

### 模型包装层
对模型输出进行统一结构化封装，便于前端展示与后续复核。

当前推理响应已提供：
- 根节点全概率分布（`root_probabilities`）
- 子节点全概率分布（`sub_probabilities`）
- 实体命中坐标（`entity_hits`）

---

## 5.5 数据飞轮模块

### 低置信度样本识别
根据模型置信度阈值，把样本送入复核队列：
- 第一层置信度 < 0.65
- 第二层置信度 < 0.65

### 人工复核
由人工确认或修改结果。

### 手动入队补充机制
分析页可将当前问答对手动加入待复核队列，用于覆盖“高置信但疑似误判”场景。

### 可选 Agent 复核建议
仅对低置信度样本给建议，且由人工点击触发，不进入默认主流程。

### 数据回流
把高质量复核结果写入训练集。

### 词库候选词发现（离线）
通过 `utils/discover_candidate_terms.py` 从历史问答语料抽取高频 n-gram，
并结合 TF-IDF 排序产出候选词清单。

### 词库人工审核与回流（离线）
通过 `utils/review_candidate_terms.py` 对候选词逐条做 `y / n / s` 审核，
输出本次审核快照（`custom_entities.reviewed.txt` 与 `custom_stopwords.reviewed.txt`），
并将新增词增量写入 `custom_entities.txt` 与 `custom_stopwords.txt`。

> 当前阶段明确不做 Agent 全自动复核与自动入库能力。

### 周期性微调
使用 LoRA 等方式做持续迭代。

---

## 6. 数据模型设计

## 6.1 核心设计目标
数据模型既要支持当前产品功能，也要支持后续扩展。
重点支撑以下能力：
- 问答样本存储
- 模型输出存储
- 低置信度队列管理
- 人工复核记录
- 审计与导出
- 训练集回流

---

## 6.2 表设计

### 表 1：companies
存储上市公司基础信息。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| name | TEXT | 公司名称 |
| ticker | TEXT | 股票代码 |
| industry | TEXT | 行业 |

---

### 表 2：qa_samples
存储问答样本原始内容。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| company_id | UUID | 公司 ID |
| qa_time | TIMESTAMP | 问答时间 |
| question_text | TEXT | 问题文本 |
| answer_text | TEXT | 回答文本 |
| raw_source | JSONB | 原始来源信息 |
| created_at | TIMESTAMP | 创建时间 |

---

### 表 3：model_outputs
存储模型推理结果。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| sample_id | UUID | 样本 ID |
| model_version | TEXT | 模型版本 |
| layer1_label | TEXT | 第一层标签 |
| layer1_confidence | FLOAT | 第一层置信度 |
| layer2_json | JSONB | 第二层输出 |
| is_low_confidence | BOOLEAN | 是否低置信度 |
| processed_at | TIMESTAMP | 推理时间 |

#### `layer2_json` 内容
- 如果为直接回答：财务实体列表
- 如果为逃避回答：逃避战术列表与概率分布

---

### 表 4：annotations
存储人工或 Agent 复核结果。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| sample_id | UUID | 样本 ID |
| source | TEXT | 来源：human / agent / model |
| annotator_id | UUID | 标注人 |
| annotation | JSONB | 标注结果 |
| annotator_confidence | FLOAT | 复核置信度 |
| created_at | TIMESTAMP | 创建时间 |
| audit_trail | JSONB | 修改轨迹 |

---

### 表 5：jobs
存储批量任务与异步任务状态。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| job_type | TEXT | 任务类型 |
| params | JSONB | 任务参数 |
| status | TEXT | 状态 |
| progress | FLOAT | 进度 |
| created_at | TIMESTAMP | 创建时间 |
| finished_at | TIMESTAMP | 完成时间 |

---

### 表 6：audit_logs
存储审计日志。

| 字段 | 类型 | 说明 |
|---|---|---|
| id | UUID | 主键 |
| entity_type | TEXT | 实体类型 |
| entity_id | UUID | 实体 ID |
| user_id | UUID | 操作人 |
| action | TEXT | 操作类型 |
| old_value | JSONB | 修改前内容 |
| new_value | JSONB | 修改后内容 |
| timestamp | TIMESTAMP | 时间 |

---

## 6.3 关键索引建议
- `qa_samples(company_id, qa_time)`
- `model_outputs(sample_id)`
- `model_outputs(is_low_confidence)`
- `annotations(sample_id)`
- `jobs(status)`

这样可以支撑：
- 按公司与时间查询历史样本
- 快速拉取待复核队列
- 快速查看某个样本的全链路记录

---

## 7. 代码规范建议

## 7.1 Python 规范

### 基础规范
- Python 版本统一使用 3.10+
- 使用 `black` 统一格式化
- 使用 `isort` 统一 import 顺序
- 使用 `flake8` 或 `ruff` 检查风格
- 使用 `mypy` 做类型检查

### 命名规范
- 文件名：小写下划线，例如 `inference_service.py`
- 类名：大驼峰，例如 `InferenceService`
- 函数名：小写下划线，例如 `create_batch_job`
- 常量：全大写，例如 `LOW_CONF_THRESHOLD`

### 工程建议
- API 层不写复杂业务逻辑
- 业务逻辑集中在 `services/`
- 配置统一放在 `core/config.py`
- 数据模型统一放在 `models/`
- 日志统一封装，不直接 print

---

## 7.2 前端规范

### 技术规范
- 使用 TypeScript
- 使用 ESLint + Prettier
- 组件与页面分层清晰

### 目录规范
- `pages/`：页面级组件
- `components/`：复用组件
- `services/`：接口请求
- `hooks/`：通用逻辑
- `types/`：类型定义

### 交互规范
- 统一使用 Ant Design 风格体系
- 分析页、列表页、复核页保持一致布局
- 批量操作要有明确的状态反馈

---

## 7.3 Git 与协作规范

### 分支规范
- `main`：主分支
- `develop`：开发分支
- `feature/*`：功能分支
- `fix/*`：修复分支

### 提交规范
推荐使用：
- `feat:` 新功能
- `fix:` 修复
- `docs:` 文档
- `refactor:` 重构
- `chore:` 工程调整

---

## 7.4 测试规范

### 后端
- 使用 `pytest`
- 核心服务模块必须可单元测试
- 批量任务与推理接口至少覆盖基本用例

### 前端
- 使用 Vitest / React Testing Library
- 核心页面至少覆盖渲染和主要交互

---

## 8. 部署方式建议

## 8.1 MVP 部署方式
### Docker Compose
服务包括：
- frontend
- api
- worker
- postgres
- redis
- minio

**原因：**
- 最适合本项目当前体量
- 本地开发与演示友好
- 易于说明私有化部署能力

## 8.2 扩展部署方式
### Kubernetes
适合后续扩展：
- 多实例部署
- 模型服务拆分
- 更复杂的监控与扩容

当前不作为必须项。

---

## 9. 为什么这个架构是“刚刚好”

### 对当前需求刚刚好
- 已覆盖单条分析、批量分析、低置信度复核、数据飞轮等核心需求
- 保持了与现有项目代码的延续性
- 不要求一开始重做全部工程体系

### 对未来扩展有空间
- 可扩展更多页面
- 可扩展更多模型版本
- 可接入更完善的权限与审计系统
- 可从 Docker Compose 平滑过渡到 Kubernetes

### 没有过度设计
- 没有一开始引入复杂微服务拆分
- 没有上来就做复杂多租户
- 没有引入过重的云原生体系
- 架构成本与项目目标匹配

---

## 10. 总结

这套架构的目标不是追求“最复杂”或“最炫”的工程方案，而是服务于这个项目的真实目标：

- 在已有模型基础上完成产品化包装
- 展示对实际业务场景的理解
- 展示从 AI 模型到业务产品的完整思考路径
- 形成一个适合面试展示、可讲清楚、可跑起来、可扩展的项目原型

因此，这是一套围绕 **轻量、清晰、可维护、可扩展** 设计出来的产品架构方案。

---

## 11. 阶段E编排契约（Production Path）

阶段E默认演示链路仅包含四个服务：
- api
- worker
- redis
- postgres

关键契约：
- `api/worker` 复用同一后端镜像（统一 Dockerfile）。
- `api` 不使用 `--reload`，`worker` 使用固定 Celery 启动参数。
- `postgres/redis/api` 均需健康检查。
- `api/worker` 的 `depends_on` 统一使用 `service_healthy`。
- Compose 场景强制 `QA_DATABASE_FALLBACK_TO_SQLITE=false`，DB 故障应 fail-fast。
- `/api/health` 作为容器探活接口。

验收建议：
1. `docker compose up --build -d postgres redis api worker`
2. `docker compose ps` 确认四服务可用
3. `curl http://127.0.0.1:8000/api/health` 返回 `status=ok`
4. 复核链路验证：入队 -> Agent 建议 -> 任务轮询 -> 导出
