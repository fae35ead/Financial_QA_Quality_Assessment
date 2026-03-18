# 金融问答质量评估系统
## 项目亮点
- 覆盖能力：支持 Root 三分类 `Direct / Intermediate / Evasive`，以及子类细分（`Direct` 5 类 + `Evasive` 4 类）。
- 关键指标：教师模型宏 F1 = `0.68`；学生根节点宏 F1 = `0.77`；财务方向子节点宏 F1 = `0.63`；逃避战术子节点宏 F1 = `0.66`。
- 生产链路：提供 `/infer`、`/batch_infer`、复核队列、CSV 导出接口（详见接口表与 API 示例）。
- 数据闭环：低置信度自动入队并回流训练集；支持 Dify Agent 辅助复核。
- 目前已完成：本地推理服务 + 前端原型 + Docker Compose 编排（见“当前进度”）。

## 项目简介
本项目面向上市公司投资者问答场景，核心目标是识别董秘答复是否正面回应，并在此基础上实现：
- 正面回答时识别财务实体与业务方向。
- 逃避回答时识别逃避战术类型。
- 对低置信度样本触发复核闭环（自动入队 -> Agent 建议 -> 人工确认 -> 训练集回流）。

## 项目界面
<img width="1898" height="2288" alt="首页分析" src="https://github.com/user-attachments/assets/7990dc49-c858-4c27-8812-af84fa48cb4f" />
<img width="1896" height="1645" alt="批量任务" src="https://github.com/user-attachments/assets/a626fd16-9a14-41a4-9ccf-9d8228023bcd" />
<img width="1912" height="956" alt="人工复核" src="https://github.com/user-attachments/assets/63d11a7b-b3ed-4583-90c3-653d577cd674" />

---
## 如何运行
### 环境准备
根据项目根目录下的 `requirements.txt` 安装 Python 依赖，建议使用虚拟环境：

```powershell
# 1) 先启动 Redis（若未启动）
docker compose up -d redis

# 2) 启动后端 API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 3) 启动 Worker（Windows 本地）
celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1
```


## 快速开始（Minimal Example）
### 一段命令跑通（PowerShell）
> 说明：该最小链路默认使用 SQLite，且将 Celery 设为 eager（无需单独启动 Redis/Worker），便于面试官快速体验接口能力。

```powershell
cd F:\软件\学习相关\PycharmProjects\QA
Copy-Item .env.example .env -Force
python -m pip install -r requirements.txt
$env:QA_DATABASE_URL = "sqlite:///./data/processed/stage_c.db"
$env:QA_DATABASE_FALLBACK_TO_SQLITE = "true"
$env:QA_CELERY_TASK_ALWAYS_EAGER = "true"
$env:QA_DIFY_API_URL = ""
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
# 1) 先启动 Redis（若未启动）
docker compose up -d redis

# 2) 启动后端 API
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 3) 启动 Worker（Windows 本地）
celery -A app.core.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1
```

### 快速 Demo 请求
新开一个 PowerShell 窗口执行：

```powershell
Invoke-RestMethod -Method POST "http://127.0.0.1:8000/infer" -ContentType "application/json" -Body '{"company_name":"测试公司","question":"湖南基地何时投产？","answer":"公司湖南基地项目正在稳步推进，请关注后续定期报告。"}'
```

### 最小运行必需环境变量（建议默认）
| 变量名 | 推荐默认值 | 作用 |
|---|---|---|
| `QA_DATABASE_URL` | `sqlite:///./data/processed/stage_c.db` | 本地最小可运行数据库 |
| `QA_DATABASE_FALLBACK_TO_SQLITE` | `true` | 当 Postgres 不可达时自动回退 |
| `QA_CELERY_TASK_ALWAYS_EAGER` | `true`（最小 Demo） | 不启 Worker 也可执行 Agent 任务 |
| `QA_DIFY_API_URL` | 空字符串（最小 Demo） | 为空时走本地 fallback，不外部调用 |
| `QA_ROOT_MODEL_DIR` | `models/student_100k_T4_distilbert` | 根节点模型路径 |
| `QA_DIRECT_MODEL_DIR` | `models/lcppn_direct_classifier` | Direct 子分类模型路径 |
| `QA_EVASIVE_MODEL_DIR` | `models/lcppn_evasive_classifier` | Evasive 子分类模型路径 |

---

## 如何评价实验结果
在本项目这种多分类且类别分布不均衡的任务中，`Macro-F1` 比准确率更能反映模型在少数类上的表现。

### 用哪些指标评估
- 主指标：`Macro-F1`（根节点与子节点任务都使用）。
- 辅助指标：`Precision`、`Recall`、混淆矩阵（用于定位错分模式）。
- 线上闭环指标（工程侧）：低置信度入队率、复核通过率、人工修订率、回流样本量。

### 当前离线结果
| 模型/任务 | 宏 F1 |
|---|---|
| 教师模型 | `0.68` |
| 学生模型（根节点） | `0.77` |
| 子节点（财务方向） | `0.63` |
| 子节点（逃避战术） | `0.66` |

### 测试数据划分
- 教师模型：训练集 : 验证集 = `850 : 150`
- 学生模型根节点：训练集 : 验证集 = `90000 : 10000`
- 子节点（财务方向）：训练集 : 验证集 = `329 + 80 : 59`
- 子节点（逃避战术）：训练集 : 验证集 = `294 + 194 : 52`

### 样本大小与标注流程
- 样本规模：如上划分所示。
- 标注流程：教师模型与学生模型子节点分类器均采用 LLM 标注流程。

### 指标定义与评测口径
- 指标定义口径基于 EvasionBench 思路。
- 标签与提示词规范可参考 `prompt/prompt.txt` 与 `prompt/dify_agent.txt`。

### 低置信度阈值
- 当前阈值设为 `0.65`（根/子节点同口径，可通过环境变量调整）。
- 设定依据：在验证集置信度分布与人工复核容量之间做折中，目标是提升疑难样本召回，同时控制复核队列规模。
- 工程实践建议：按业务周期定期重扫阈值（如 `0.50~0.80` 网格），以“复核质量收益/队列成本”联合最优为准。

---

## 标注协议
### 数据来源与合规
- 数据来自公开、合规渠道，为互动e平台的公开数据。

### 敏感信息处理
- 不保留个人身份敏感字段。

### 标注说明
- 标签定义：Root 三类 + 子类（财务方向 5 类、逃避战术 4 类）。
- 标注示例与指引：见 `prompt/prompt.txt`。
- 标注工具：LLM 标注流程 + 人工复核确认。

---

## API Reference
### 接口总览
| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/infer` | 单条推理（主接口） |
| `POST` | `/api/evaluate` | 旧前端兼容接口 |
| `POST` | `/batch_infer` | 批量推理任务提交 |
| `GET` | `/jobs/{job_id}` | 查询批任务/Agent任务状态 |
| `GET` | `/jobs/{job_id}/export` | 导出批任务 CSV |
| `GET` | `/review/queue` | 复核队列分页查询 |
| `GET` | `/review/{sample_id}` | 复核详情（下文 id 即 sample_id） |
| `POST` | `/review/{sample_id}/agent-suggestion` | 请求 Agent 建议 |
| `POST` | `/review/{sample_id}/enqueue` | 手动加入复核队列 |
| `POST` | `/annotate` | 人工确认并回流训练集 |

### 1) `POST /infer`
请求示例：
```json
{
  "company_name": "测试公司",
  "qa_time": "2026-03-18T10:30:00",
  "question": "湖南基地何时投产？",
  "answer": "公司湖南基地项目正在稳步推进，请关注后续定期报告。"
}
```

响应示例：
```json
{
  "root_id": 2,
  "root_label": "Evasive (打太极)",
  "root_confidence": 81.23,
  "sub_label": "推迟回答",
  "sub_confidence": 72.11,
  "root_probabilities": {
    "Direct (直接响应)": 0.08,
    "Intermediate (避重就轻)": 0.11,
    "Evasive (打太极)": 0.81
  },
  "sub_probabilities": {
    "推迟回答": 0.72,
    "转移话题": 0.13,
    "战略性模糊": 0.10,
    "外部归因": 0.05
  },
  "entity_hits": [
    {"text": "投产", "start": 7, "end": 9, "source_text": "question"}
  ],
  "warning": "根节点置信度过低，建议人工复核",
  "sample_id": "b2d9...",
  "is_low_confidence": true,
  "review_status": "pending_review"
}
```

字段解释：
- `root_probabilities`：根节点每个候选类别的概率分布（0~1）。
- `sub_probabilities`：子节点每个候选类别的概率分布（0~1）。
- `warning`：在实体门卫触发或根节点低置信度时出现，否则为 `null`。
- `is_low_confidence`：是否命中低置信度规则（用于自动入队）。

### 2) `POST /batch_infer` + `GET /jobs/{job_id}`
请求示例：
```json
{
  "items": [
    {
      "sample_id": "s1",
      "company_name": "测试公司A",
      "question": "今年分红是否提升？",
      "answer": "请关注公司后续公告。"
    },
    {
      "sample_id": "s2",
      "company_name": "测试公司B",
      "question": "新产线预计何时量产？",
      "answer": "预计三季度投产。"
    }
  ]
}
```

提交响应：
```json
{
  "job_id": "9f6e...",
  "status": "pending",
  "total": 2
}
```

查询响应（节选）：
```json
{
  "job_id": "9f6e...",
  "status": "completed",
  "total": 2,
  "completed": 2,
  "failed": 0,
  "progress": 100.0,
  "results": [
    {
      "index": 0,
      "sample_id": "s1",
      "status": "completed",
      "result": {
        "root_label": "Evasive (打太极)",
        "sub_label": "推迟回答"
      }
    }
  ]
}
```

### 3) `GET /review/{sample_id}`
响应示例：
```json
{
  "sample_id": "b2d9...",
  "company_name": "测试公司",
  "question_text": "湖南基地何时投产？",
  "answer_text": "公司湖南基地项目正在稳步推进，请关注后续定期报告。",
  "model_output": {
    "layer1_label": "Evasive (打太极)",
    "layer1_confidence": 81.23,
    "layer2_json": {
      "sub_label": "推迟回答",
      "sub_confidence": 72.11
    }
  },
  "agent_suggestion": {
    "root_label": "Evasive (打太极)",
    "sub_label": "推迟回答",
    "confidence": 0.95,
    "reason": "答复未直接回应投产进度，且引导关注后续定期报告。"
  },
  "human_annotation": null
}
```

---

## 数据飞轮闭环
1. 模型推理：`/infer` 或 `/batch_infer` 产出预测并落库。
2. 自动入队：低置信度样本进入 `pending_review` 队列。
3. Agent 辅助：`/review/{sample_id}/agent-suggestion` 生成建议。
4. 人工确认：通过 `/annotate` 进行最终标签确认或修订。
5. 回流训练：写入 `data/processed/review_training_corpus.csv`，用于下一轮训练迭代。

---

## 当前进度
### 已完成
- 本地推理服务（Root + 子类 + 概率分布 + 实体命中）。
- 批量任务、任务查询、CSV 导出。
- 低置信度自动入队 + 手动入队 + 复核详情。
- Dify Agent 建议链路（含超时与 fallback）。
- React 前端原型（分析页/批任务页/复核页）。
- Docker Compose 编排（`api + worker + redis + postgres`）。

### 进行中
- 批量任务持久化增强（当前 `InMemoryJobStore` 在服务重启后不可恢复）。
- 复核台产品化细节（权限、审计、运营看板）。
- 自动化再训练与版本发布流程。

---

## 技术栈
- 后端：FastAPI、SQLAlchemy、Celery、Redis、PostgreSQL/SQLite
- 前端：React 19、TypeScript、Vite 8、Ant Design
- 模型：PyTorch、Transformers、FlashText、Jieba
- 数据：Pandas、Scikit-learn

## Docker Compose（生产链路演示）
```bash
docker compose up --build -d postgres redis api worker
docker compose ps
```

## License
当前仓库未提供 `LICENSE` 文件。
