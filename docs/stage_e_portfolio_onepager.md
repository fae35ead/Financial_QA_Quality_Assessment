# 阶段E作品集一页稿：金融问答复核闭环系统

## 1. 业务背景
- 面向金融问答场景，识别董秘是否正面回答。
- 若正面回答，提取财务实体；若逃避回答，识别逃避战术。
- 对低置信度与疑似误判样本进入复核链路，形成数据飞轮。

## 2. 核心问题
- 人工逐条筛查成本高、时效性差。
- 传统开发态部署不稳定，难以支撑演示与复用。
- 异步任务成功/失败路径缺乏系统化验证。

## 3. 阶段E方案
- 编排统一：`api + worker + redis + postgres`（Docker Compose）。
- 后端镜像统一：`api/worker` 复用单一 Dockerfile，去除运行时安装依赖。
- 可靠性增强：
  - `postgres/redis/api` 健康检查
  - `depends_on: service_healthy` 启动依赖闭环
  - `QA_DATABASE_FALLBACK_TO_SQLITE=false`（Compose 场景）
- 业务接口保持兼容：不改核心业务 API 契约。

## 4. 关键接口与链路
- 分析：`POST /api/evaluate`、`POST /infer`
- 复核队列：`GET /review/queue`、`POST /review/{sample_id}/enqueue`
- Agent 任务：`POST /review/{sample_id}/agent-suggestion`、`GET /jobs/{job_id}`
- 探活：`GET /api/health`

## 5. 测试与可靠性
- 新增阶段E关键测试（10项）覆盖：
  - Compose 合约（服务、命令、健康检查、依赖、fallback 禁用）
  - Celery 成功/失败路径
  - 持久化任务状态回读
  - 手动入队幂等性
  - fallback 关闭时 DB 故障 fail-fast

## 6. 指标呈现
- 模型指标数据源：`notebooks/evaluation/student_compare/model_summary.csv`
- 工程指标口径：
  - 启动链路可用率（四服务健康）
  - 队列闭环完成率（自动/手动入队 -> 复核 -> 回流）
  - 异步任务成功率与失败可观测性
  - 导出能力与回读一致性

## 7. 数据飞轮价值
- 低置信度样本不再“丢弃”，而是进入复核并沉淀为训练资产。
- 通过 Agent 建议 + 人工确认，提高复核效率与标签质量。
- 支持持续迭代，逐步提升长尾子标签表现与系统稳健性。

## 8. 迭代方向
- 自动化评估与增量训练流水线。
- 任务SLA与错误分类监控。
- 权限、审计、合规报表产品化。
