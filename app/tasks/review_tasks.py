'''复核异步任务：处理 Agent 建议生成与任务状态更新。'''

from __future__ import annotations

import logging

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.services.agent_service import AgentService
from app.services.review_service import ReviewService

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.review_tasks.generate_agent_suggestion_task")
def generate_agent_suggestion_task(job_id: str, sample_id: str) -> dict:
    settings = get_settings()
    review_service = ReviewService(settings)
    agent_service = AgentService()

    try:
        review_service.mark_job_started(job_id)
        detail = review_service.get_review_detail(sample_id)
        suggestion = agent_service.suggest(
            question=detail["question_text"],
            answer=detail["answer_text"],
            model_result={
                "root_label": detail["model_output"]["layer1_label"],
                "sub_label": detail["model_output"]["layer2_json"].get("sub_label"),
                "root_confidence": detail["model_output"]["layer1_confidence"],
            },
        )
        review_service.finish_agent_job(job_id=job_id, sample_id=sample_id, suggestion=suggestion)
        return {"job_id": job_id, "status": "completed"}
    except Exception as exc:
        # 二次更新状态失败不应覆盖原始业务异常，避免排障信息丢失。
        try:
            review_service.fail_job(job_id, str(exc))
        except Exception:  # pragma: no cover - 仅在数据库异常等极端路径触发
            logger.exception("Agent任务失败后写回job状态失败: job_id=%s", job_id)
        raise
