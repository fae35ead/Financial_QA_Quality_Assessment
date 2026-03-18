'''复核接口路由：提供待复核队列、样本详情、Agent建议触发能力。'''

from __future__ import annotations

from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.celery_app import has_live_workers
from app.models.schemas import (
    AgentSuggestionJobResponse,
    ManualReviewEnqueueResponse,
    ReviewDetailResponse,
    ReviewQueueItem,
    ReviewQueueResponse,
)
from app.tasks.review_tasks import generate_agent_suggestion_task

router = APIRouter(tags=["review"])
logger = logging.getLogger(__name__)


@router.get("/review/queue", response_model=ReviewQueueResponse)
def review_queue(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    status: str = Query(default="pending_review,agent_suggested"),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
):
    service = request.app.state.review_service
    statuses = [x.strip() for x in status.split(",") if x.strip()]
    data = service.list_queue(page=page, page_size=page_size, statuses=statuses, date_from=date_from, date_to=date_to)
    return ReviewQueueResponse(
        page=page,
        page_size=page_size,
        total=data["total"],
        items=[ReviewQueueItem(**item) for item in data["items"]],
    )


@router.get("/review/{sample_id}", response_model=ReviewDetailResponse)
def review_detail(sample_id: str, request: Request):
    service = request.app.state.review_service
    try:
        data = service.get_review_detail(sample_id)
        return ReviewDetailResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/review/{sample_id}/agent-suggestion", response_model=AgentSuggestionJobResponse)
def request_agent_suggestion(sample_id: str, request: Request):
    service = request.app.state.review_service
    try:
        job_id = service.create_agent_job(sample_id=sample_id, requester_id="human_reviewer")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        if has_live_workers():
            generate_agent_suggestion_task.delay(job_id, sample_id)
        else:
            # 无可用 worker 时兜底为进程内执行，避免任务长期 pending。
            logger.warning("未检测到可用Celery worker，回退为进程内执行Agent任务: job_id=%s", job_id)
            generate_agent_suggestion_task.apply(args=(job_id, sample_id), throw=True)
    except Exception as exc:
        service.fail_job(job_id, f"Agent任务投递失败: {exc}")
        raise HTTPException(status_code=503, detail=f"Agent任务投递失败: {exc}") from exc

    return AgentSuggestionJobResponse(job_id=job_id, status="pending", sample_id=sample_id)


@router.post("/review/{sample_id}/enqueue", response_model=ManualReviewEnqueueResponse)
def enqueue_review(sample_id: str, request: Request):
    service = request.app.state.review_service
    try:
        result = service.enqueue_manual_review(sample_id=sample_id, requester_id="analysis_user")
        return ManualReviewEnqueueResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
