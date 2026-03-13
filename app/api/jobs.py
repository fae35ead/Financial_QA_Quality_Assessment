'''任务查询接口路由：根据 job_id 返回批量任务状态与结果。'''

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import JobResultItem, JobStatusResponse

router = APIRouter(tags=["jobs"])


# 查询批任务状态，支持前端轮询进度
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, request: Request):
    job_store = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        # 若内存任务未命中，再查询持久化任务（例如 Agent 建议任务）。
        persisted = request.app.state.review_service.get_persisted_job(job_id)
        if not persisted:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")  # 非法或过期任务ID
        return JobStatusResponse(
            job_id=persisted["job_id"],
            status=persisted["status"],
            total=1,
            completed=1 if persisted["status"] == "completed" else 0,
            failed=1 if persisted["status"] == "failed" else 0,
            progress=float(persisted["progress"]),
            created_at=persisted["created_at"],
            finished_at=persisted["finished_at"],
            results=[],
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=job.total,
        completed=job.completed,
        failed=job.failed,
        progress=job.progress,
        created_at=job.created_at,
        finished_at=job.finished_at,
        results=[JobResultItem(**item) for item in job.results],
    )
