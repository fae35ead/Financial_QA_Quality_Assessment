'''任务查询接口路由：根据 job_id 返回批量任务状态与结果。'''

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

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


# 将批任务结果序列化为 CSV 文本，供前端下载
def _job_to_csv(job) -> str:
    fieldnames = [
        "index",
        "sample_id",
        "status",
        "root_id",
        "root_label",
        "root_confidence",
        "sub_label",
        "sub_confidence",
        "root_probabilities",
        "sub_probabilities",
        "entity_hits",
        "error",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for item in job.results:
        result = item.get("result") or {}
        writer.writerow(
            {
                "index": item.get("index"),
                "sample_id": item.get("sample_id"),
                "status": item.get("status"),
                "root_id": result.get("root_id"),
                "root_label": result.get("root_label"),
                "root_confidence": result.get("root_confidence"),
                "sub_label": result.get("sub_label"),
                "sub_confidence": result.get("sub_confidence"),
                "root_probabilities": json.dumps(result.get("root_probabilities", {}), ensure_ascii=False),
                "sub_probabilities": json.dumps(result.get("sub_probabilities", {}), ensure_ascii=False),
                "entity_hits": json.dumps(result.get("entity_hits", []), ensure_ascii=False),
                "error": item.get("error"),
            }
        )
    return output.getvalue()


# 导出已完成批任务结果（CSV）
@router.get("/jobs/{job_id}/export")
def export_job_result(job_id: str, request: Request):
    job_store = request.app.state.job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"批任务不存在: {job_id}")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="仅已完成批任务允许导出。")

    csv_content = _job_to_csv(job)
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="batch_job_{job_id}.csv"',
        },
    )
