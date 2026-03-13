'''批量任务接口路由：接收样本列表并创建异步分析任务。'''

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.models.schemas import BatchInferRequest, BatchInferResponse

router = APIRouter(tags=["batch"])


# 创建批量推理任务，立即返回 job_id 供后续轮询
@router.post("/batch_infer", response_model=BatchInferResponse)
def batch_infer(payload: BatchInferRequest, background_tasks: BackgroundTasks, request: Request):
    settings = request.app.state.settings
    # 批量上限保护，避免单请求占用过多资源
    if len(payload.items) > settings.max_batch_items:
        raise HTTPException(
            status_code=400,
            detail=f"单次批量数量超过上限 {settings.max_batch_items}，请拆分后重试。",
        )

    batch_service = request.app.state.batch_service
    job_id = batch_service.submit(payload.items, background_tasks)  # 任务入队并异步执行
    return BatchInferResponse(job_id=job_id, status="pending", total=len(payload.items))
