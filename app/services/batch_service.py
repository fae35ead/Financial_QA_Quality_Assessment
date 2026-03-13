'''批量服务层：负责创建批任务并在后台逐条执行推理。'''

from __future__ import annotations

from fastapi import BackgroundTasks

from app.models.schemas import BatchInferItem
from app.services.inference_service import InferenceService
from app.services.job_service import InMemoryJobStore
from app.services.review_service import ReviewService


# 批处理服务：将批量请求转为可跟踪的任务执行过程
class BatchService:
    # 初始化批处理服务依赖
    def __init__(
        self,
        inference_service: InferenceService,
        job_store: InMemoryJobStore,
        review_service: ReviewService,
    ):
        self.inference_service = inference_service
        self.job_store = job_store
        self.review_service = review_service

    # 创建批任务并注册后台执行
    def submit(self, items: list[BatchInferItem], background_tasks: BackgroundTasks) -> str:
        job = self.job_store.create(total=len(items))  # 先创建任务记录，获得 job_id
        background_tasks.add_task(self._run_job, job.job_id, items)  # 后台异步执行
        return job.job_id

    # 实际的批任务执行逻辑：逐条推理并回写任务状态
    def _run_job(self, job_id: str, items: list[BatchInferItem]) -> None:
        self.job_store.mark_running(job_id)
        try:
            for index, item in enumerate(items):
                try:
                    result = self.inference_service.evaluate(item.question, item.answer)
                    self.review_service.record_inference(
                        payload={
                            "company_name": item.company_name,
                            "qa_time": item.qa_time,
                            "question": item.question,
                            "answer": item.answer,
                            "sample_id": item.sample_id,
                        },
                        result=result,
                    )
                    self.job_store.add_result(
                        job_id=job_id,
                        item_result={
                            "index": index,
                            "sample_id": item.sample_id,
                            "status": "success",
                            "result": result,
                            "error": None,
                        },
                        success=True,  # 成功样本
                    )
                except Exception as exc:
                    self.job_store.add_result(
                        job_id=job_id,
                        item_result={
                            "index": index,
                            "sample_id": item.sample_id,
                            "status": "failed",
                            "result": None,
                            "error": str(exc),
                        },
                        success=False,  # 失败样本
                    )
            self.job_store.mark_done(job_id)  # 批任务全部完成
        except Exception as exc:
            self.job_store.mark_failed(job_id, f"批任务执行异常: {exc}")  # 任务级异常
