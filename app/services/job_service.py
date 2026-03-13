'''任务状态服务层：提供批任务记录的创建、更新和查询能力。'''

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Optional
from uuid import uuid4


# 生成统一的 UTC 时间戳字符串
def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# 批任务实体：保存任务元数据、进度和结果明细
@dataclass
class JobRecord:
    job_id: str
    total: int
    status: str = "pending"
    completed: int = 0
    failed: int = 0
    created_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    results: list[dict[str, Any]] = field(default_factory=list)

    # 计算当前任务进度百分比
    @property
    def progress(self) -> float:
        if self.total <= 0:
            return 0.0
        return round((self.completed / self.total) * 100, 2)


# 线程安全的内存任务仓库（后续可替换为 Redis / PostgreSQL）
class InMemoryJobStore:
    # 初始化内存存储与互斥锁
    def __init__(self):
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}

    # 创建新任务并返回任务记录
    def create(self, total: int) -> JobRecord:
        with self._lock:
            job = JobRecord(job_id=str(uuid4()), total=total)  # 使用 UUID 作为任务ID
            self._jobs[job.job_id] = job  # 持久到内存字典
            return job

    # 将任务状态置为运行中
    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"

    # 写入单条样本结果并更新计数
    def add_result(self, job_id: str, item_result: dict[str, Any], success: bool) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.results.append(item_result)
            job.completed += 1
            if not success:
                job.failed += 1

    # 标记任务成功完成
    def mark_done(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.finished_at = _now_iso()

    # 标记任务失败并追加任务级错误信息
    def mark_failed(self, job_id: str, error_message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.finished_at = _now_iso()
            job.results.append(
                {
                    "index": -1,
                    "sample_id": None,
                    "status": "failed",
                    "error": error_message,
                    "result": None,
                }
            )

    # 按任务ID查询任务记录
    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    # 获取当前任务总数（用于监控）
    def total_jobs(self) -> int:
        with self._lock:
            return len(self._jobs)
