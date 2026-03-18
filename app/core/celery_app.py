'''Celery 配置模块：负责异步任务队列实例创建与统一配置。'''

from __future__ import annotations

import os
from typing import Final

from celery import Celery
from celery.signals import worker_init, worker_process_init

from app.core.env import load_project_env

load_project_env()

WINDOWS_DEFAULT_POOL: Final[str] = "solo"
LINUX_DEFAULT_POOL: Final[str] = "prefork"


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(value, minimum)


def _default_worker_pool() -> str:
    explicit = os.getenv("QA_CELERY_WORKER_POOL", "").strip()
    if explicit:
        return explicit
    # Windows 下 prefork/billiard 容易触发 fast_trace_task 崩溃，默认切到 solo。
    if os.name == "nt":
        return WINDOWS_DEFAULT_POOL
    return LINUX_DEFAULT_POOL


def _default_worker_concurrency(worker_pool: str) -> int:
    fallback = 1 if worker_pool == WINDOWS_DEFAULT_POOL else 2
    return _as_int("QA_CELERY_WORKER_CONCURRENCY", fallback)


def _prepare_worker_database() -> None:
    # 与 API 启动逻辑对齐：在 worker 进程启动时执行 init_db，
    # 以便触发 PostgreSQL 不可用时的 SQLite 回退（若配置允许）。
    from app.core.database import init_db

    init_db()


def has_live_workers(timeout: float = 0.8) -> bool:
    # eager 模式等价于本地可执行任务。
    if bool(celery_app.conf.task_always_eager):
        return True
    try:
        inspect = celery_app.control.inspect(timeout=timeout)
        if inspect is None:
            return False
        ping = inspect.ping()
        return bool(ping)
    except Exception:
        return False


broker_url = os.getenv("QA_CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("QA_CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
worker_pool = _default_worker_pool()
worker_concurrency = _default_worker_concurrency(worker_pool)

celery_app = Celery("qa_review", broker=broker_url, backend=result_backend, include=["app.tasks.review_tasks"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
    result_expires=3600,
    task_always_eager=_as_bool("QA_CELERY_TASK_ALWAYS_EAGER", False),
    task_eager_propagates=True,
    worker_pool=worker_pool,
    worker_concurrency=worker_concurrency,
    broker_connection_retry_on_startup=True,
)


@worker_init.connect
def _on_worker_init(**_: object) -> None:
    _prepare_worker_database()


@worker_process_init.connect
def _on_worker_process_init(**_: object) -> None:
    _prepare_worker_database()
