'''Celery 配置模块：负责异步任务队列实例创建与统一配置。'''

from __future__ import annotations

import os

from celery import Celery

from app.core.env import load_project_env

load_project_env()


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


broker_url = os.getenv("QA_CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("QA_CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

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
)
