import os
from pathlib import Path


# 测试默认使用独立 SQLite，避免受本地 .env 中 PostgreSQL 依赖影响。
if "QA_DATABASE_URL" not in os.environ:
    test_db = (Path(__file__).resolve().parents[1] / "data" / "processed" / "test_stage_c.db").as_posix()
    os.environ["QA_DATABASE_URL"] = f"sqlite:///{test_db}"

os.environ.setdefault("QA_CELERY_TASK_ALWAYS_EAGER", "true")
