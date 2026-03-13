'''数据库基础模块：负责数据库连接、会话管理与表初始化。'''

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker


def _default_database_url() -> str:
    project_root = Path(__file__).resolve().parents[2]
    sqlite_path = (project_root / "data" / "processed" / "stage_c.db").resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.as_posix()}"


DATABASE_URL = os.getenv("QA_DATABASE_URL", _default_database_url()).strip()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # 延迟导入，确保 Base 已经创建再加载模型。
    from app.models import review_db  # noqa: F401

    Base.metadata.create_all(bind=engine)
