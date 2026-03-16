'''数据库基础模块：负责数据库连接、会话管理与表初始化。'''

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.env import load_project_env

load_project_env()
logger = logging.getLogger(__name__)


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _fallback_verbose() -> bool:
    return _as_bool("QA_DATABASE_FALLBACK_VERBOSE", False)


def _default_database_url() -> str:
    project_root = Path(__file__).resolve().parents[2]
    sqlite_path = (project_root / "data" / "processed" / "stage_c.db").resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.as_posix()}"


def _build_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


DATABASE_URL = os.getenv("QA_DATABASE_URL", _default_database_url()).strip()
try:
    engine = _build_engine(DATABASE_URL)
except ModuleNotFoundError as exc:
    allow_fallback = _as_bool("QA_DATABASE_FALLBACK_TO_SQLITE", True)
    if allow_fallback and not DATABASE_URL.startswith("sqlite"):
        fallback_url = _default_database_url()
        if _fallback_verbose():
            logger.warning(
                "数据库驱动缺失（%s），已自动切换到 SQLite：%s",
                exc,
                fallback_url,
            )
        else:
            logger.warning("数据库驱动不可用，已自动切换到 SQLite：%s", fallback_url)
        DATABASE_URL = fallback_url
        engine = _build_engine(DATABASE_URL)
    else:
        raise
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def _switch_to_database(url: str) -> None:
    global DATABASE_URL, engine
    DATABASE_URL = url
    engine = _build_engine(url)
    SessionLocal.configure(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # 延迟导入，确保 Base 已经创建再加载模型。
    from app.models import review_db  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        allow_fallback = _as_bool("QA_DATABASE_FALLBACK_TO_SQLITE", True)
        if not allow_fallback or DATABASE_URL.startswith("sqlite"):
            raise

        fallback_url = _default_database_url()
        if _fallback_verbose():
            logger.warning(
                "数据库连接不可用，已自动切换到 SQLite：%s；原错误：%s",
                fallback_url,
                exc,
            )
        else:
            logger.warning("数据库连接不可用，已自动切换到 SQLite：%s", fallback_url)
        _switch_to_database(fallback_url)
        Base.metadata.create_all(bind=engine)
