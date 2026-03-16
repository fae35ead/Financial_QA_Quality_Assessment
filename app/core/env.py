"""环境变量加载：优先读取项目根目录 .env，并保证全进程只加载一次。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - 兼容未安装 python-dotenv 的场景
    load_dotenv = None


@lru_cache(maxsize=1)
def load_project_env() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    if env_file.exists() and load_dotenv is not None:
        load_dotenv(env_file, override=False)
        return env_file
    return None
