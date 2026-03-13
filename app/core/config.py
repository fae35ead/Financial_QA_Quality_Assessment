'''应用配置模块：集中管理环境变量解析、路径解析与CORS配置，并提供可缓存的全局 Settings 对象。'''

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# 将环境变量字符串解析为布尔值，未设置时使用默认值。
def _as_bool(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)  # 读取环境变量原始值。
    if raw is None:
        return default  # 环境变量缺失时回退到默认值。
    return raw.strip().lower() in {"1", "true", "yes", "on"}  # 支持常见真值写法。


# 解析目录类配置：优先使用环境变量，其次使用相对项目根目录的默认路径。
def _resolve_path(project_root: Path, env_name: str, default_relative: str) -> Path:
    raw_value = os.getenv(env_name, "").strip()  # 读取并去除首尾空白，避免无效空字符串。
    candidate = Path(raw_value).expanduser() if raw_value else project_root / default_relative  # 支持 ~ 家目录语法。
    return candidate.resolve()  # 统一转为绝对路径，减少运行时路径歧义。


# 解析允许跨域的来源列表，并根据是否为通配符决定是否允许携带凭据。
def _parse_cors_origins() -> tuple[list[str], bool]:
    raw = os.getenv(
        "QA_ALLOW_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:63342,http://127.0.0.1:63342",
    ).strip()  # 默认覆盖常见本地前端端口（含 JetBrains 预览端口 63342）。
    if raw == "*":
        return ["*"], False  # CORS 规范下通配符来源不能与 credentials=True 同时使用。

    origins = [item.strip() for item in raw.split(",") if item.strip()]  # 解析逗号分隔来源并清洗空项。
    if not origins:
        origins = [
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:63342",
            "http://127.0.0.1:63342",
        ]  # 兜底默认值，避免环境变量被误设为空导致全部拒绝。
    return origins, True  # 非通配符来源时允许携带凭据（如 Cookie/Authorization）。


# 统一保存应用运行配置，避免散落在代码中的硬编码。
@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    project_root: Path
    root_model_dir: Path
    direct_model_dir: Path
    evasive_model_dir: Path
    max_len: int
    low_conf_threshold: float
    entity_min_length: int
    max_batch_items: int
    lazy_load_models: bool
    allow_origins: list[str]
    allow_credentials: bool


# 构建并缓存 Settings；整个进程生命周期内复用同一份配置对象。
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    current_dir = Path(__file__).resolve().parent  # 当前配置文件所在目录。
    project_root = current_dir.parent.parent  # 约定项目根目录在 app/ 的上两级。

    allow_origins, allow_credentials = _parse_cors_origins()  # 先解析 CORS，供中间件直接使用。

    return Settings(
        app_name=os.getenv("QA_APP_NAME", "LCPPN 金融问答质量评估服务"),
        app_version=os.getenv("QA_APP_VERSION", "2.1.0"),
        project_root=project_root,
        root_model_dir=_resolve_path(project_root, "QA_ROOT_MODEL_DIR", "models/student_100k_T4_distilbert"),
        direct_model_dir=_resolve_path(project_root, "QA_DIRECT_MODEL_DIR", "models/lcppn_direct_classifier"),
        evasive_model_dir=_resolve_path(project_root, "QA_EVASIVE_MODEL_DIR", "models/lcppn_evasive_classifier"),
        max_len=int(os.getenv("QA_MAX_LEN", "384")),  # 推理最大序列长度。
        low_conf_threshold=float(os.getenv("QA_LOW_CONF_THRESHOLD", "0.45")),  # 低置信度判定阈值。
        entity_min_length=int(os.getenv("QA_ENTITY_MIN_LENGTH", "5")),  # 实体最小长度阈值。
        max_batch_items=int(os.getenv("QA_MAX_BATCH_ITEMS", "200")),  # 批量接口单次最大条目数。
        lazy_load_models=_as_bool("QA_LAZY_LOAD_MODELS", True),  # 是否延迟加载模型以加快启动。
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
    )
