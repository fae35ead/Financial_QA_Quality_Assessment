from app.core import config as config_module
from app.core import env as env_module


def test_project_env_overrides_stale_process_value(monkeypatch):
    monkeypatch.setenv("QA_MAX_BATCH_ITEMS", "200")

    env_module.load_project_env.cache_clear()
    env_module.load_project_env()
    config_module.get_settings.cache_clear()

    settings = config_module.get_settings()
    assert settings.max_batch_items == 500
