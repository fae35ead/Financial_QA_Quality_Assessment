from app.core import database


def test_as_bool_prefers_env_file_value_when_enabled(monkeypatch):
    monkeypatch.setenv("QA_DATABASE_FALLBACK_TO_SQLITE", "false")
    monkeypatch.setattr(
        database,
        "_read_env_file_value",
        lambda key: "true" if key == "QA_DATABASE_FALLBACK_TO_SQLITE" else None,
    )

    assert database._as_bool("QA_DATABASE_FALLBACK_TO_SQLITE", default=False, prefer_env_file=True) is True


def test_as_bool_uses_process_env_when_prefer_env_file_disabled(monkeypatch):
    monkeypatch.setenv("QA_DATABASE_FALLBACK_TO_SQLITE", "false")
    monkeypatch.setattr(database, "_read_env_file_value", lambda key: "true")

    assert database._as_bool("QA_DATABASE_FALLBACK_TO_SQLITE", default=True, prefer_env_file=False) is False
