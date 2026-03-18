from __future__ import annotations

import importlib
import os

import app.core.database as database_module
import app.core.celery_app as celery_module


def _reload_celery_module():
    return importlib.reload(celery_module)


def test_default_worker_pool_matches_platform(monkeypatch):
    monkeypatch.delenv("QA_CELERY_WORKER_POOL", raising=False)
    monkeypatch.delenv("QA_CELERY_WORKER_CONCURRENCY", raising=False)

    module = _reload_celery_module()
    expected_pool = "solo" if os.name == "nt" else "prefork"
    expected_concurrency = 1 if expected_pool == "solo" else 2

    assert module.celery_app.conf.worker_pool == expected_pool
    assert int(module.celery_app.conf.worker_concurrency) == expected_concurrency


def test_worker_pool_and_concurrency_allow_env_override(monkeypatch):
    monkeypatch.setenv("QA_CELERY_WORKER_POOL", "threads")
    monkeypatch.setenv("QA_CELERY_WORKER_CONCURRENCY", "4")

    module = _reload_celery_module()
    assert module.celery_app.conf.worker_pool == "threads"
    assert int(module.celery_app.conf.worker_concurrency) == 4


def test_invalid_concurrency_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("QA_CELERY_WORKER_POOL", "threads")
    monkeypatch.setenv("QA_CELERY_WORKER_CONCURRENCY", "not_a_number")

    module = _reload_celery_module()
    assert module.celery_app.conf.worker_pool == "threads"
    assert int(module.celery_app.conf.worker_concurrency) == 2


def test_prepare_worker_database_calls_init_db(monkeypatch):
    calls = {"count": 0}

    def _fake_init_db():
        calls["count"] += 1

    monkeypatch.setattr(database_module, "init_db", _fake_init_db)
    module = _reload_celery_module()
    module._prepare_worker_database()
    assert calls["count"] == 1


def test_has_live_workers_returns_true_when_eager_enabled():
    module = _reload_celery_module()
    old_value = module.celery_app.conf.task_always_eager
    module.celery_app.conf.task_always_eager = True
    try:
        assert module.has_live_workers()
    finally:
        module.celery_app.conf.task_always_eager = old_value


def test_has_live_workers_returns_false_when_ping_empty(monkeypatch):
    class _Inspect:
        @staticmethod
        def ping():
            return {}

    module = _reload_celery_module()
    old_value = module.celery_app.conf.task_always_eager
    module.celery_app.conf.task_always_eager = False
    monkeypatch.setattr(module.celery_app.control, "inspect", lambda timeout=0.8: _Inspect())
    try:
        assert not module.has_live_workers()
    finally:
        module.celery_app.conf.task_always_eager = old_value
