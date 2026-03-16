from __future__ import annotations

import re
from pathlib import Path


def _compose_text() -> str:
    compose_file = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    return compose_file.read_text(encoding="utf-8")


def _service_block(compose_text: str, service_name: str) -> str:
    lines = compose_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.match(rf"^\s{{2}}{re.escape(service_name)}:\s*$", line):
            start = idx
            break
    if start is None:
        return ""

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if re.match(r"^\s{2}[a-zA-Z0-9_-]+:\s*$", lines[idx]):
            end = idx
            break
    return "\n".join(lines[start:end])


def test_compose_contains_required_services():
    content = _compose_text()
    for service in ("api", "worker", "redis", "postgres"):
        assert _service_block(content, service), f"missing service: {service}"


def test_api_command_disallows_reload():
    api_block = _service_block(_compose_text(), "api")
    assert "--reload" not in api_block


def test_api_worker_command_disallow_runtime_pip_install():
    content = _compose_text()
    assert "pip install" not in _service_block(content, "api")
    assert "pip install" not in _service_block(content, "worker")


def test_healthchecks_and_service_healthy_dependencies_present():
    content = _compose_text()
    postgres_block = _service_block(content, "postgres")
    redis_block = _service_block(content, "redis")
    api_block = _service_block(content, "api")
    worker_block = _service_block(content, "worker")

    assert "healthcheck:" in postgres_block
    assert "healthcheck:" in redis_block
    assert "healthcheck:" in api_block

    assert re.search(r"postgres:\s*\n\s*condition:\s*service_healthy", api_block)
    assert re.search(r"redis:\s*\n\s*condition:\s*service_healthy", api_block)
    assert re.search(r"postgres:\s*\n\s*condition:\s*service_healthy", worker_block)
    assert re.search(r"redis:\s*\n\s*condition:\s*service_healthy", worker_block)
    assert re.search(r"api:\s*\n\s*condition:\s*service_healthy", worker_block)


def test_compose_enforces_no_sqlite_fallback_for_stage_e():
    content = _compose_text()
    api_block = _service_block(content, "api")
    worker_block = _service_block(content, "worker")
    expected = 'QA_DATABASE_FALLBACK_TO_SQLITE: "false"'
    assert expected in api_block
    assert expected in worker_block
