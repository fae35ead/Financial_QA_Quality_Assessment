from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


def test_options_preflight_allows_localhost_random_port(monkeypatch):
    monkeypatch.setenv("QA_DATABASE_URL", "sqlite:///./data/processed/test_stage_c.db")
    monkeypatch.setenv("QA_ALLOW_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500")
    monkeypatch.setenv("QA_ALLOW_ORIGIN_REGEX", r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        response = client.options(
            "/api/evaluate",
            headers={
                "Origin": "http://127.0.0.1:3720",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:3720"

    get_settings.cache_clear()

