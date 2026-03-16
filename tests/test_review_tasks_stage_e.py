from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, desc, select

from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.main import create_app
from app.models.review_db import Annotation, AuditLog, Job, ModelOutput, QASample
from app.services.agent_service import AgentService
from app.services.review_service import ReviewService
from app.tasks.review_tasks import generate_agent_suggestion_task


def _reset_tables():
    with SessionLocal() as db:
        db.execute(delete(AuditLog))
        db.execute(delete(Annotation))
        db.execute(delete(Job))
        db.execute(delete(ModelOutput))
        db.execute(delete(QASample))
        db.commit()


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    _reset_tables()
    yield
    _reset_tables()


def _create_pending_sample_and_job(service: ReviewService) -> tuple[str, str]:
    info = service.record_inference(
        payload={
            "company_name": "test_company",
            "qa_time": None,
            "question": "项目什么时候投产？",
            "answer": "请关注后续定期报告。",
        },
        result={
            "root_id": 2,
            "root_label": "Evasive (打太极)",
            "root_confidence": 40.0,
            "sub_label": "推迟回答",
            "sub_confidence": 52.0,
            "warning": "低置信度",
        },
    )
    job_id = service.create_agent_job(info["sample_id"], requester_id="tester")
    return info["sample_id"], job_id


def test_generate_agent_task_success_path(monkeypatch):
    service = ReviewService(get_settings())
    sample_id, job_id = _create_pending_sample_and_job(service)

    def _fake_suggest(self, question: str, answer: str, model_result: dict):
        return {
            "root_label": "Evasive (打太极)",
            "sub_label": "推迟回答",
            "confidence": 0.91,
            "reason": "未给出明确投产时间",
            "provider": "dify",
            "raw_response": {"mocked": True},
        }

    monkeypatch.setattr(AgentService, "suggest", _fake_suggest)

    result = generate_agent_suggestion_task(job_id, sample_id)
    assert result == {"job_id": job_id, "status": "completed"}

    persisted = service.get_persisted_job(job_id)
    assert persisted is not None
    assert persisted["status"] == "completed"

    with SessionLocal() as db:
        agent_annotation = db.scalar(
            select(Annotation)
            .where(Annotation.sample_id == sample_id, Annotation.source == "agent")
            .order_by(desc(Annotation.created_at))
            .limit(1)
        )
        assert agent_annotation is not None
        assert agent_annotation.annotation["root_label"] == "Evasive (打太极)"


def test_generate_agent_task_failure_path(monkeypatch):
    service = ReviewService(get_settings())
    sample_id, job_id = _create_pending_sample_and_job(service)

    def _raise_suggest(self, question: str, answer: str, model_result: dict):
        raise RuntimeError("dify unavailable")

    monkeypatch.setattr(AgentService, "suggest", _raise_suggest)

    with pytest.raises(RuntimeError, match="dify unavailable"):
        generate_agent_suggestion_task(job_id, sample_id)

    persisted = service.get_persisted_job(job_id)
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert "dify unavailable" in (persisted["error"] or "")


def test_jobs_endpoint_reads_persisted_agent_job_state():
    service = ReviewService(get_settings())
    _, job_id = _create_pending_sample_and_job(service)
    service.fail_job(job_id, "manual failure for polling")

    app = create_app()
    with TestClient(app) as client:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()

    assert body["job_id"] == job_id
    assert body["status"] == "failed"
    assert body["failed"] == 1
    assert body["completed"] == 0
