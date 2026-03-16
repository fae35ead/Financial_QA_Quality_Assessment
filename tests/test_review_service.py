from __future__ import annotations

import pytest
from sqlalchemy import delete

from app.core.config import get_settings
from app.core.database import SessionLocal, init_db
from app.models.review_db import Annotation, AuditLog, Job, ModelOutput, QASample
from app.services.review_service import ReviewService


def _reset_tables():
    with SessionLocal() as db:
        db.execute(delete(AuditLog))
        db.execute(delete(Annotation))
        db.execute(delete(Job))
        db.execute(delete(ModelOutput))
        db.execute(delete(QASample))
        db.commit()


def _root_label(service: ReviewService, prefix: str) -> str:
    return next(label for label in service.ROOT_LABELS if label.startswith(prefix))


def _direct_sub_label(service: ReviewService) -> str:
    return sorted(service.DIRECT_SUB_LABELS)[0]


def _evasive_sub_label(service: ReviewService) -> str:
    return sorted(service.EVASIVE_SUB_LABELS)[0]


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    _reset_tables()
    yield
    _reset_tables()


def test_low_confidence_sample_enters_review_queue():
    service = ReviewService(get_settings())
    _reset_tables()

    service.record_inference(
        payload={"company_name": "test_company", "qa_time": None, "question": "q1", "answer": "a1"},
        result={
            "root_id": 2,
            "root_label": _root_label(service, "Evasive"),
            "root_confidence": 30.0,
            "sub_label": _evasive_sub_label(service),
            "sub_confidence": 55.0,
            "warning": "low confidence",
        },
    )
    queue = service.list_queue(page=1, page_size=10, statuses=["pending_review"], date_from=None, date_to=None)
    assert queue["total"] == 1
    assert queue["items"][0]["review_status"] == "pending_review"


def test_annotation_validation_rejects_invalid_sub_label():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={"company_name": None, "qa_time": None, "question": "q2", "answer": "a2"},
        result={
            "root_id": 0,
            "root_label": _root_label(service, "Direct"),
            "root_confidence": 35.0,
            "sub_label": _direct_sub_label(service),
            "sub_confidence": 40.0,
            "warning": "low confidence",
        },
    )
    with pytest.raises(ValueError):
        service.submit_human_annotation(
            sample_id=info["sample_id"],
            root_label=_root_label(service, "Direct"),
            sub_label="invalid_sub_label",
            note="illegal label pair",
            annotator_id="tester",
            annotator_confidence=0.9,
        )


def test_agent_job_only_allowed_before_human_confirm():
    service = ReviewService(get_settings())
    _reset_tables()

    evasive_root = _root_label(service, "Evasive")
    evasive_sub = _evasive_sub_label(service)
    info = service.record_inference(
        payload={"company_name": None, "qa_time": None, "question": "q3", "answer": "a3"},
        result={
            "root_id": 2,
            "root_label": evasive_root,
            "root_confidence": 20.0,
            "sub_label": evasive_sub,
            "sub_confidence": 25.0,
            "warning": "low confidence",
        },
    )
    job_id = service.create_agent_job(info["sample_id"], requester_id="tester")
    assert isinstance(job_id, str) and job_id

    service.submit_human_annotation(
        sample_id=info["sample_id"],
        root_label=evasive_root,
        sub_label=evasive_sub,
        note="human confirmed",
        annotator_id="tester",
        annotator_confidence=0.95,
    )
    with pytest.raises(ValueError):
        service.create_agent_job(info["sample_id"], requester_id="tester")


def test_root_confidence_below_65_enters_review_queue():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={"company_name": "test_company", "qa_time": None, "question": "q4", "answer": "a4"},
        result={
            "root_id": 0,
            "root_label": _root_label(service, "Direct"),
            "root_confidence": 60.0,
            "sub_label": _direct_sub_label(service),
            "sub_confidence": 90.0,
            "warning": None,
        },
    )
    assert info["is_low_confidence"] is True
    assert info["review_status"] == "pending_review"


def test_manual_enqueue_for_high_confidence_sample():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={"company_name": "test_company", "qa_time": None, "question": "q5", "answer": "a5"},
        result={
            "root_id": 0,
            "root_label": _root_label(service, "Direct"),
            "root_confidence": 92.0,
            "sub_label": _direct_sub_label(service),
            "sub_confidence": 93.0,
            "warning": None,
        },
    )
    assert info["review_status"] is None

    enqueue = service.enqueue_manual_review(sample_id=info["sample_id"], requester_id="tester")
    assert enqueue["review_status"] == "pending_review"
    assert enqueue["enqueued"] is True

    queue = service.list_queue(page=1, page_size=10, statuses=["pending_review"], date_from=None, date_to=None)
    assert queue["total"] == 1
    assert queue["items"][0]["sample_id"] == info["sample_id"]


def test_manual_enqueue_is_idempotent_when_already_pending():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={
            "company_name": "test_company",
            "qa_time": None,
            "question": "when will the project be delivered",
            "answer": "the project is progressing as planned",
        },
        result={
            "root_id": 0,
            "root_label": "Direct (mock)",
            "root_confidence": 97.0,
            "sub_label": "project_progress",
            "sub_confidence": 96.0,
            "warning": None,
        },
    )

    first = service.enqueue_manual_review(sample_id=info["sample_id"], requester_id="tester")
    second = service.enqueue_manual_review(sample_id=info["sample_id"], requester_id="tester")

    assert first["review_status"] == "pending_review"
    assert first["enqueued"] is True
    assert second["review_status"] == "pending_review"
    assert second["enqueued"] is False

    queue = service.list_queue(page=1, page_size=10, statuses=["pending_review"], date_from=None, date_to=None)
    matching = [item for item in queue["items"] if item["sample_id"] == info["sample_id"]]
    assert len(matching) == 1
