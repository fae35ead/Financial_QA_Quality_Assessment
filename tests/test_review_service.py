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
        payload={"company_name": "测试公司", "qa_time": None, "question": "今年利润目标？", "answer": "请关注后续公告。"},
        result={
            "root_id": 2,
            "root_label": "Evasive (打太极)",
            "root_confidence": 30.0,
            "sub_label": "推迟回答",
            "sub_confidence": 55.0,
            "warning": "低置信度",
        },
    )
    queue = service.list_queue(page=1, page_size=10, statuses=["pending_review"], date_from=None, date_to=None)
    assert queue["total"] == 1
    assert queue["items"][0]["review_status"] == "pending_review"


def test_annotation_validation_rejects_invalid_sub_label():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={"company_name": None, "qa_time": None, "question": "研发投入？", "answer": "我们持续推进研发。"},
        result={
            "root_id": 0,
            "root_label": "Direct (直接响应)",
            "root_confidence": 35.0,
            "sub_label": "技术与研发进展",
            "sub_confidence": 40.0,
            "warning": "低置信度",
        },
    )
    with pytest.raises(ValueError):
        service.submit_human_annotation(
            sample_id=info["sample_id"],
            root_label="Direct (直接响应)",
            sub_label="外部归因",
            note="非法组合",
            annotator_id="tester",
            annotator_confidence=0.9,
        )


def test_agent_job_only_allowed_before_human_confirm():
    service = ReviewService(get_settings())
    _reset_tables()

    info = service.record_inference(
        payload={"company_name": None, "qa_time": None, "question": "分红计划？", "answer": "请关注公告。"},
        result={
            "root_id": 2,
            "root_label": "Evasive (打太极)",
            "root_confidence": 20.0,
            "sub_label": "推迟回答",
            "sub_confidence": 25.0,
            "warning": "低置信度",
        },
    )
    job_id = service.create_agent_job(info["sample_id"], requester_id="tester")
    assert isinstance(job_id, str) and job_id

    service.submit_human_annotation(
        sample_id=info["sample_id"],
        root_label="Evasive (打太极)",
        sub_label="推迟回答",
        note="人工确认",
        annotator_id="tester",
        annotator_confidence=0.95,
    )
    with pytest.raises(ValueError):
        service.create_agent_job(info["sample_id"], requester_id="tester")
