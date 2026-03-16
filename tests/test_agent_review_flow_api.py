import time

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.agent_service import AgentService
from app.services.inference_service import InferenceService


def test_minimal_review_flow_with_agent_suggestion(monkeypatch):
    def _fake_evaluate(self, question: str, answer: str):
        return {
            "root_id": 2,
            "root_label": "Evasive (打太极)",
            "root_confidence": 22.0,
            "sub_label": "推迟回答",
            "sub_confidence": 31.0,
            "warning": "低置信度，建议复核",
        }

    def _fake_suggest(self, question: str, answer: str, model_result: dict):
        return {
            "root_label": "Evasive (打太极)",
            "sub_label": "推迟回答",
            "confidence": 0.88,
            "reason": "未给出明确时间或事实，属于典型推迟回答。",
            "provider": "dify",
            "raw_response": {"mocked": True},
        }

    monkeypatch.setattr(InferenceService, "evaluate", _fake_evaluate)
    monkeypatch.setattr(AgentService, "suggest", _fake_suggest)

    app = create_app()
    with TestClient(app) as client:
        infer_resp = client.post(
            "/infer",
            json={
                "company_name": "测试公司",
                "question": "今年分红会提升吗？",
                "answer": "请关注公司后续公告。",
            },
        )
        assert infer_resp.status_code == 200
        infer_payload = infer_resp.json()
        sample_id = infer_payload["sample_id"]
        assert infer_payload["review_status"] == "pending_review"

        queue_resp = client.get("/review/queue?page=1&page_size=10")
        assert queue_resp.status_code == 200
        queue = queue_resp.json()
        assert queue["total"] >= 1
        assert any(item["sample_id"] == sample_id for item in queue["items"])

        job_resp = client.post(f"/review/{sample_id}/agent-suggestion")
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        latest_status = None
        for _ in range(5):
            status_resp = client.get(f"/jobs/{job_id}")
            assert status_resp.status_code == 200
            latest_status = status_resp.json()["status"]
            if latest_status in {"completed", "failed"}:
                break
            time.sleep(0.1)
        assert latest_status == "completed"

        detail_resp = client.get(f"/review/{sample_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["agent_suggestion"] is not None
        assert detail["agent_suggestion"]["root_label"] == "Evasive (打太极)"
        assert detail["agent_suggestion"]["sub_label"] == "推迟回答"
