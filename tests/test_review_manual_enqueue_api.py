from fastapi.testclient import TestClient

from app.main import create_app
from app.services.inference_service import InferenceService


def test_manual_enqueue_from_analysis_flow(monkeypatch):
    def _fake_evaluate(self, question: str, answer: str):
        return {
            "root_id": 0,
            "root_label": "Direct (直接响应)",
            "root_confidence": 96.0,
            "sub_label": "财务表现指引",
            "sub_confidence": 94.0,
            "warning": None,
        }

    monkeypatch.setattr(InferenceService, "evaluate", _fake_evaluate)

    app = create_app()
    with TestClient(app) as client:
        infer_resp = client.post(
            "/api/evaluate",
            json={
                "question": "今年净利润指引是什么？",
                "answer": "预计保持稳健增长。",
            },
        )
        assert infer_resp.status_code == 200
        payload = infer_resp.json()
        assert payload["sample_id"]
        assert payload["review_status"] is None

        enqueue_resp = client.post(f"/review/{payload['sample_id']}/enqueue")
        assert enqueue_resp.status_code == 200
        enqueue_data = enqueue_resp.json()
        assert enqueue_data["review_status"] == "pending_review"

        queue_resp = client.get("/review/queue?page=1&page_size=20&status=pending_review")
        assert queue_resp.status_code == 200
        queue_data = queue_resp.json()
        assert any(item["sample_id"] == payload["sample_id"] for item in queue_data["items"])

