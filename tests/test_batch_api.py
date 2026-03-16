from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import batch


class FakeBatchService:
    def __init__(self):
        self.submitted_count = 0

    def submit(self, items, background_tasks):
        self.submitted_count = len(items)
        return "job-test-1"


def _payload(size: int) -> dict:
    return {
        "items": [
            {
                "question": f"问题{i}",
                "answer": f"回答{i}",
            }
            for i in range(size)
        ]
    }


def test_batch_infer_accepts_500_items():
    app = FastAPI()
    fake_batch_service = FakeBatchService()
    app.state.settings = SimpleNamespace(max_batch_items=500)
    app.state.batch_service = fake_batch_service
    app.include_router(batch.router)

    client = TestClient(app)
    response = client.post("/batch_infer", json=_payload(500))

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-test-1"
    assert data["total"] == 500
    assert fake_batch_service.submitted_count == 500


def test_batch_infer_rejects_over_500_items():
    app = FastAPI()
    app.state.settings = SimpleNamespace(max_batch_items=500)
    app.state.batch_service = FakeBatchService()
    app.include_router(batch.router)

    client = TestClient(app)
    response = client.post("/batch_infer", json=_payload(501))

    assert response.status_code == 400
    assert "500" in response.json()["detail"]
