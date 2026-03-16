from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import infer, jobs
from app.services.inference_service import InferenceService
from app.services.job_service import InMemoryJobStore


class FakeInferenceService:
    def evaluate(self, question: str, answer: str):
        assert question and answer
        return {
            "root_id": 2,
            "root_label": "Evasive (打太极)",
            "root_confidence": 88.1,
            "sub_label": "推迟回答",
            "sub_confidence": 77.2,
            "root_probabilities": {
                "Direct (直接响应)": 0.05,
                "Intermediate (避重就轻)": 0.08,
                "Evasive (打太极)": 0.87,
            },
            "sub_probabilities": {
                "推迟回答": 0.77,
                "转移话题": 0.05,
                "战略性模糊": 0.1,
                "外部归因": 0.08,
            },
            "entity_hits": [
                {"text": "净利润", "start": 2, "end": 5, "source_text": "question"},
                {"text": "公告", "start": 0, "end": 2, "source_text": "answer"},
            ],
            "warning": None,
        }


class FakeReviewService:
    def record_inference(self, payload, result):
        return {
            "sample_id": "sample-1",
            "is_low_confidence": False,
            "review_status": None,
        }

    def get_persisted_job(self, job_id: str):
        return None


def test_infer_response_contains_explainability_fields():
    app = FastAPI()
    app.state.inference_service = FakeInferenceService()
    app.state.review_service = FakeReviewService()
    app.include_router(infer.router)
    client = TestClient(app)

    resp = client.post(
        "/infer",
        json={
            "question": "公司净利润下滑原因是什么？",
            "answer": "请关注后续公告。",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert abs(sum(data["root_probabilities"].values()) - 1.0) < 1e-6
    assert "sub_probabilities" in data and data["sub_probabilities"]
    assert data["entity_hits"]
    for hit in data["entity_hits"]:
        assert hit["start"] >= 0
        assert hit["end"] > hit["start"]
        assert hit["source_text"] in {"question", "answer"}


def test_probability_map_sum_is_stable():
    prob_map = InferenceService._to_probability_map(
        probabilities=[0.1, 0.2, 0.7],
        label_map={0: "A", 1: "B", 2: "C"},
    )
    assert abs(sum(prob_map.values()) - 1.0) < 1e-6


def test_export_job_csv_success_and_guardrails():
    app = FastAPI()
    store = InMemoryJobStore()
    app.state.job_store = store
    app.state.review_service = FakeReviewService()
    app.include_router(jobs.router)
    client = TestClient(app)

    # 未完成任务不可导出
    pending = store.create(total=1)
    resp_pending = client.get(f"/jobs/{pending.job_id}/export")
    assert resp_pending.status_code == 400

    # 完成任务可导出
    completed = store.create(total=1)
    store.mark_running(completed.job_id)
    store.add_result(
        completed.job_id,
        {
            "index": 0,
            "sample_id": "s1",
            "status": "success",
            "result": {
                "root_id": 0,
                "root_label": "Direct (直接响应)",
                "root_confidence": 92.5,
                "sub_label": "财务表现指引",
                "sub_confidence": 80.0,
                "root_probabilities": {"Direct (直接响应)": 0.9},
                "sub_probabilities": {"财务表现指引": 0.8},
                "entity_hits": [{"text": "营收", "start": 1, "end": 3, "source_text": "question"}],
            },
            "error": None,
        },
        success=True,
    )
    store.mark_done(completed.job_id)

    resp_ok = client.get(f"/jobs/{completed.job_id}/export")
    assert resp_ok.status_code == 200
    assert "text/csv" in resp_ok.headers["content-type"]
    body = resp_ok.text
    assert "root_probabilities" in body
    assert "entity_hits" in body
    assert "Direct (直接响应)" in body

    # 不存在任务
    resp_404 = client.get("/jobs/not-exists/export")
    assert resp_404.status_code == 404
