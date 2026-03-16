import httpx

from app.services.agent_service import AgentService


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, body, recorder=None):
        self._body = body
        self._recorder = recorder if recorder is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self._recorder["url"] = url
        self._recorder["headers"] = headers
        self._recorder["json"] = json
        return _FakeResponse(self._body)


class _TimeoutClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        raise httpx.ReadTimeout("timeout")


def _set_base_env(monkeypatch):
    monkeypatch.setenv("QA_DIFY_API_URL", "https://example.com/workflows/run")
    monkeypatch.setenv("QA_DIFY_API_KEY", "test-key")
    monkeypatch.setenv("QA_DIFY_USER", "qa_reviewer")


def test_parse_data_outputs_result(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.delenv("QA_DIFY_OUTPUT_PATH", raising=False)

    body = {
        "data": {
            "outputs": {
                "result": '{"root_label":"Evasive (打太极)","sub_label":"转移话题","confidence":0.83,"reason":"命中话术"}'
            }
        }
    }
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body))

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Direct (直接响应)", "sub_label": "财务表现指引", "root_confidence": 55.0},
    )

    assert result["provider"] == "dify"
    assert result["root_label"] == "Evasive (打太极)"
    assert result["sub_label"] == "转移话题"
    assert result["confidence"] == 0.83


def test_parse_custom_output_path_with_dict(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.setenv("QA_DIFY_OUTPUT_PATH", "data.outputs.review_result")

    body = {
        "data": {
            "outputs": {
                "review_result": {
                    "root_label": "Direct (直接响应)",
                    "sub_label": "财务表现指引",
                    "confidence": 0.92,
                    "reason": "含具体指标",
                }
            }
        }
    }
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body))

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Evasive (打太极)", "sub_label": "推迟回答", "root_confidence": 20.0},
    )

    assert result["provider"] == "dify"
    assert result["root_label"] == "Direct (直接响应)"
    assert result["confidence"] == 0.92


def test_invalid_root_label_falls_back_to_model_result(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.delenv("QA_DIFY_OUTPUT_PATH", raising=False)

    body = {
        "answer": '{"root_label":"UnknownLabel","sub_label":"无效","confidence":1.2,"reason":"异常输出"}',
    }
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body))

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Intermediate (避重就轻)", "sub_label": "无下游细分 (部分响应)", "root_confidence": 48.0},
    )

    assert result["provider"] == "dify"
    assert result["root_label"] == "Intermediate (避重就轻)"
    assert result["confidence"] == 1.0


def test_missing_output_returns_invalid_response(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.delenv("QA_DIFY_OUTPUT_PATH", raising=False)

    body = {"data": {"outputs": {}}}
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body))

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Evasive (打太极)", "sub_label": "推迟回答", "root_confidence": 32.0},
    )

    assert result["provider"] == "dify_invalid_response"
    assert result["root_label"] == "Evasive (打太极)"
    assert result["confidence"] == 0.0


def test_user_sample_alias_can_be_normalized(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.delenv("QA_DIFY_OUTPUT_PATH", raising=False)

    body = {
        "answer": '{"root_label":"Evasive","sub_label":"以定期报告为准","confidence":0.95,"reason":"以披露节奏回避问题","evidence":"请关注后续定期报告"}',
    }
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body))

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Direct (直接响应)", "sub_label": "财务表现指引", "root_confidence": 55.0},
    )

    assert result["provider"] == "dify"
    assert result["root_label"] == "Evasive (打太极)"
    assert result["sub_label"] == "推迟回答"
    assert result["confidence"] == 0.95
    assert result["evidence"] == "请关注后续定期报告"


def test_model_result_sent_as_json_string_by_default(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.delenv("QA_DIFY_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("QA_DIFY_MODEL_RESULT_AS_JSON", raising=False)

    body = {
        "answer": '{"root_label":"Evasive","sub_label":"以定期报告为准","confidence":0.95,"reason":"以披露节奏回避问题"}',
    }
    recorder = {}
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _FakeClient(body, recorder=recorder))

    service = AgentService()
    service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Evasive (打太极)", "sub_label": "推迟回答", "root_confidence": 55.0},
    )

    assert isinstance(recorder["json"]["inputs"]["model_result"], str)


def test_timeout_returns_fallback_result(monkeypatch):
    _set_base_env(monkeypatch)
    monkeypatch.setattr("app.services.agent_service.httpx.Client", lambda timeout: _TimeoutClient())

    service = AgentService()
    result = service.suggest(
        question="问题",
        answer="回答",
        model_result={"root_label": "Direct (直接响应)", "sub_label": "财务表现指引", "root_confidence": 81.0},
    )

    assert result["provider"] == "dify_timeout_fallback"
    assert result["root_label"] == "Direct (直接响应)"
    assert result["sub_label"] == "财务表现指引"
    assert result["confidence"] == 0.81
