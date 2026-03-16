'''Agent服务层：负责向 Dify 请求复核建议并规范化返回结构。'''

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from app.core.env import load_project_env

load_project_env()


class AgentService:
    ALLOWED_ROOT_LABELS = {
        "Direct (直接响应)",
        "Intermediate (避重就轻)",
        "Evasive (打太极)",
    }
    ROOT_LABEL_ALIASES = {
        "direct": "Direct (直接响应)",
        "intermediate": "Intermediate (避重就轻)",
        "evasive": "Evasive (打太极)",
        "fully evasive": "Evasive (打太极)",
    }
    SUB_LABEL_ALIASES = {
        "Direct (直接响应)": {},
        "Intermediate (避重就轻)": {
            "部分响应": "无下游细分 (部分响应)",
            "避重就轻": "无下游细分 (部分响应)",
        },
        "Evasive (打太极)": {
            "以定期报告为准": "推迟回答",
            "请关注后续公告": "推迟回答",
            "后续披露": "推迟回答",
        },
    }

    def __init__(self):
        self.api_url = os.getenv("QA_DIFY_API_URL", "").strip()
        self.api_key = os.getenv("QA_DIFY_API_KEY", "").strip()
        self.user = os.getenv("QA_DIFY_USER", "qa_reviewer").strip()
        # 可选：自定义 Dify 输出字段路径（示例：data.outputs.result）
        self.output_path = os.getenv("QA_DIFY_OUTPUT_PATH", "").strip()
        # 默认将 model_result 作为 JSON 字符串传入，兼容 Dify text-input。
        self.model_result_as_json = os.getenv("QA_DIFY_MODEL_RESULT_AS_JSON", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Dify 请求超时配置：默认读超时 20 秒，避免复核任务长时间阻塞。
        self.connect_timeout = self._parse_timeout("QA_DIFY_CONNECT_TIMEOUT", default=5.0)
        self.write_timeout = self._parse_timeout("QA_DIFY_WRITE_TIMEOUT", default=10.0)
        self.read_timeout = self._parse_timeout("QA_DIFY_READ_TIMEOUT", default=20.0)

    def _parse_timeout(self, env_name: str, default: float) -> float:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return max(value, 0.1)

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Agent 返回内容中未找到可解析的 JSON 对象。")
        return json.loads(match.group(0))

    def _dig_value(self, data: dict[str, Any], path: str) -> Any:
        if not path:
            return None
        current: Any = data
        for token in path.split("."):
            token = token.strip()
            if not token or not isinstance(current, dict) or token not in current:
                return None
            current = current[token]
        return current

    def _extract_output(self, body: dict[str, Any]) -> Any:
        if self.output_path:
            custom = self._dig_value(body, self.output_path)
            if custom not in (None, ""):
                return custom

        for path in (
            "answer",
            "output_text",
            "data.answer",
            "data.outputs.answer",
            "data.outputs.result",
            "data.outputs.text",
            "outputs.answer",
            "outputs.result",
            "outputs.text",
        ):
            value = self._dig_value(body, path)
            if value not in (None, ""):
                return value
        return None

    def _normalize_suggestion(self, parsed: dict[str, Any], model_result: dict[str, Any]) -> dict[str, Any]:
        raw_root_label = str(parsed.get("root_label", model_result.get("root_label", "未知"))).strip()
        root_key = raw_root_label.lower()
        root_label = self.ROOT_LABEL_ALIASES.get(root_key, raw_root_label)
        if root_label not in self.ALLOWED_ROOT_LABELS:
            root_label = str(model_result.get("root_label", "未知"))

        raw_sub_label = str(parsed.get("sub_label", model_result.get("sub_label", "未知"))).strip()
        sub_aliases = self.SUB_LABEL_ALIASES.get(root_label, {})
        sub_label = sub_aliases.get(raw_sub_label, raw_sub_label)
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        reason = str(parsed.get("reason", "")).strip()
        evidence = str(parsed.get("evidence", "")).strip()
        return {
            "root_label": root_label,
            "sub_label": sub_label or str(model_result.get("sub_label", "未知")),
            "confidence": confidence,
            "reason": reason,
            "evidence": evidence or None,
        }

    def _invalid_response(self, model_result: dict[str, Any], body: dict[str, Any], message: str) -> dict[str, Any]:
        return {
            "root_label": model_result.get("root_label", "未知"),
            "sub_label": model_result.get("sub_label", "未知"),
            "confidence": 0.0,
            "reason": message,
            "provider": "dify_invalid_response",
            "raw_response": body,
        }

    def _fallback(self, model_result: dict[str, Any]) -> dict[str, Any]:
        return {
            "root_label": model_result.get("root_label", "未知"),
            "sub_label": model_result.get("sub_label", "未知"),
            "confidence": float(model_result.get("root_confidence", 0.0)) / 100.0,
            "reason": "Dify 未配置，返回基于模型结论的占位建议。",
            "provider": "fallback",
            "raw_response": None,
        }

    def suggest(self, question: str, answer: str, model_result: dict[str, Any]) -> dict[str, Any]:
        if not self.api_url:
            return self._fallback(model_result)

        model_result_input: Any = model_result
        if self.model_result_as_json:
            model_result_input = json.dumps(model_result, ensure_ascii=False)

        prompt = (
            "你是A股问答复核助手。请根据问题与回答，输出JSON："
            '{"root_label":"...","sub_label":"...","confidence":0-1,"reason":"...","evidence":"..."}。'
            "root_label 仅能为 Direct (直接响应) / Intermediate (避重就轻) / Evasive (打太极)。"
            "sub_label 需使用系统标准标签，不要输出自定义描述。"
        )
        payload = {
            "inputs": {
                "question": question,
                "answer": answer,
                "model_result": model_result_input,
            },
            "query": prompt,
            "response_mode": "blocking",
            "user": self.user,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(connect=self.connect_timeout, read=self.read_timeout, write=self.write_timeout, pool=5.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(self.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
        except httpx.TimeoutException:
            return {
                "root_label": model_result.get("root_label", "未知"),
                "sub_label": model_result.get("sub_label", "未知"),
                "confidence": float(model_result.get("root_confidence", 0.0)) / 100.0,
                "reason": "Dify 响应超时，已回退为模型结论。",
                "provider": "dify_timeout_fallback",
                "raw_response": None,
            }

        raw_output = self._extract_output(body)
        if raw_output is None:
            return self._invalid_response(
                model_result=model_result,
                body=body,
                message="Dify 返回结构未命中输出字段，已回退为模型结论。",
            )

        try:
            parsed = raw_output if isinstance(raw_output, dict) else self._extract_json(str(raw_output))
        except (ValueError, json.JSONDecodeError) as exc:
            return self._invalid_response(
                model_result=model_result,
                body=body,
                message=f"Dify 输出解析失败: {exc}",
            )

        normalized = self._normalize_suggestion(parsed, model_result)
        return {
            **normalized,
            "provider": "dify",
            "raw_response": body,
        }
