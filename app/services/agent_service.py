'''Agent服务层：负责向 Dify 请求复核建议并规范化返回结构。'''

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


class AgentService:
    def __init__(self):
        self.api_url = os.getenv("QA_DIFY_API_URL", "").strip()
        self.api_key = os.getenv("QA_DIFY_API_KEY", "").strip()
        self.user = os.getenv("QA_DIFY_USER", "qa_reviewer").strip()

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

        prompt = (
            "你是A股问答复核助手。请根据问题与回答，输出JSON："
            '{"root_label":"...","sub_label":"...","confidence":0-1,"reason":"..."}。'
            "root_label 仅能为 Direct (直接响应) / Intermediate (避重就轻) / Evasive (打太极)。"
        )
        payload = {
            "inputs": {
                "question": question,
                "answer": answer,
                "model_result": model_result,
            },
            "query": prompt,
            "response_mode": "blocking",
            "user": self.user,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=45) as client:
            resp = client.post(self.api_url, headers=headers, json=payload)
            resp.raise_for_status()
            body = resp.json()

        raw_text = str(body.get("answer") or body.get("output_text") or body.get("data", {}).get("answer") or "")
        if not raw_text:
            # 若 Dify 返回结构与预期不同，保留原始响应供人工定位。
            return {
                "root_label": model_result.get("root_label", "未知"),
                "sub_label": model_result.get("sub_label", "未知"),
                "confidence": 0.0,
                "reason": "Dify 返回结构不包含 answer 字段，已回退为模型结论。",
                "provider": "dify_invalid_response",
                "raw_response": body,
            }

        parsed = self._extract_json(raw_text)
        return {
            "root_label": parsed.get("root_label", model_result.get("root_label", "未知")),
            "sub_label": parsed.get("sub_label", model_result.get("sub_label", "未知")),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", "")),
            "provider": "dify",
            "raw_response": body,
        }
