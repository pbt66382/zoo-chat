"""
LLM 意图分类器：让 DeepSeek 把用户问题归入预定义意图之一，并给出置信度。

策略
----
* Prompt 中列出所有意图 id + name + description + 部分示例。
* 强制 LLM 用 JSON 格式输出 ``{"intent_id": "...", "confidence": 0~1}``，
  方便程序解析并兜底。
* 解析失败/输出非法时降级为 ``out_of_scope``，confidence=0。
* 启用低 temperature（0.0）确保稳定性。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.intent.intents import INTENT_OUT_OF_SCOPE, list_intents
from app.llm.deepseek_client import get_llm

logger = logging.getLogger("zoo_chat.intent")


@dataclass
class IntentResult:
    intent_id: str
    confidence: float
    raw_response: str = ""


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _build_system_prompt() -> str:
    intents = list_intents()
    lines = ["你是一个专业的意图分类器，负责将用户问题归类到下面的意图之一：", ""]
    for it in intents:
        examples_str = "、".join(f'"{ex}"' for ex in it.examples[:3])
        lines.append(f"- {it.id} ({it.name}): {it.description}")
        if examples_str:
            lines.append(f"  示例：{examples_str}")
    lines.extend([
        "",
        "请严格按以下 JSON 格式返回，不要额外解释：",
        '{"intent_id": "<上面列出的某个 id>", "confidence": <0~1 之间的浮点数>}',
        "",
        "判定原则：",
        "1. 如果问题与 Zoo 会议服务完全无关，必须返回 out_of_scope。",
        "2. confidence 反映把握程度：clearly 命中 -> 0.9 以上；模糊或可能多义 -> 0.5~0.8；几乎是猜的 -> 0.3 以下。",
        "3. 只能返回上面列出的 intent_id，禁止编造新的。",
    ])
    return "\n".join(lines)


def _parse_response(text: str) -> Optional[dict[str, Any]]:
    """从 LLM 输出中提取 JSON dict；容忍前后多余文字。"""
    text = text.strip()
    if not text:
        return None
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退而求其次：抓第一个 {...}
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class IntentClassifier:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm
        self._system_prompt: Optional[str] = None

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm(temperature=0.0, max_tokens=64)
        return self._llm

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _build_system_prompt()
        return self._system_prompt

    def classify(self, question: str) -> IntentResult:
        """同步调用：返回 IntentResult。失败一律降级为 out_of_scope。"""
        if not question or not question.strip():
            return IntentResult(INTENT_OUT_OF_SCOPE, 0.0, raw_response="<empty input>")

        llm = self._get_llm()
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=question),
        ]
        try:
            response = llm.invoke(messages)
        except Exception as exc:  # 网络/鉴权错误：降级
            logger.exception("intent_classify_llm_error question=%r err=%s", question, exc)
            return IntentResult(INTENT_OUT_OF_SCOPE, 0.0, raw_response=f"<error: {exc}>")

        raw = getattr(response, "content", str(response)) or ""
        parsed = _parse_response(raw)
        if not parsed:
            logger.warning("intent_classify_parse_failed raw=%r", raw[:200])
            return IntentResult(INTENT_OUT_OF_SCOPE, 0.0, raw_response=raw)

        intent_id = parsed.get("intent_id", INTENT_OUT_OF_SCOPE)
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # 校验 id 合法性
        valid_ids = {it.id for it in list_intents()}
        if intent_id not in valid_ids:
            logger.warning("intent_classify_invalid_id id=%r raw=%r", intent_id, raw[:200])
            intent_id = INTENT_OUT_OF_SCOPE
            confidence = 0.0

        return IntentResult(intent_id=intent_id, confidence=confidence, raw_response=raw)


_default_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """单例入口，方便统一替换/测试时 monkeypatch。"""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = IntentClassifier()
    return _default_classifier
