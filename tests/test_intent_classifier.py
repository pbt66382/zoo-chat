"""
意图分类器单元测试：用 mock LLM 验证解析、降级与边界。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.intent.classifier import IntentClassifier
from app.intent.intents import INTENT_OUT_OF_SCOPE, list_intents


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=content)
    return llm


class TestIntentParsing:
    def test_parses_clean_json(self):
        llm = _make_llm('{"intent_id": "screen_share", "confidence": 0.95}')
        c = IntentClassifier(llm=llm)
        result = c.classify("怎么共享屏幕")
        assert result.intent_id == "screen_share"
        assert result.confidence == pytest.approx(0.95)

    def test_extracts_json_from_noisy_text(self):
        llm = _make_llm('好的，结果是：{"intent_id": "meeting_create", "confidence": 0.8} 谢谢。')
        c = IntentClassifier(llm=llm)
        result = c.classify("怎么开会")
        assert result.intent_id == "meeting_create"
        assert result.confidence == pytest.approx(0.8)

    def test_clamps_confidence_to_unit_interval(self):
        llm = _make_llm('{"intent_id": "greet", "confidence": 1.7}')
        c = IntentClassifier(llm=llm)
        result = c.classify("你好")
        assert result.confidence == 1.0

    def test_unparseable_response_falls_back_to_oos(self):
        llm = _make_llm("我也不知道这是个什么问题")
        c = IntentClassifier(llm=llm)
        result = c.classify("???")
        assert result.intent_id == INTENT_OUT_OF_SCOPE
        assert result.confidence == 0.0

    def test_unknown_intent_id_falls_back(self):
        llm = _make_llm('{"intent_id": "make_coffee", "confidence": 0.99}')
        c = IntentClassifier(llm=llm)
        result = c.classify("帮我冲杯咖啡")
        assert result.intent_id == INTENT_OUT_OF_SCOPE
        assert result.confidence == 0.0

    def test_empty_input_short_circuits(self):
        llm = _make_llm("不应该被调用")
        c = IntentClassifier(llm=llm)
        result = c.classify("")
        assert result.intent_id == INTENT_OUT_OF_SCOPE
        llm.invoke.assert_not_called()

    def test_llm_raises_falls_back_gracefully(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("connection reset")
        c = IntentClassifier(llm=llm)
        result = c.classify("怎么共享屏幕")
        assert result.intent_id == INTENT_OUT_OF_SCOPE
        assert result.confidence == 0.0


class TestPromptBuilding:
    def test_system_prompt_lists_all_intents(self):
        llm = _make_llm('{"intent_id": "greet", "confidence": 0.9}')
        c = IntentClassifier(llm=llm)
        c.classify("你好")
        prompt_text = llm.invoke.call_args.args[0][0].content
        for it in list_intents():
            assert it.id in prompt_text
