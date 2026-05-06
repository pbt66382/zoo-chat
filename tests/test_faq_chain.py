"""
Unit tests for the FAQ Chain.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.chains.faq_chain import (
    build_faq_chain,
    find_faq_by_question,
    FAQ_CONTEXT,
)
from data import FAQ_MEETINGS


class TestFAQData:
    """Test FAQ data is loaded correctly."""

    def test_faq_meetings_not_empty(self):
        assert len(FAQ_MEETINGS) >= 20, "Should have at least 20 FAQ entries"

    def test_faq_meetings_have_required_fields(self):
        for faq in FAQ_MEETINGS:
            assert "id" in faq
            assert "question" in faq
            assert "answer" in faq
            assert "tags" in faq

    def test_faq_context_is_not_empty(self):
        assert len(FAQ_CONTEXT) > 0
        assert "【问题" in FAQ_CONTEXT
        assert "【回答】" in FAQ_CONTEXT


class TestFindFAQByQuestion:
    """Test simple keyword-based FAQ lookup."""

    def test_find_by_screen_share_tag(self):
        result = find_faq_by_question("如何共享屏幕")
        assert result is not None
        assert "共享" in result["question"]

    def test_find_by_audio_tag(self):
        result = find_faq_by_question("会议中听不到声音")
        assert result is not None
        assert "audio" in result["tags"] or "声音" in result["question"]

    def test_find_by_meeting_create_tag(self):
        result = find_faq_by_question("meeting_create")
        assert result is not None
        assert result["id"] == 1

    def test_find_by_meeting_create_question(self):
        result = find_faq_by_question("新建")
        assert result is not None
        assert result["id"] == 1

    def test_find_by_speaker_tag(self):
        result = find_faq_by_question("speaker")
        assert result is not None
        assert result["id"] == 14

    def test_no_match_returns_none(self):
        result = find_faq_by_question("今天天气怎么样")
        # No FAQ about weather - should not crash


class TestBuildFAQChain:
    """Test FAQ chain construction."""

    def test_build_chain_returns_runnable(self):
        chain = build_faq_chain()
        assert chain is not None
        assert hasattr(chain, 'invoke')


class TestChainIntegration:
    """Integration-style tests using a mock LLM that mimics LangChain's LCEL interface.

    LCEL (LangChain Expression Language) calls LLM via __call__ internally
    (via the | pipe operator), not via .invoke() directly. The chain pipeline
    is: dict -> prompt -> LLM -> StrOutputParser -> str.

    So the mock needs to:
    1. Return a proper object when called as a callable (mimics LLM.__call__)
    2. That object needs .content attribute (mimics AIMessage)
    """

    def _make_mock_llm(self, response_text: str):
        """Create a mock LLM that returns the given text as an AIMessage-like response."""
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock()
        # LCEL pipe operator calls the LLM as a callable
        mock_llm.return_value = AIMessage(content=response_text)
        return mock_llm

    def test_chain_invoke_returns_string(self):
        """Test that the full chain returns a string when LLM returns an AIMessage."""
        mock_llm = self._make_mock_llm("测试回答")

        with patch('app.chains.faq_chain.get_llm', return_value=mock_llm):
            chain = build_faq_chain()
            result = chain.invoke({
                "faq_context": FAQ_CONTEXT,
                "question": "如何创建会议",
            })

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_with_screen_share_question(self):
        """Test chain handles a screen share question."""
        mock_llm = self._make_mock_llm("要共享屏幕，请点击共享按钮...")

        with patch('app.chains.faq_chain.get_llm', return_value=mock_llm):
            from app.chains.faq_chain import invoke_faq_chain
            result = invoke_faq_chain("如何共享屏幕")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_rejects_out_of_scope(self):
        """Test that out-of-scope questions get a rejection response from the chain."""
        mock_llm = self._make_mock_llm("抱歉，这个问题我暂时无法回答")

        with patch('app.chains.faq_chain.get_llm', return_value=mock_llm):
            from app.chains.faq_chain import invoke_faq_chain
            result = invoke_faq_chain("今天天气怎么样")

        assert isinstance(result, str)
        assert len(result) > 0
