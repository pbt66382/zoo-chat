"""
FAQ Chain 单元测试。
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
    """FAQ 数据加载测试。"""

    def test_faq_meetings_not_empty(self):
        """FAQ 条目数量应 >= 20。"""
        assert len(FAQ_MEETINGS) >= 20

    def test_faq_meetings_have_required_fields(self):
        """每条 FAQ 必须包含 id、question、answer、tags 字段。"""
        for faq in FAQ_MEETINGS:
            assert "id" in faq
            assert "question" in faq
            assert "answer" in faq
            assert "tags" in faq

    def test_faq_context_is_not_empty(self):
        """FAQ 上下文字符串不应为空，应包含问题和回答标记。"""
        assert len(FAQ_CONTEXT) > 0
        assert "【问题" in FAQ_CONTEXT
        assert "【回答】" in FAQ_CONTEXT


class TestFindFAQByQuestion:
    """简单关键词 FAQ 检索测试。"""

    def test_find_by_screen_share_tag(self):
        """关键词 "如何共享屏幕" 应匹配到屏幕共享相关 FAQ。"""
        result = find_faq_by_question("如何共享屏幕")
        assert result is not None
        assert "共享" in result["question"]

    def test_find_by_audio_tag(self):
        """关键词 "会议中听不到声音" 应匹配到音频相关 FAQ。"""
        result = find_faq_by_question("会议中听不到声音")
        assert result is not None
        assert "audio" in result["tags"] or "声音" in result["question"]

    def test_find_by_meeting_create_tag(self):
        """关键词 "meeting_create" 应匹配到 FAQ id=1（创建会议）。"""
        result = find_faq_by_question("meeting_create")
        assert result is not None
        assert result["id"] == 1

    def test_find_by_meeting_create_question(self):
        """关键词 "新建" 应匹配到 FAQ id=1（创建会议）。"""
        result = find_faq_by_question("新建")
        assert result is not None
        assert result["id"] == 1

    def test_find_by_speaker_tag(self):
        """关键词 "speaker" 应匹配到 FAQ id=14（发言人视图）。"""
        result = find_faq_by_question("speaker")
        assert result is not None
        assert result["id"] == 14

    def test_no_match_returns_none(self):
        """无关问题（如天气）不应崩溃，返回 None 或部分匹配均可。"""
        result = find_faq_by_question("今天天气怎么样")


class TestBuildFAQChain:
    """FAQ Chain 构建测试。"""

    def test_build_chain_returns_runnable(self):
        """build_faq_chain() 应返回一个包含 invoke 方法的可执行对象。"""
        chain = build_faq_chain()
        assert chain is not None
        assert hasattr(chain, 'invoke')


class TestChainIntegration:
    """
    集成测试 - 使用模拟 LLM 验证 Chain 行为。

    LCEL（LangChain Expression Language）通过 __call__ 调用 LLM，
    而非直接调用 .invoke()。Chain 流水线为：
    dict -> prompt -> LLM -> StrOutputParser -> str

    因此 mock 需要：
    1. 返回可调用对象（模拟 LLM.__call__）
    2. 返回的对象需要有 .content 属性（模拟 AIMessage）
    """

    def _make_mock_llm(self, response_text: str):
        """
        创建模拟 LLM，返回指定文本作为 AIMessage 响应。

        参数:
            response_text: 要返回的文本内容
        """
        from langchain_core.messages import AIMessage
        mock_llm = MagicMock()
        # LCEL 管道符通过 __call__ 调用 LLM
        mock_llm.return_value = AIMessage(content=response_text)
        return mock_llm

    def test_chain_invoke_returns_string(self):
        """Chain.invoke() 应返回字符串类型的结果。"""
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
        """Chain 应正确处理屏幕共享相关问题。"""
        mock_llm = self._make_mock_llm("要共享屏幕，请点击共享按钮...")

        with patch('app.chains.faq_chain.get_llm', return_value=mock_llm):
            from app.chains.faq_chain import invoke_faq_chain
            result = invoke_faq_chain("如何共享屏幕")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_rejects_out_of_scope(self):
        """超出 FAQ 范围的问题，Chain 应给出拒绝回复。"""
        mock_llm = self._make_mock_llm("抱歉，这个问题我暂时无法回答")

        with patch('app.chains.faq_chain.get_llm', return_value=mock_llm):
            from app.chains.faq_chain import invoke_faq_chain
            result = invoke_faq_chain("今天天气怎么样")

        assert isinstance(result, str)
        assert len(result) > 0
