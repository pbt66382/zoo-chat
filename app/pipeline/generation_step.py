"""
GenerationStep：基于 ctx.retrieved_docs 与 ctx.history 让 LLM 生成最终回答。

* 提供同步 ``run()`` 路径（默认 chat 接口走这里）。
* 预留 ``stream()`` 路径，调 ``llm.astream()`` 逐 token 产出 chunk。
* 提示词包含意图名（如果有），让 LLM 更聚焦答案。
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.intent.intents import get_intent
from app.llm.deepseek_client import get_llm
from app.pipeline.base import ChatContext

logger = logging.getLogger("zoo_chat.pipeline.generation")


SYSTEM_PROMPT = """你是一个专业、友好的 Zoo 会议服务客服助手。{intent_hint}

以下是与用户问题最相关的 FAQ 参考内容：
{context}

请基于以上 FAQ 内容回答用户问题，遵守以下原则：
1. 优先采用 FAQ 中的事实，不要编造未提及的功能或步骤。
2. 用通俗、亲切的中文回答，控制在 200 字以内（步骤类可以适度展开）。
3. 如果 FAQ 不足以回答，请明确告诉用户"我目前的资料不足以解答这个问题"，并建议联系人工客服。
4. 不要在回答里出现"FAQ"、"参考内容"等内部术语。

{history_block}"""


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "（暂无相关 FAQ）"
    return "\n".join(
        f"【FAQ {d.metadata.get('faq_id', '?')}】{d.page_content}" for d in docs
    )


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "【历史对话】（无）"
    lines = ["【历史对话】"]
    for msg in history[-10:]:  # 只取最近 10 条避免 prompt 过长
        role = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _build_messages(ctx: ChatContext) -> list:
    intent = get_intent(ctx.intent_id) if ctx.intent_id else None
    intent_hint = f"当前用户意图判定为「{intent.name}」。" if intent else ""
    system_text = SYSTEM_PROMPT.format(
        intent_hint=intent_hint,
        context=_format_context(ctx.retrieved_docs),
        history_block=_format_history(ctx.history),
    )
    return [SystemMessage(content=system_text), HumanMessage(content=ctx.question)]


class GenerationStep:
    name = "generation"

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _invoke_sync(self, messages: list) -> str:
        response = self._get_llm().invoke(messages)
        return getattr(response, "content", str(response)) or ""

    async def run(self, ctx: ChatContext) -> None:
        messages = _build_messages(ctx)
        answer = await asyncio.to_thread(self._invoke_sync, messages)
        ctx.answer = answer
        logger.info(
            "generation_done request_id=%s answer_len=%d",
            ctx.request_id,
            len(answer),
        )

    async def stream(self, ctx: ChatContext) -> AsyncIterator[str]:
        """
        异步逐 chunk 输出（SSE 用）。

        本期暂不被路由调用；预留接口让上层可以无侵入接 streaming。
        """
        messages = _build_messages(ctx)
        async for chunk in self._get_llm().astream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text
