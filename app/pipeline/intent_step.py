"""
IntentStep：对当前 question 做意图分类，写入 ctx。

如果会话存在 ``pending_slot`` 上下文（上一轮槽位还没填齐），
本轮 question 视为对追问的回答而非新意图，跳过分类继承上轮意图。
"""
from __future__ import annotations

import asyncio
import logging

from app.intent.classifier import IntentClassifier
from app.pipeline.base import ChatContext

logger = logging.getLogger("zoo_chat.pipeline.intent")


class IntentStep:
    name = "intent"

    def __init__(self, classifier: IntentClassifier) -> None:
        self._classifier = classifier

    async def run(self, ctx: ChatContext) -> None:
        # 如果在槽位填充中（由 chat_service 把 session 的 pending_slot 灌进 ctx），
        # 不再做意图分类，沿用上轮意图。
        if ctx.pending_slot and ctx.intent_id:
            logger.debug(
                "intent_step_skip request_id=%s reason=slot_filling intent=%s slot=%s",
                ctx.request_id,
                ctx.intent_id,
                ctx.pending_slot,
            )
            return

        result = await asyncio.to_thread(self._classifier.classify, ctx.question)
        ctx.intent_id = result.intent_id
        ctx.intent_confidence = result.confidence
        logger.info(
            "intent_classified request_id=%s intent=%s confidence=%.2f",
            ctx.request_id,
            result.intent_id,
            result.confidence,
        )
