"""
RouterStep：根据意图分类结果决定下一步走向。

逻辑
----
* ``needs_retrieval=False`` 的意图（greet / out_of_scope）：
  直接置 ``ctx.answer = auto_reply``，后续 step 短路。
* 置信度过低（< LOW_CONFIDENCE_THRESHOLD）：
  视作 out_of_scope，礼貌兜底。
* 其他情况（普通意图 + 故障类意图）：
  什么都不做，让后续 step（SlotFillingStep / RetrievalStep / GenerationStep）继续处理。
"""
from __future__ import annotations

import logging

from app.intent.intents import INTENT_OUT_OF_SCOPE, get_intent
from app.pipeline.base import ChatContext

logger = logging.getLogger("zoo_chat.pipeline.router")

# 低于此阈值视为不可信，走 out_of_scope 兜底回复。
LOW_CONFIDENCE_THRESHOLD = 0.45


class RouterStep:
    name = "router"

    def __init__(self, low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = low_confidence_threshold

    async def run(self, ctx: ChatContext) -> None:
        intent_id = ctx.intent_id or INTENT_OUT_OF_SCOPE
        intent = get_intent(intent_id)

        # 槽位填充流程中：跳过路由判断，让 slot_filling_step 主导
        if ctx.pending_slot:
            logger.debug(
                "router_skip request_id=%s reason=slot_filling intent=%s slot=%s",
                ctx.request_id,
                intent_id,
                ctx.pending_slot,
            )
            return

        # 置信度兜底
        if intent and intent.needs_retrieval and ctx.intent_confidence < self._threshold:
            logger.info(
                "router_low_confidence request_id=%s intent=%s confidence=%.2f -> oos",
                ctx.request_id,
                intent_id,
                ctx.intent_confidence,
            )
            intent = get_intent(INTENT_OUT_OF_SCOPE)
            ctx.intent_id = INTENT_OUT_OF_SCOPE

        # 不需要检索的意图直接给固定回复
        if intent and not intent.needs_retrieval:
            ctx.answer = intent.auto_reply or "好的。"
            logger.info(
                "router_auto_reply request_id=%s intent=%s",
                ctx.request_id,
                intent.id,
            )
