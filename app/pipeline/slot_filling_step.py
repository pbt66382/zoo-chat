"""
SlotFillingStep：故障排查类意图的多轮追问。

执行模型
--------
1. 进入此 step 时 ``ctx.intent_id`` 已确定，``ctx.slots`` 是上轮已收集的槽位。
2. 如果当前正处于"等待用户回答某槽位"状态（``ctx.pending_slot`` 非空），
   把本轮 ``ctx.question`` 作为该槽位的回答塞进去，再清空 pending_slot。
3. 找下一个未填槽位：
   * 还有 → 设置 ``ctx.pending_slot``，置 ``answer`` 为追问话术，``needs_followup=True`` 短路返回。
   * 没有 → 把所有槽位拼进 ``ctx.question``（增强检索 query），让后续 step 继续走 RAG。
"""
from __future__ import annotations

import logging

from app.intent.intents import get_intent
from app.intent.slots import get_slot_schema, next_missing_slot
from app.pipeline.base import ChatContext

logger = logging.getLogger("zoo_chat.pipeline.slots")


class SlotFillingStep:
    name = "slot_filling"

    async def run(self, ctx: ChatContext) -> None:
        intent = get_intent(ctx.intent_id) if ctx.intent_id else None
        if not intent or not intent.needs_slots:
            return

        if not get_slot_schema(intent.id):
            return

        # 1. 用本轮回答填充上轮 pending_slot
        if ctx.pending_slot:
            ctx.slots[ctx.pending_slot] = ctx.question.strip()
            logger.info(
                "slot_filled request_id=%s intent=%s slot=%s value=%r",
                ctx.request_id,
                intent.id,
                ctx.pending_slot,
                ctx.slots[ctx.pending_slot],
            )
            ctx.pending_slot = None

        # 2. 检查下一个待填槽位
        missing = next_missing_slot(intent.id, ctx.slots)
        if missing is not None:
            ctx.pending_slot = missing.name
            ctx.answer = missing.question
            ctx.needs_followup = True
            logger.info(
                "slot_followup request_id=%s intent=%s next_slot=%s",
                ctx.request_id,
                intent.id,
                missing.name,
            )
            return

        # 3. 槽位齐备，把槽位信息拼进 query 增强后续检索
        slot_summary = "；".join(
            f"{name}={value}" for name, value in ctx.slots.items() if value
        )
        if slot_summary:
            ctx.question = f"{ctx.question}（用户补充：{slot_summary}）"
        logger.info(
            "slot_filling_complete request_id=%s intent=%s slots=%s",
            ctx.request_id,
            intent.id,
            ctx.slots,
        )
