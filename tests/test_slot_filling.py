"""
SlotFillingStep 与 ChatService 多轮槽位追问测试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from app.intent.classifier import IntentResult
from app.memory.session import SessionStore
from app.pipeline.base import ChatContext, ChatPipeline
from app.pipeline.intent_step import IntentStep
from app.pipeline.router_step import RouterStep
from app.pipeline.slot_filling_step import SlotFillingStep


def _make_ctx(question: str, **kwargs) -> ChatContext:
    return ChatContext(request_id="r", session_id="s", question=question, **kwargs)


class _FakeClassifier:
    def __init__(self, intent_id: str, confidence: float = 0.9) -> None:
        self._intent_id = intent_id
        self._confidence = confidence

    def classify(self, question: str) -> IntentResult:
        return IntentResult(intent_id=self._intent_id, confidence=self._confidence)


@pytest.mark.asyncio
async def test_slot_filling_first_turn_asks_for_device():
    step = SlotFillingStep()
    ctx = _make_ctx("对方听不到我说话")
    ctx.intent_id = "troubleshoot_audio"
    await step.run(ctx)

    assert ctx.needs_followup is True
    assert ctx.pending_slot == "device_type"
    assert ctx.answer is not None and "音频设备" in ctx.answer


@pytest.mark.asyncio
async def test_slot_filling_second_turn_asks_for_scenario():
    step = SlotFillingStep()
    ctx = _make_ctx("蓝牙耳机")
    ctx.intent_id = "troubleshoot_audio"
    ctx.pending_slot = "device_type"
    await step.run(ctx)

    assert ctx.slots.get("device_type") == "蓝牙耳机"
    assert ctx.pending_slot == "scenario"
    assert ctx.needs_followup is True


@pytest.mark.asyncio
async def test_slot_filling_complete_enriches_question():
    step = SlotFillingStep()
    ctx = _make_ctx("会议中途突然没声音")
    ctx.intent_id = "troubleshoot_audio"
    ctx.pending_slot = "scenario"
    ctx.slots = {"device_type": "蓝牙耳机"}
    await step.run(ctx)

    assert ctx.needs_followup is False
    assert ctx.pending_slot is None
    assert "device_type=蓝牙耳机" in ctx.question
    assert "scenario=" in ctx.question


@pytest.mark.asyncio
async def test_slot_filling_skipped_for_non_slot_intent():
    step = SlotFillingStep()
    ctx = _make_ctx("怎么共享屏幕")
    ctx.intent_id = "screen_share"
    await step.run(ctx)
    assert ctx.needs_followup is False
    assert ctx.pending_slot is None
    assert ctx.slots == {}


@pytest.mark.asyncio
async def test_full_pipeline_followup_flow_end_to_end():
    """端到端：意图 → 路由 → 槽位追问 → 短路返回。"""
    intent_step = IntentStep(classifier=_FakeClassifier("troubleshoot_audio", 0.95))
    router_step = RouterStep()
    slot_step = SlotFillingStep()
    pipeline = ChatPipeline(steps=[intent_step, router_step, slot_step])

    ctx = await pipeline.run(_make_ctx("对方听不到我说话"))

    assert ctx.intent_id == "troubleshoot_audio"
    assert ctx.needs_followup is True
    assert ctx.pending_slot == "device_type"
    assert ctx.answer is not None


@pytest.mark.asyncio
async def test_chat_service_persists_slot_state_across_turns(monkeypatch):
    """通过两次 chat_service.chat 调用验证 session 槽位会被 round-trip。"""
    from app.services import chat_service as svc_module

    # 用独立 SessionStore 隔离测试
    test_store = SessionStore()
    monkeypatch.setattr(svc_module, "get_session_store", lambda: test_store)

    fake_intent_step = IntentStep(classifier=_FakeClassifier("troubleshoot_video", 0.9))

    # GenerationStep 在追问轮次不会被调用，但齐备后会被调到，先 mock 掉
    gen_step = AsyncMock()
    gen_step.name = "generation"
    async def _gen(ctx):
        ctx.answer = "请重启 Zoo 客户端并检查摄像头权限。"
    gen_step.run.side_effect = _gen

    # 检索也 mock，避免触发 Milvus
    retr_step = AsyncMock()
    retr_step.name = "retrieval"
    async def _retr(ctx):
        ctx.retrieved_docs = []
    retr_step.run.side_effect = _retr

    pipeline = ChatPipeline(steps=[
        fake_intent_step,
        RouterStep(),
        SlotFillingStep(),
        retr_step,
        gen_step,
    ])
    service = svc_module.ChatService(pipeline=pipeline)

    # 第一轮：触发槽位追问
    r1 = await service.chat(question="摄像头打不开", session_id="t-session")
    assert r1.needs_followup is True
    assert r1.pending_slot == "device_type"
    assert "摄像头" in r1.answer

    # 第二轮：填 device_type，应继续追问 scenario
    r2 = await service.chat(question="笔记本内置摄像头", session_id="t-session")
    assert r2.needs_followup is True
    assert r2.pending_slot == "scenario"

    # 第三轮：填 scenario，槽位齐备 → 走 generation 给最终回答
    r3 = await service.chat(question="完全打不开", session_id="t-session")
    assert r3.needs_followup is False
    assert r3.pending_slot is None
    assert "请重启" in r3.answer
    # 槽位状态应当被清空
    session = test_store.get("t-session")
    assert session is not None
    assert session.pending_slot is None
    assert session.slots == {}
