"""
Pipeline 编排器与 Router/Intent step 单元测试。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from app.intent.classifier import IntentResult
from app.pipeline.base import ChatContext, ChatPipeline, PipelineStep
from app.pipeline.intent_step import IntentStep
from app.pipeline.router_step import RouterStep


def _make_ctx(question: str = "你好") -> ChatContext:
    return ChatContext(request_id="r1", session_id="s1", question=question)


@dataclass
class _StaticStep:
    name: str
    set_answer: Optional[str] = None
    set_followup: bool = False
    raise_exc: Optional[Exception] = None
    called: bool = False

    async def run(self, ctx: ChatContext) -> None:
        self.called = True
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.set_answer is not None:
            ctx.answer = self.set_answer
        if self.set_followup:
            ctx.needs_followup = True


class _FakeClassifier:
    def __init__(self, intent_id: str, confidence: float = 0.9) -> None:
        self.intent_id = intent_id
        self.confidence = confidence
        self.calls = 0

    def classify(self, question: str) -> IntentResult:
        self.calls += 1
        return IntentResult(intent_id=self.intent_id, confidence=self.confidence)


@pytest.mark.asyncio
async def test_pipeline_runs_all_steps_in_order():
    s1 = _StaticStep(name="s1")
    s2 = _StaticStep(name="s2")
    s3 = _StaticStep(name="s3", set_answer="done")

    pipeline = ChatPipeline(steps=[s1, s2, s3])
    ctx = await pipeline.run(_make_ctx())

    assert s1.called and s2.called and s3.called
    assert ctx.answer == "done"
    assert "s1_ms" in ctx.metrics and "s2_ms" in ctx.metrics and "s3_ms" in ctx.metrics
    assert [t["step"] for t in ctx.trace] == ["s1", "s2", "s3"]


@pytest.mark.asyncio
async def test_pipeline_short_circuits_on_answer():
    s1 = _StaticStep(name="s1", set_answer="early answer")
    s2 = _StaticStep(name="s2")

    pipeline = ChatPipeline(steps=[s1, s2])
    ctx = await pipeline.run(_make_ctx())

    assert s1.called and not s2.called
    assert ctx.answer == "early answer"


@pytest.mark.asyncio
async def test_pipeline_short_circuits_on_followup():
    s1 = _StaticStep(name="s1", set_followup=True, set_answer="请告诉我设备型号")
    s2 = _StaticStep(name="s2")

    pipeline = ChatPipeline(steps=[s1, s2])
    ctx = await pipeline.run(_make_ctx())

    assert s1.called and not s2.called
    assert ctx.needs_followup is True


@pytest.mark.asyncio
async def test_pipeline_propagates_step_exception():
    s1 = _StaticStep(name="s1", raise_exc=RuntimeError("boom"))
    s2 = _StaticStep(name="s2")
    pipeline = ChatPipeline(steps=[s1, s2])

    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.run(_make_ctx())
    assert not s2.called


@pytest.mark.asyncio
async def test_intent_step_writes_ctx_fields():
    classifier = _FakeClassifier("screen_share", 0.92)
    step = IntentStep(classifier=classifier)
    ctx = _make_ctx("怎么共享屏幕")
    await step.run(ctx)
    assert ctx.intent_id == "screen_share"
    assert ctx.intent_confidence == pytest.approx(0.92)
    assert classifier.calls == 1


@pytest.mark.asyncio
async def test_intent_step_skipped_during_slot_filling():
    """处于槽位追问轮次时不应再次走分类。"""
    classifier = _FakeClassifier("greet", 0.99)
    step = IntentStep(classifier=classifier)
    ctx = _make_ctx("蓝牙耳机")
    ctx.intent_id = "troubleshoot_audio"
    ctx.pending_slot = "device_type"
    await step.run(ctx)
    assert classifier.calls == 0
    assert ctx.intent_id == "troubleshoot_audio"


@pytest.mark.asyncio
async def test_router_returns_auto_reply_for_greet():
    step = RouterStep()
    ctx = _make_ctx("你好")
    ctx.intent_id = "greet"
    ctx.intent_confidence = 0.95
    await step.run(ctx)
    assert ctx.answer is not None and "Zoo" in ctx.answer


@pytest.mark.asyncio
async def test_router_returns_auto_reply_for_oos():
    step = RouterStep()
    ctx = _make_ctx("天气怎么样")
    ctx.intent_id = "out_of_scope"
    ctx.intent_confidence = 0.9
    await step.run(ctx)
    assert ctx.answer is not None and "Zoo" in ctx.answer


@pytest.mark.asyncio
async def test_router_low_confidence_falls_back_to_oos():
    step = RouterStep(low_confidence_threshold=0.5)
    ctx = _make_ctx("我有点问题")
    ctx.intent_id = "screen_share"
    ctx.intent_confidence = 0.2
    await step.run(ctx)
    assert ctx.intent_id == "out_of_scope"
    assert ctx.answer is not None


@pytest.mark.asyncio
async def test_router_passes_through_for_high_confidence_intent():
    step = RouterStep()
    ctx = _make_ctx("怎么共享屏幕")
    ctx.intent_id = "screen_share"
    ctx.intent_confidence = 0.9
    await step.run(ctx)
    assert ctx.answer is None
    assert ctx.intent_id == "screen_share"
