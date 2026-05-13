"""
Pipeline 基础设施：上下文、Step 协议、编排器。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.documents import Document

logger = logging.getLogger("zoo_chat.pipeline")


@dataclass
class ChatContext:
    """单次 chat 请求贯穿全流程的可变上下文。

    Step 之间不直接传参，全部读写此对象，方便插拔与日志聚合。
    """

    request_id: str
    session_id: str
    question: str
    history: list[dict[str, str]] = field(default_factory=list)

    # 意图识别结果
    intent_id: str | None = None
    intent_confidence: float = 0.0

    # 槽位状态
    slots: dict[str, str] = field(default_factory=dict)
    pending_slot: str | None = None
    needs_followup: bool = False

    # 检索结果
    retrieved_docs: list[Document] = field(default_factory=list)

    # 最终回答
    answer: str | None = None

    # Phase 4：产品线信息（由 ProductDetectionStep 填写）
    product_id: str | None = None
    product_name: str | None = None
    product_confidence: float = 0.0
    milvus_collection: str | None = None

    # Phase 4：Agent 模式标记
    used_agent: bool = False
    agent_steps: list[dict[str, Any]] = field(default_factory=list)

    # 观测：每步耗时与输入/输出快照
    metrics: dict[str, float] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def short_circuit(self) -> bool:
        """当存在最终回答或需要追问时短路后续 step。"""
        return self.answer is not None or self.needs_followup


class PipelineStep(Protocol):
    """
    Step 协议：拥有 ``name`` 用于日志/指标命名，``run`` 修改 ctx。

    需要支持同步调用的 step 可直接定义 ``async def run``，
    内部用 ``await asyncio.to_thread(...)`` 包同步代码即可。
    """

    name: str

    async def run(self, ctx: ChatContext) -> None:  # pragma: no cover - protocol
        ...


class ChatPipeline:
    """按顺序执行 Step，自动收集每步耗时与短路日志。"""

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps

    async def run(self, ctx: ChatContext) -> ChatContext:
        for step in self.steps:
            if ctx.short_circuit():
                logger.debug(
                    "pipeline_short_circuit request_id=%s before_step=%s answer_set=%s followup=%s",
                    ctx.request_id,
                    step.name,
                    ctx.answer is not None,
                    ctx.needs_followup,
                )
                break

            t0 = time.perf_counter()
            try:
                await step.run(ctx)
            except Exception:
                logger.exception(
                    "pipeline_step_failed request_id=%s step=%s",
                    ctx.request_id,
                    step.name,
                )
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ctx.metrics[f"{step.name}_ms"] = round(elapsed_ms, 2)

            ctx.trace.append({
                "step": step.name,
                "elapsed_ms": round(elapsed_ms, 2),
                "intent_id": ctx.intent_id,
                "intent_confidence": ctx.intent_confidence,
                "pending_slot": ctx.pending_slot,
                "needs_followup": ctx.needs_followup,
                "answer_preview": (ctx.answer or "")[:80] if ctx.answer else None,
            })

        return ctx
