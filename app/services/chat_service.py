"""
ChatService（Phase 4 升级版）：编排 Pipeline、维护 Session、Agent 模式分流。

两种执行模式
-----------
Pipeline 模式（默认）：固定步骤顺序执行
  ProductDetection → Intent → Router → SlotFilling → Retrieval → Generation

Agent 模式（AGENT_MODE_ENABLED=true 或请求参数 use_agent=true）：
  Agent 自主决定调用哪些工具，支持多步推理和动态分支。

产品线持久化
-----------
同一 session 内只做一次产品检测，结果存到 SessionState，后续轮次直接复用。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.intent.classifier import get_classifier
from app.memory.session import SessionState, get_session_store
from app.pipeline.base import ChatContext, ChatPipeline
from app.pipeline.generation_step import GenerationStep
from app.pipeline.intent_step import IntentStep
from app.pipeline.product_detection_step import ProductDetectionStep
from app.pipeline.retrieval_step import RetrievalStep
from app.pipeline.router_step import RouterStep
from app.pipeline.slot_filling_step import SlotFillingStep
from app.utils.rag_logger import RAGLog, log_rag_invocation
from config.settings import get_settings

logger = logging.getLogger("zoo_chat.service")


@dataclass
class ChatResult:
    answer: str
    session_id: str
    intent_id: Optional[str]
    intent_confidence: float
    needs_followup: bool
    pending_slot: Optional[str]
    product_id: Optional[str]
    product_name: Optional[str]
    used_agent: bool
    agent_steps: list[dict]
    metrics: dict[str, float]
    trace: list[dict]


class ChatService:
    def __init__(self, pipeline: ChatPipeline | None = None) -> None:
        self._pipeline = pipeline or self._build_default_pipeline()

    @staticmethod
    def _build_default_pipeline() -> ChatPipeline:
        return ChatPipeline(steps=[
            ProductDetectionStep(),
            IntentStep(classifier=get_classifier()),
            RouterStep(),
            SlotFillingStep(),
            RetrievalStep(),
            GenerationStep(),
        ])

    async def chat(
        self,
        question: str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        history_override: Optional[list[dict[str, str]]] = None,
        use_agent: bool = False,
    ) -> ChatResult:
        settings = get_settings()
        store = get_session_store()
        sid = session_id or f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        session = store.get_or_create(sid)

        history = history_override if history_override is not None else list(session.history)

        # 决定是否使用 Agent 模式
        run_agent = use_agent or settings.agent_mode_enabled

        if run_agent:
            return await self._run_agent(question, session, sid, request_id, history, history_override)
        return await self._run_pipeline(question, session, sid, request_id, history, history_override)

    async def _run_pipeline(
        self,
        question: str,
        session: SessionState,
        sid: str,
        request_id: Optional[str],
        history: list[dict[str, str]],
        history_override: Optional[list[dict[str, str]]],
    ) -> ChatResult:
        ctx = ChatContext(
            request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
            session_id=sid,
            question=question,
            history=history,
            intent_id=session.pending_intent,
            pending_slot=session.pending_slot,
            slots=dict(session.slots),
            # 从 session 恢复产品线（避免每轮重新检测）
            product_id=session.product_id,
            milvus_collection=session.milvus_collection,
        )

        t_start = time.perf_counter()
        ctx = await self._pipeline.run(ctx)
        total_ms = (time.perf_counter() - t_start) * 1000
        ctx.metrics["total_ms"] = round(total_ms, 2)

        # 更新 session
        if history_override is None:
            session.append_message("user", question)
            if ctx.answer:
                session.append_message("assistant", ctx.answer)

        if ctx.needs_followup:
            session.pending_intent = ctx.intent_id
            session.pending_slot = ctx.pending_slot
            session.slots = dict(ctx.slots)
        else:
            session.reset_slot_filling()

        # 持久化产品线到 session
        if ctx.product_id:
            session.product_id = ctx.product_id
            session.milvus_collection = ctx.milvus_collection

        if ctx.retrieved_docs and not ctx.needs_followup:
            self._record_rag_log(ctx, total_ms)

        return ChatResult(
            answer=ctx.answer or "",
            session_id=sid,
            intent_id=ctx.intent_id,
            intent_confidence=ctx.intent_confidence,
            needs_followup=ctx.needs_followup,
            pending_slot=ctx.pending_slot,
            product_id=ctx.product_id,
            product_name=ctx.product_name,
            used_agent=False,
            agent_steps=[],
            metrics=ctx.metrics,
            trace=ctx.trace,
        )

    async def _run_agent(
        self,
        question: str,
        session: SessionState,
        sid: str,
        request_id: Optional[str],
        history: list[dict[str, str]],
        history_override: Optional[list[dict[str, str]]],
    ) -> ChatResult:
        import asyncio
        from app.agent.zoo_agent import ZooAgent
        from app.product.detector import get_product_detector, PRODUCT_LINES

        settings = get_settings()

        # 产品线检测（session 内复用）
        if session.product_id and session.milvus_collection:
            product_id = session.product_id
            collection = session.milvus_collection
            product_name = PRODUCT_LINES.get(product_id, {}).get("name", "通用")
        else:
            result = await asyncio.to_thread(get_product_detector().detect, question)
            product_id = result.product_id
            collection = result.collection
            product_name = result.product_name
            session.product_id = product_id
            session.milvus_collection = collection

        agent = ZooAgent(product_name=product_name, collection=collection)
        agent_result = await asyncio.to_thread(agent.chat, question, history)

        # 更新 session
        if history_override is None:
            session.append_message("user", question)
            if agent_result.answer:
                session.append_message("assistant", agent_result.answer)

        return ChatResult(
            answer=agent_result.answer,
            session_id=sid,
            intent_id=None,
            intent_confidence=0.0,
            needs_followup=False,
            pending_slot=None,
            product_id=product_id,
            product_name=product_name,
            used_agent=True,
            agent_steps=agent_result.steps,
            metrics={"total_ms": agent_result.total_ms, "tool_calls": float(agent_result.tool_calls)},
            trace=[],
        )

    @staticmethod
    def _record_rag_log(ctx: ChatContext, total_ms: float) -> None:
        scores = [d.metadata.get("score", 0.0) for d in ctx.retrieved_docs]
        top_score = scores[0] if scores else 0.0
        score_gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        try:
            log_rag_invocation(RAGLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                request_id=ctx.request_id,
                question=ctx.question,
                history="\n".join(f"{m.get('role')}: {m.get('content')}" for m in ctx.history) or "（无）",
                recalled_faqs=[
                    {
                        "faq_id": d.metadata.get("faq_id", "?"),
                        "score": round(d.metadata.get("score", 0.0), 4),
                        "tags": d.metadata.get("tags", ""),
                    }
                    for d in ctx.retrieved_docs
                ],
                retrieval_latency_ms=ctx.metrics.get("retrieval_only_ms", 0.0),
                embedding_latency_ms=ctx.metrics.get("embedding_ms", 0.0),
                total_latency_ms=round(total_ms, 2),
                llm_answer_length=len(ctx.answer or ""),
                top_score=round(top_score, 4),
                score_gap=round(score_gap, 4),
            ))
        except Exception:
            logger.exception("rag_log_record_failed request_id=%s", ctx.request_id)


_default_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _default_service
    if _default_service is None:
        _default_service = ChatService()
    return _default_service
