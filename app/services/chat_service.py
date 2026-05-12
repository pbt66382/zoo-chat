"""
ChatService：编排 Pipeline、维护 Session、记录 RAG 日志。

职责拆分
--------
* API 层 (``app/api/chat.py``)：HTTP I/O、DTO 校验。
* 本服务层：构造 ``ChatContext``、注入 session 上下文、跑 pipeline、写日志。
* Pipeline 层 (``app/pipeline``)：纯业务 step。
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
from app.pipeline.retrieval_step import RetrievalStep
from app.pipeline.router_step import RouterStep
from app.pipeline.slot_filling_step import SlotFillingStep
from app.utils.rag_logger import RAGLog, log_rag_invocation

logger = logging.getLogger("zoo_chat.service")


@dataclass
class ChatResult:
    answer: str
    session_id: str
    intent_id: Optional[str]
    intent_confidence: float
    needs_followup: bool
    pending_slot: Optional[str]
    metrics: dict[str, float]
    trace: list[dict]


class ChatService:
    def __init__(self, pipeline: ChatPipeline | None = None) -> None:
        self._pipeline = pipeline or self._build_default_pipeline()

    @staticmethod
    def _build_default_pipeline() -> ChatPipeline:
        return ChatPipeline(steps=[
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
    ) -> ChatResult:
        store = get_session_store()
        sid = session_id or f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        session = store.get_or_create(sid)

        # history 优先用调用方显式传入的（兼容前端老协议），
        # 没传则使用服务端 session 维护的历史。
        history = history_override if history_override is not None else list(session.history)

        ctx = ChatContext(
            request_id=request_id or f"req_{uuid.uuid4().hex[:12]}",
            session_id=sid,
            question=question,
            history=history,
            intent_id=session.pending_intent,
            pending_slot=session.pending_slot,
            slots=dict(session.slots),
        )

        t_start = time.perf_counter()
        ctx = await self._pipeline.run(ctx)
        total_ms = (time.perf_counter() - t_start) * 1000
        ctx.metrics["total_ms"] = round(total_ms, 2)

        # 把对话写回 session（仅当我们没用调用方传入的 history 覆盖时）
        if history_override is None:
            session.append_message("user", question)
            if ctx.answer:
                session.append_message("assistant", ctx.answer)

        # 维护 session 的槽位状态
        if ctx.needs_followup:
            session.pending_intent = ctx.intent_id
            session.pending_slot = ctx.pending_slot
            session.slots = dict(ctx.slots)
        else:
            # 一轮完整问答结束 → 清空槽位上下文
            session.reset_slot_filling()

        # 记录 RAG 日志（只有真正走了检索 + 生成才有意义）
        if ctx.retrieved_docs and not ctx.needs_followup:
            self._record_rag_log(ctx, total_ms)

        return ChatResult(
            answer=ctx.answer or "",
            session_id=sid,
            intent_id=ctx.intent_id,
            intent_confidence=ctx.intent_confidence,
            needs_followup=ctx.needs_followup,
            pending_slot=ctx.pending_slot,
            metrics=ctx.metrics,
            trace=ctx.trace,
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
