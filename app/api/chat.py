"""
FastAPI 路由 - chat 接口（Phase 4/6 升级版）。

新增字段
--------
ChatRequest.use_agent  bool   是否使用 Agent 模式（覆盖全局配置）
ChatResponse.product_id/product_name  产品线检测结果
ChatResponse.used_agent/agent_steps   Agent 模式调试信息

Phase 6 新增
------------
POST /api/chat/stream  SSE 流式输出，逐 token 推送
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_service import get_chat_service


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的消息", min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=None, description="客户端提供的历史对话")
    session_id: Optional[str] = Field(default=None, description="会话标识；不传则服务端生成")
    use_agent: bool = Field(default=False, description="是否使用 Agent 模式（覆盖全局配置）")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: float
    intent_id: Optional[str] = None
    intent_confidence: float = 0.0
    needs_followup: bool = False
    pending_slot: Optional[str] = None
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    used_agent: bool = False
    agent_steps: List[Any] = Field(default_factory=list)


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """处理用户消息：产品检测 + 意图识别 + RAG/Agent 生成。"""
    request_id = getattr(http_request.state, "request_id", None)
    history_override = (
        [{"role": m.role, "content": m.content} for m in request.history]
        if request.history is not None
        else None
    )

    try:
        result = await get_chat_service().chat(
            question=request.message,
            session_id=request.session_id,
            request_id=request_id,
            history_override=history_override,
            use_agent=request.use_agent,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成回答失败：{exc}") from exc

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        latency_ms=result.metrics.get("total_ms", 0.0),
        intent_id=result.intent_id,
        intent_confidence=result.intent_confidence,
        needs_followup=result.needs_followup,
        pending_slot=result.pending_slot,
        product_id=result.product_id,
        product_name=result.product_name,
        used_agent=result.used_agent,
        agent_steps=result.agent_steps,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> StreamingResponse:
    """SSE 流式响应（Phase 6）：逐 token 推送，前端用 fetch + ReadableStream 消费。

    消息格式（每行）：
      data: {"type":"chunk","content":"..."}\n\n
      data: {"type":"done","session_id":"...","product_id":"...",...}\n\n
      data: {"type":"error","message":"..."}\n\n
    """
    request_id = getattr(http_request.state, "request_id", None)
    history_override = (
        [{"role": m.role, "content": m.content} for m in request.history]
        if request.history is not None
        else None
    )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for data in get_chat_service().stream(
                question=request.message,
                session_id=request.session_id,
                request_id=request_id,
                history_override=history_override,
            ):
                yield f"data: {data}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲，保证实时推送
            "Connection": "keep-alive",
        },
    )
