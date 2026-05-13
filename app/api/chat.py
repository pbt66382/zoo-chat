"""
FastAPI 路由 - chat 接口（Phase 4 升级版）。

新增字段
--------
ChatRequest.use_agent  bool   是否使用 Agent 模式（覆盖全局配置）
ChatResponse.product_id/product_name  产品线检测结果
ChatResponse.used_agent/agent_steps   Agent 模式调试信息
"""
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
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
async def chat_stream(request: ChatRequest) -> JSONResponse:
    """SSE 流式响应（占位，待实现）。"""
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Streaming endpoint is not enabled in this build. Use POST /api/chat instead.",
            "hint": "GenerationStep.stream() exists; wire it to StreamingResponse when needed.",
        },
    )
