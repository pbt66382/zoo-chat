"""
FastAPI 路由 - chat 接口。

职责仅限于 HTTP I/O：
* 校验 / 解析请求体
* 调 ``ChatService.chat()`` 拿结果
* 转 DTO 返回，并在 500 时给出可读 detail
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.chat_service import get_chat_service


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="消息角色：'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户的消息", min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=None, description="可选：客户端提供的历史对话")
    session_id: Optional[str] = Field(default=None, description="会话标识；不传则服务端生成")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    latency_ms: float
    intent_id: Optional[str] = None
    intent_confidence: float = 0.0
    needs_followup: bool = False
    pending_slot: Optional[str] = None


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    """处理用户消息：意图识别 + 槽位追问 + RAG 生成。"""
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
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成回答失败：{exc}") from exc

    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        latency_ms=result.metrics.get("total_ms", 0.0),
        intent_id=result.intent_id,
        intent_confidence=result.intent_confidence,
        needs_followup=result.needs_followup,
        pending_slot=result.pending_slot,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> JSONResponse:
    """
    SSE 流式响应（占位）。

    GenerationStep 已经实现了 ``stream()`` 接口；本期 API 层暂未对接 SSE，
    返回 501 提醒前端走非流式接口。
    """
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Streaming endpoint is not enabled in this build. Use POST /api/chat instead.",
            "hint": "GenerationStep.stream() exists; wire it to StreamingResponse when needed.",
        },
    )
