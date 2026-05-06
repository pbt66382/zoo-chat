"""
FastAPI 路由模块 - 聊天 API 接口。
"""
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chains.faq_chain import invoke_faq_chain


# --- 请求/响应模型 ---

class ChatMessage(BaseModel):
    """单条聊天消息。"""
    role: str = Field(default="user", description="消息角色：'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """POST /chat 接口的请求体。"""
    message: str = Field(..., description="用户的消息", min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=None, description="聊天历史")
    session_id: Optional[str] = Field(default=None, description="会话标识符")


class ChatResponse(BaseModel):
    """POST /chat 接口的响应体。"""
    answer: str = Field(..., description="AI 助手的回答")
    session_id: str = Field(..., description="会话标识符")
    latency_ms: float = Field(..., description="响应时间（毫秒）")


# --- 路由定义 ---

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    处理用户消息，返回 AI 生成的回答。

    流程：
    1. 接收用户消息
    2. 调用 FAQ Chain（LLM + LangChain + DeepSeek）
    3. 返回生成的回答及元数据

    参数:
        request: 包含用户消息的 ChatRequest

    返回:
        包含 AI 回答和元数据的 ChatResponse
    """
    start_time = time.time()

    # 如果没有提供 session_id，生成一个
    session_id = request.session_id or f"session_{int(start_time * 1000)}"

    try:
        answer = invoke_faq_chain(request.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成回答失败：{str(e)}"
        )

    elapsed_ms = (time.time() - start_time) * 1000

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        latency_ms=round(elapsed_ms, 2),
    )
