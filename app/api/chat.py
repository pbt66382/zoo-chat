"""
FastAPI route handlers for the chat API.
"""
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chains.faq_chain import invoke_faq_chain


# --- Request/Response Models ---

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(default="user", description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    message: str = Field(..., description="User's message", min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=None, description="Chat history")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""
    answer: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session identifier")
    latency_ms: float = Field(..., description="Response time in milliseconds")


# --- Router ---

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle a chat message and return an AI-generated response.
    
    This endpoint:
    1. Accepts a user message
    2. Invokes the FAQ Chain (LLM + LangChain + DeepSeek)
    3. Returns the generated response with metadata
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        ChatResponse with the AI's answer and metadata
    """
    start_time = time.time()
    
    # Generate a session ID if not provided
    session_id = request.session_id or f"session_{int(start_time * 1000)}"
    
    try:
        # Invoke the FAQ chain to get the answer
        answer = invoke_faq_chain(request.message)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        latency_ms=round(elapsed_ms, 2),
    )
