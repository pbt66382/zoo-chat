"""
DeepSeek LLM client wrapper.
Uses LangChain's ChatOpenAI with OpenAI-compatible interface to connect to DeepSeek.
"""
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI

from config.settings import get_settings


def get_llm() -> BaseChatModel:
    """
    Get a configured DeepSeek LLM instance.
    
    DeepSeek uses an OpenAI-compatible API, so we use ChatOpenAI with:
    - base_url pointing to DeepSeek API endpoint
    - api_key from .env
    - model name (deepseek-chat)
    
    Returns:
        Configured ChatOpenAI instance (compatible with BaseChatModel)
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    
    return llm


def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """
    Format chat history for inclusion in a prompt context.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Formatted string representation of the conversation history
    """
    if not messages:
        return "（无历史对话）"
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted.append(f"用户: {content}")
        elif role == "assistant":
            formatted.append(f"助手: {content}")
        elif role == "system":
            formatted.append(f"系统: {content}")
    
    return "\n".join(formatted)
