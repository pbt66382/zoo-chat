"""
DeepSeek LLM 客户端封装模块。
使用 LangChain 的 ChatOpenAI（OpenAI 兼容接口）连接 DeepSeek。
"""
from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from config.settings import get_settings


def get_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **extra: Any,
) -> BaseChatModel:
    """
    获取配置好的 DeepSeek LLM 实例。

    DeepSeek 使用 OpenAI 兼容接口，因此使用 ChatOpenAI：
    - base_url 指向 DeepSeek API 端点
    - api_key 从 .env 读取
    - model 名称为 deepseek-chat

    参数 ``temperature`` / ``max_tokens`` 可针对特定场景覆盖（例如意图分类需要 0.0）。
    """
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=settings.max_tokens if max_tokens is None else max_tokens,
        **extra,
    )
    return llm


def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """
    格式化对话历史，供 Prompt 上下文使用。

    参数:
        messages: 消息列表，每个消息包含 'role' 和 'content'

    返回:
        格式化的对话历史字符串
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
