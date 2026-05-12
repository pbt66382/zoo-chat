"""服务层：编排 pipeline，给 API 层提供干净的入口。"""
from app.services.chat_service import ChatResult, ChatService, get_chat_service

__all__ = ["ChatResult", "ChatService", "get_chat_service"]
