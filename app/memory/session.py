"""
内存版 SessionStore：以 session_id 为 key 维护对话历史 + 槽位状态。

设计取舍
--------
* 使用进程内 dict 而不是 Redis：本期目标是把多轮对话跑通，不引入额外依赖。
  Phase 4/5 如果上量再换 Redis，``SessionStore`` 接口保持不变即可。
* 故意不引入 ``langchain.memory.ConversationBufferMemory``：它的 API 较重
  且默认强绑 LLM。这里我们只需要"按角色追加 + 取最近 N 条"的能力，自己
  维护 ``list[dict]`` 反而更清晰可测。
* 加入超时清理：避免长期运行时内存膨胀。

并发说明：FastAPI 默认在 asyncio event loop 中处理请求；由于 ``dict`` 的
单条赋值/读取在 CPython 中是原子操作，本期不显式加锁；多 worker 部署时
请改为 Redis 实现。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

# 默认会话存活时长（秒）。每次访问会刷新 last_seen。
DEFAULT_TTL_SEC = 60 * 60  # 1 小时
MAX_HISTORY_TURNS = 20


@dataclass
class SessionState:
    """单个会话的状态快照。"""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # 历史消息：[{"role": "user"/"assistant", "content": "..."}, ...]
    history: list[dict[str, str]] = field(default_factory=list)

    # 跨轮槽位填充：意图与已收集的槽位
    pending_intent: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    pending_slot: str | None = None

    def touch(self) -> None:
        self.last_seen = time.time()

    def append_message(self, role: str, content: str) -> None:
        if not content:
            return
        self.history.append({"role": role, "content": content})
        # 限制历史长度，按"轮"裁剪（user+assistant 一轮）
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-MAX_HISTORY_TURNS * 2 :]
        self.touch()

    def reset_slot_filling(self) -> None:
        """槽位齐备/重新进入新意图时清空槽位上下文。"""
        self.pending_intent = None
        self.slots = {}
        self.pending_slot = None

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.last_seen) > ttl


class SessionStore:
    """进程内 session 仓库。"""

    def __init__(self, ttl: float = DEFAULT_TTL_SEC) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._ttl = ttl

    def get_or_create(self, session_id: str) -> SessionState:
        self._gc()
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
        else:
            state.touch()
        return state

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def size(self) -> int:
        return len(self._sessions)

    def _gc(self) -> None:
        """惰性清理过期会话；调用频率 = 请求频率，对小流量足够。"""
        if not self._sessions:
            return
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(self._ttl)]
        for sid in expired:
            self._sessions.pop(sid, None)


_global_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """单例入口，方便依赖注入与测试时替换。"""
    global _global_store
    if _global_store is None:
        _global_store = SessionStore()
    return _global_store
