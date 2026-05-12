"""
意图定义加载器：从 ``data/intents.json`` 读取意图清单。

每个意图除了 id/name/description/examples 外，还携带：
* ``needs_retrieval``：是否要走 RAG 检索；False 时直接用 ``auto_reply``。
* ``needs_slots``：是否需要进入槽位填充（仅故障排查类意图为 True）。
* ``auto_reply``：当 ``needs_retrieval=False`` 时使用的固定回复。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

INTENT_OUT_OF_SCOPE = "out_of_scope"

_INTENTS_PATH = Path(__file__).parent.parent.parent / "data" / "intents.json"


@dataclass(frozen=True)
class IntentDef:
    id: str
    name: str
    description: str
    examples: tuple[str, ...]
    needs_retrieval: bool = True
    needs_slots: bool = False
    auto_reply: Optional[str] = None


@lru_cache(maxsize=1)
def load_intents() -> tuple[IntentDef, ...]:
    with _INTENTS_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("intents", [])
    return tuple(
        IntentDef(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            examples=tuple(item.get("examples", [])),
            needs_retrieval=bool(item.get("needs_retrieval", True)),
            needs_slots=bool(item.get("needs_slots", False)),
            auto_reply=item.get("auto_reply"),
        )
        for item in items
    )


def list_intents() -> list[IntentDef]:
    return list(load_intents())


@lru_cache(maxsize=64)
def get_intent(intent_id: str) -> Optional[IntentDef]:
    for intent in load_intents():
        if intent.id == intent_id:
            return intent
    return None
