"""意图体系：定义、加载、分类与槽位。"""
from app.intent.intents import (
    INTENT_OUT_OF_SCOPE,
    IntentDef,
    get_intent,
    list_intents,
    load_intents,
)

__all__ = [
    "INTENT_OUT_OF_SCOPE",
    "IntentDef",
    "get_intent",
    "list_intents",
    "load_intents",
]
