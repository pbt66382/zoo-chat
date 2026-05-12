"""
RAG 调用结构化日志：将每次问答的召回与耗时写入 YAML 文件。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

import yaml

from config.settings import get_settings


@dataclass
class RAGLog:
    timestamp: str
    request_id: str
    question: str
    history: str
    recalled_faqs: List[dict[str, Any]]
    retrieval_latency_ms: float
    embedding_latency_ms: float
    total_latency_ms: float
    llm_answer_length: int
    top_score: float
    score_gap: float


def log_rag_invocation(log: RAGLog) -> None:
    settings = get_settings()
    if settings.rag_log_level.lower() in ("none", "off", "0"):
        return

    path = settings.rag_log_path
    path.parent.mkdir(parents=True, exist_ok=True)

    records: list = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                records = []
            elif isinstance(data, list):
                records = data
            else:
                records = [data]

    records.append(asdict(log))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(records, f, allow_unicode=True, sort_keys=False)
