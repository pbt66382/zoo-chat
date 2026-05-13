"""
RetrievalStep（Phase 2/4/5 升级版）。

Phase 2：基础向量检索（固定 collection）
Phase 4：动态 collection（由 ctx.milvus_collection 决定，支持多产品线）
Phase 5：支持三种召回策略（RETRIEVAL_STRATEGY 环境变量控制）
  - vector：纯向量检索（默认）
  - bm25：纯 BM25 关键词检索
  - hybrid：向量 + BM25 加权混合，可选 Reranker 精排
"""
from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.llm.embedding_client import get_embedding_client
from app.pipeline.base import ChatContext
from config.settings import get_settings

logger = logging.getLogger("zoo_chat.pipeline.retrieval")


def _vector_search(query: str, collection: str, top_k: int) -> tuple[list[Document], float, float]:
    """纯向量检索，返回 (docs, embedding_ms, search_ms)。"""
    settings = get_settings()
    embeddings = get_embedding_client()

    t0 = time.perf_counter()
    vector = embeddings.embed_query(query)
    embedding_ms = (time.perf_counter() - t0) * 1000

    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    t1 = time.perf_counter()
    results = client.search(
        collection_name=collection,
        data=[vector],
        limit=top_k,
        output_fields=["faq_id", "tags", "text"],
    )
    search_ms = (time.perf_counter() - t1) * 1000

    docs: list[Document] = []
    for hit in results[0]:
        entity = hit.get("entity", {})
        docs.append(Document(
            page_content=entity.get("text", ""),
            metadata={
                "faq_id": entity.get("faq_id", "?"),
                "tags": entity.get("tags", ""),
                "score": hit.get("distance", 0.0),
                "source": "vector",
            },
        ))
    return docs, embedding_ms, search_ms


def _hybrid_search(query: str, collection: str, top_k: int) -> tuple[list[Document], float, float]:
    """
    Hybrid Search：向量检索 + BM25 关键词检索，RRF 融合排序。

    Reciprocal Rank Fusion (RRF) 公式：
        score(d) = Σ 1 / (k + rank_in_list_i)    k=60 为平滑常数
    相比简单加权，RRF 对单一列表的排名更鲁棒，无需调权重。
    """
    from app.retrieval.bm25_retriever import get_bm25_retriever
    from app.retrieval.reranker import get_reranker

    settings = get_settings()

    # --- 向量检索 ---
    t0 = time.perf_counter()
    vector_docs, embed_ms, vec_ms = _vector_search(query, collection, top_k * 3)
    t1 = time.perf_counter()

    # --- BM25 检索 ---
    bm25 = get_bm25_retriever(collection)
    bm25_docs = bm25.search(query, top_k=top_k * 3)
    bm25_ms = (time.perf_counter() - t1) * 1000

    # --- RRF 融合 ---
    k = 60
    scores: dict[str, float] = {}
    content_map: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        content_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in content_map:
            content_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    fused_docs = []
    for key in sorted_keys:
        doc = content_map[key]
        doc.metadata["score"] = round(scores[key], 6)
        doc.metadata["source"] = "hybrid"
        fused_docs.append(doc)

    # --- 可选 Reranker 精排 ---
    if settings.rerank_enabled and fused_docs:
        reranker = get_reranker()
        fused_docs = reranker.rerank(query, fused_docs, top_n=top_k)

    total_ms = (time.perf_counter() - t0) * 1000
    return fused_docs, embed_ms, total_ms


class RetrievalStep:
    name = "retrieval"

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k

    def _resolve_collection(self, ctx: ChatContext) -> str:
        settings = get_settings()
        return ctx.milvus_collection or settings.default_collection

    def _do_retrieve(self, query: str, collection: str) -> tuple[list[Document], float, float]:
        settings = get_settings()
        top_k = self._top_k or settings.top_k
        strategy = settings.retrieval_strategy

        if strategy == "hybrid":
            return _hybrid_search(query, collection, top_k)
        elif strategy == "bm25":
            from app.retrieval.bm25_retriever import get_bm25_retriever
            t0 = time.perf_counter()
            docs = get_bm25_retriever(collection).search(query, top_k=top_k)
            ms = (time.perf_counter() - t0) * 1000
            return docs, 0.0, ms
        else:
            return _vector_search(query, collection, top_k)

    async def run(self, ctx: ChatContext) -> None:
        collection = self._resolve_collection(ctx)
        docs, embed_ms, retr_ms = await asyncio.to_thread(
            self._do_retrieve, ctx.question, collection
        )
        ctx.retrieved_docs = docs
        ctx.metrics["embedding_ms"] = round(embed_ms, 2)
        ctx.metrics["retrieval_only_ms"] = round(retr_ms, 2)
        logger.info(
            "retrieval_done request_id=%s collection=%s strategy=%s top_k=%d hit_ids=%s top_score=%.4f",
            ctx.request_id,
            collection,
            get_settings().retrieval_strategy,
            len(docs),
            [d.metadata.get("faq_id") for d in docs],
            docs[0].metadata.get("score", 0.0) if docs else 0.0,
        )
