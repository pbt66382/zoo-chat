"""
RetrievalStep：用 ctx.question 做 embedding + Milvus 向量检索，写入 ctx.retrieved_docs。

把原 ``faq_chain._retrieve_docs`` 升级为 step：
* 支持 ``top_k`` 注入。
* 把 embedding 与检索两段耗时分别记到 ``ctx.metrics``，方便观测瓶颈。
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


class RetrievalStep:
    name = "retrieval"

    def __init__(self, top_k: int = 3) -> None:
        self._top_k = top_k

    def _retrieve_sync(self, query: str) -> tuple[list[Document], float, float]:
        settings = get_settings()
        embeddings = get_embedding_client()

        t0 = time.perf_counter()
        vector = embeddings.embed_query(query)
        embedding_ms = (time.perf_counter() - t0) * 1000

        client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        t1 = time.perf_counter()
        results = client.search(
            collection_name=settings.milvus_collection,
            data=[vector],
            limit=self._top_k,
            output_fields=["faq_id", "tags", "text"],
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000

        docs: list[Document] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        "faq_id": entity.get("faq_id", "?"),
                        "tags": entity.get("tags", ""),
                        "score": hit.get("distance", 0.0),
                    },
                )
            )
        return docs, embedding_ms, retrieval_ms

    async def run(self, ctx: ChatContext) -> None:
        docs, embed_ms, retr_ms = await asyncio.to_thread(self._retrieve_sync, ctx.question)
        ctx.retrieved_docs = docs
        ctx.metrics["embedding_ms"] = round(embed_ms, 2)
        ctx.metrics["retrieval_only_ms"] = round(retr_ms, 2)
        logger.info(
            "retrieval_done request_id=%s top_k=%d hit_ids=%s top_score=%.4f",
            ctx.request_id,
            len(docs),
            [d.metadata.get("faq_id") for d in docs],
            docs[0].metadata.get("score", 0.0) if docs else 0.0,
        )
