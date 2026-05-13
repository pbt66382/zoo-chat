"""
Cross-Encoder Reranker（Phase 5）。

Reranker 工作原理
-----------------
向量检索（Bi-Encoder）：
  将查询和文档分别编码为向量，用余弦相似度排序。
  优点：检索速度快（O(1) 向量查找）。
  缺点：查询和文档独立编码，无法捕获精细的词级交互。

Cross-Encoder（精排）：
  将查询和文档拼接后一起输入模型，输出一个相关性得分。
  模型可以看到两段文本的完整 token 级交互（注意力机制），排序精度更高。
  缺点：每对 (query, doc) 都需要一次前向传播，无法预计算，只适合小规模重排。

两阶段策略（推荐）
-----------------
第一阶段：向量检索召回 Top-20（粗排，快）
第二阶段：Cross-Encoder 对 Top-20 精排，取 Top-3 返回（精排，慢但准）

推荐模型
--------
中文：BAAI/bge-reranker-base（BAAI 出品，与 BGE 系列嵌入模型搭配效果好）
      BAAI/bge-reranker-large（更大更准，约 4x 延迟）
多语：cross-encoder/ms-marco-multilingual-MiniLM-L12-v2
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document

from config.settings import get_settings

logger = logging.getLogger("zoo_chat.retrieval.reranker")


class CrossEncoderReranker:
    """使用 sentence-transformers CrossEncoder 对候选文档重新排序。"""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("Reranker 需要 sentence-transformers：pip install sentence-transformers") from e

        logger.info("reranker_loading model=%s", model_name)
        self._model = CrossEncoder(model_name, max_length=512)
        logger.info("reranker_loaded model=%s", model_name)

    def rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: Optional[int] = None,
    ) -> list[Document]:
        """
        对 docs 按与 query 的相关性重排序。

        Args:
            query: 用户查询
            docs:  候选文档列表（来自向量检索或 Hybrid Search）
            top_n: 返回前 N 个（None 表示全部返回）

        Returns:
            重排后的 Document 列表，metadata["rerank_score"] 存新得分。
        """
        if not docs:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        scored_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score in scored_docs[:top_n]:
            doc.metadata["rerank_score"] = round(float(score), 6)
            doc.metadata["score"] = round(float(score), 6)
            result.append(doc)

        logger.debug(
            "rerank_done query_len=%d candidates=%d returned=%d top_score=%.4f",
            len(query), len(docs), len(result),
            result[0].metadata.get("rerank_score", 0.0) if result else 0.0,
        )
        return result


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """获取（或延迟加载）全局 Reranker 单例。"""
    settings = get_settings()
    return CrossEncoderReranker(model_name=settings.rerank_model)
