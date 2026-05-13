"""
BM25 关键词检索器（Phase 5）。

BM25（Best Match 25）原理
-------------------------
传统 TF-IDF 的升级版，公式：
  score(q, d) = Σ IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))

其中：
  tf    = 词 t 在文档 d 中的出现频率
  IDF   = log((N - df + 0.5) / (df + 0.5))  （逆文档频率）
  k1    = 1.5（词频饱和参数，越大越重视高频词）
  b     = 0.75（文档长度归一化参数）
  |d|   = 文档长度，avgdl = 平均文档长度

中文处理：用 jieba 分词，将中文句子切分为词序列后喂给 BM25。

适用场景
--------
- 用户问题包含精确的型号、错误码（如 "E-1204"、"H.265"）
- 问题中有独特的关键词在 embedding 空间可能分散（如品牌名、专有名词）
- 作为 Hybrid Search 中的补充信号，弥补纯向量检索的盲区

缓存策略
--------
每个 collection 的 BM25 索引按需构建后缓存在内存中。
数据更新需重启服务（或调用 BM25Retriever.clear_cache()）。
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger("zoo_chat.retrieval.bm25")

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CACHE: dict[str, "BM25Retriever"] = {}

# collection 名 → FAQ 文件路径（与 build_milvus_index.py 保持一致）
_COLLECTION_TO_FILE: dict[str, str] = {
    "zoo_faq_meetings": "faq_meetings.json",
    "zoo_faq_phone":    "faq_phone.json",
    "zoo_faq_earbuds":  "faq_earbuds.json",
    "zoo_faq_mouse":    "faq_mouse.json",
    "zoo_faq_screen":   "faq_screen.json",
}


def _tokenize(text: str) -> list[str]:
    """中文分词 + 英文 token 化。优先用 jieba，未安装时按字符切分。"""
    try:
        import jieba
        return list(jieba.cut(text.lower()))
    except ImportError:
        # 回退：按空格 + 每个汉字单独成 token
        tokens = []
        for part in text.lower().split():
            if any('一' <= c <= '鿿' for c in part):
                tokens.extend(list(part))
            else:
                tokens.append(part)
        return tokens


class BM25Retriever:
    """基于 rank_bm25 库的中文 BM25 检索器。"""

    def __init__(self, docs: list[dict]) -> None:
        """
        docs: [{"id": int, "text": str, "faq_id": int, "tags": str}, ...]
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("BM25 需要安装 rank_bm25：pip install rank-bm25") from e

        self._docs = docs
        tokenized = [_tokenize(d["text"]) for d in docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("bm25_index_built docs=%d", len(docs))

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """返回 BM25 Top-K 结果（Document 列表，分数归一化到 0~1）。"""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        if len(scores) == 0:
            return []

        # 归一化分数
        max_score = max(scores) or 1.0
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, raw_score in indexed:
            if raw_score <= 0:
                continue
            doc_meta = self._docs[idx]
            results.append(Document(
                page_content=doc_meta["text"],
                metadata={
                    "faq_id": doc_meta.get("faq_id", "?"),
                    "tags": doc_meta.get("tags", ""),
                    "score": round(raw_score / max_score, 6),
                    "source": "bm25",
                },
            ))
        return results

    @classmethod
    def clear_cache(cls) -> None:
        _CACHE.clear()
        logger.info("bm25_cache_cleared")


def _load_faq_docs(collection: str) -> list[dict]:
    """从 JSON 文件加载 FAQ 数据，构建 BM25 文档列表。"""
    filename = _COLLECTION_TO_FILE.get(collection)
    if not filename:
        logger.warning("bm25_no_faq_file_for collection=%s", collection)
        return []

    path = _PROJECT_ROOT / "data" / filename
    if not path.exists():
        logger.warning("bm25_faq_file_not_found path=%s", path)
        return []

    with path.open(encoding="utf-8") as f:
        faqs = json.load(f)

    docs = []
    for faq in faqs:
        # 拼接 Q + A 作为检索文本（与 build_milvus_index.py 的 _build_text 一致）
        text = f"{faq['question']} {faq['answer']}"
        docs.append({
            "text": text,
            "faq_id": faq["id"],
            "tags": ",".join(faq.get("tags", [])),
        })
    return docs


def get_bm25_retriever(collection: str) -> BM25Retriever:
    """按 collection 名获取（或延迟构建）BM25 检索器（内存缓存）。"""
    if collection not in _CACHE:
        docs = _load_faq_docs(collection)
        _CACHE[collection] = BM25Retriever(docs)
    return _CACHE[collection]
