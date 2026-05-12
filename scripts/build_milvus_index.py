"""
构建 Milvus 向量索引脚本（一次性运行）。

把 data/faq_meetings.json 里 150 条 FAQ + data/faq_enrichment.json 里的检索增强词
一起向量化后写入 Milvus collection。脚本是幂等的：每次运行都会先 drop 再重建。
"""
from __future__ import annotations

import json
from pathlib import Path

from pymilvus import MilvusClient

from app.llm.embedding_client import get_embedding_client
from config.settings import get_settings
from data import FAQ_MEETINGS

_PROJECT_ROOT = Path(__file__).parent.parent
_ENRICHMENT_PATH = _PROJECT_ROOT / "data" / "faq_enrichment.json"


def _load_enrichment() -> tuple[dict[int, list[str]], dict[int, str]]:
    with _ENRICHMENT_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    variations = {int(k): v for k, v in raw.get("variations", {}).items()}
    answer_kws = {int(k): v for k, v in raw.get("answer_keywords", {}).items()}
    return variations, answer_kws


def _build_enriched_text(faq: dict, variations: dict[int, list[str]], answer_kws: dict[int, str]) -> str:
    """把 FAQ 的问答拼上变体问法 + tags + 答案关键词，提升中文召回命中率。"""
    faq_id = faq["id"]
    var_text = " ".join(variations.get(faq_id, []))
    tags = " ".join(faq.get("tags", []))
    ans_kw = answer_kws.get(faq_id, "")
    original = f"{faq['question']} {faq['answer']}"
    return " ".join(filter(None, [var_text, tags, ans_kw, original]))


def build_milvus_index() -> None:
    settings = get_settings()
    embeddings = get_embedding_client()
    variations, answer_kws = _load_enrichment()

    print(f"正在连接 Milvus ({settings.milvus_host}:{settings.milvus_port})...")
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    if client.has_collection(settings.milvus_collection):
        print(f"删除旧 collection '{settings.milvus_collection}'...")
        client.drop_collection(settings.milvus_collection)

    dim = settings.embedding_dimension
    print(f"创建 collection '{settings.milvus_collection}' (dim={dim})...")
    client.create_collection(
        collection_name=settings.milvus_collection,
        dimension=dim,
        auto_id=True,
        enable_dynamic_field=True,
        vector_field_name="vector",
        index_params=[
            {
                "field_name": "vector",
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128},
            }
        ],
    )

    texts = [_build_enriched_text(faq, variations, answer_kws) for faq in FAQ_MEETINGS]
    metadatas = [
        {
            "faq_id": faq["id"],
            "tags": ",".join(faq.get("tags", [])),
            # 主要 intent 取 tags 里第一个 snake_case 形式的标签（约定）
            "intent": next((t for t in faq.get("tags", []) if "_" in t or t.isascii()), ""),
        }
        for faq in FAQ_MEETINGS
    ]

    print(f"正在计算 {len(texts)} 条 FAQ 的向量嵌入...")
    vectors = embeddings.embed_documents(texts)

    print(f"正在写入 {len(texts)} 条 FAQ 到 Milvus...")
    data = [
        {"vector": vector, "text": texts[i], **metadatas[i]}
        for i, vector in enumerate(vectors)
    ]
    client.insert(settings.milvus_collection, data)

    client.flush(settings.milvus_collection)
    count = client.query(settings.milvus_collection, output_fields=["count(*)"])
    print(f"Milvus 索引构建完成！共 {count[0]['count(*)']} 条 FAQ 已入库。")


if __name__ == "__main__":
    build_milvus_index()
