"""
构建多产品线 Milvus 向量索引脚本（Phase 4 升级版）。

支持的产品线（collection 名 → FAQ 文件路径）：
  zoo_faq_meetings  ← data/faq_meetings.json（150 条，含增强词）
  zoo_faq_phone     ← data/faq_phone.json
  zoo_faq_earbuds   ← data/faq_earbuds.json
  zoo_faq_mouse     ← data/faq_mouse.json
  zoo_faq_screen    ← data/faq_screen.json

用法：
  python scripts/build_milvus_index.py              # 重建全部产品线
  python scripts/build_milvus_index.py --product meetings  # 只重建会议服务
  python scripts/build_milvus_index.py --product phone earbuds  # 重建多个

脚本是幂等的：每次运行都先 drop 再重建对应 collection。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pymilvus import MilvusClient

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.llm.embedding_client import get_embedding_client
from config.settings import get_settings

# ---------------------------------------------------------------------------
# 产品线定义：product_key → (collection_name, faq_file, has_enrichment)
# ---------------------------------------------------------------------------
PRODUCT_CONFIGS = {
    "meetings": {
        "collection": "zoo_faq_meetings",
        "faq_file": "faq_meetings.json",
        "enrichment_file": "faq_enrichment.json",   # 只有 meetings 有增强词
    },
    "phone": {
        "collection": "zoo_faq_phone",
        "faq_file": "faq_phone.json",
        "enrichment_file": None,
    },
    "earbuds": {
        "collection": "zoo_faq_earbuds",
        "faq_file": "faq_earbuds.json",
        "enrichment_file": None,
    },
    "mouse": {
        "collection": "zoo_faq_mouse",
        "faq_file": "faq_mouse.json",
        "enrichment_file": None,
    },
    "screen": {
        "collection": "zoo_faq_screen",
        "faq_file": "faq_screen.json",
        "enrichment_file": None,
    },
}


def _load_enrichment(data_dir: Path, enrichment_file: str | None) -> tuple[dict, dict]:
    if not enrichment_file:
        return {}, {}
    path = data_dir / enrichment_file
    if not path.exists():
        return {}, {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    variations = {int(k): v for k, v in raw.get("variations", {}).items()}
    answer_kws = {int(k): v for k, v in raw.get("answer_keywords", {}).items()}
    return variations, answer_kws


def _build_text(faq: dict, variations: dict, answer_kws: dict) -> str:
    """构建向量化文本：原始 Q/A + 变体问法 + tags + 答案关键词。"""
    fid = faq["id"]
    var_text = " ".join(variations.get(fid, []))
    tags = " ".join(faq.get("tags", []))
    ans_kw = answer_kws.get(fid, "")
    original = f"{faq['question']} {faq['answer']}"
    return " ".join(filter(None, [var_text, tags, ans_kw, original]))


def build_product_index(product_key: str, client: MilvusClient, embeddings, settings) -> int:
    """为单个产品线重建 Milvus collection。返回入库条数。"""
    cfg = PRODUCT_CONFIGS[product_key]
    data_dir = settings.data_dir
    collection_name = cfg["collection"]

    faq_path = data_dir / cfg["faq_file"]
    if not faq_path.exists():
        print(f"  ⚠️  找不到 FAQ 文件 {faq_path}，跳过。")
        return 0

    with faq_path.open(encoding="utf-8") as f:
        faqs = json.load(f)

    variations, answer_kws = _load_enrichment(data_dir, cfg["enrichment_file"])

    # Drop & recreate
    if client.has_collection(collection_name):
        print(f"  删除旧 collection '{collection_name}'...")
        client.drop_collection(collection_name)

    dim = settings.embedding_dimension
    print(f"  创建 collection '{collection_name}' (dim={dim}, faqs={len(faqs)})...")
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        auto_id=True,
        enable_dynamic_field=True,
        vector_field_name="vector",
        index_params=[{
            "field_name": "vector",
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128},
        }],
    )

    texts = [_build_text(faq, variations, answer_kws) for faq in faqs]
    metadatas = [
        {
            "faq_id": faq["id"],
            "tags": ",".join(faq.get("tags", [])),
            "intent": next((t for t in faq.get("tags", []) if "_" in t or t.isascii()), ""),
        }
        for faq in faqs
    ]

    print(f"  计算 {len(texts)} 条 FAQ 的向量嵌入...")
    vectors = embeddings.embed_documents(texts)

    print(f"  写入 Milvus...")
    data = [{"vector": v, "text": texts[i], **metadatas[i]} for i, v in enumerate(vectors)]
    client.insert(collection_name, data)
    client.flush(collection_name)

    count = client.query(collection_name, output_fields=["count(*)"])
    cnt = count[0]["count(*)"]
    print(f"  ✅ {collection_name}: {cnt} 条 FAQ 入库完成。")
    return cnt


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Zoo 多产品线 Milvus 向量索引")
    parser.add_argument(
        "--product",
        nargs="*",
        choices=list(PRODUCT_CONFIGS.keys()),
        default=None,
        help="指定要重建的产品线（不传则重建全部）",
    )
    args = parser.parse_args()

    products_to_build = args.product or list(PRODUCT_CONFIGS.keys())

    settings = get_settings()
    embeddings = get_embedding_client()

    print(f"连接 Milvus ({settings.milvus_host}:{settings.milvus_port})...")
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    total = 0
    for pk in products_to_build:
        print(f"\n[{pk}] 开始构建...")
        total += build_product_index(pk, client, embeddings, settings)

    print(f"\n🎉 全部完成！共入库 {total} 条 FAQ。")


if __name__ == "__main__":
    main()
