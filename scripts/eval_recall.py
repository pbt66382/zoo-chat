"""
召回质量评估脚本（Phase 5）。

评估指标
--------
Recall@K   在 Top-K 个召回结果中，有多少包含了期望的 FAQ ID。
           Recall@K = 1 if any(expected_id in top_k_ids) else 0
           最终取全部测试样本的平均值。

MRR        Mean Reciprocal Rank，衡量期望 FAQ 在结果中的平均排名。
           MRR = avg(1 / rank_of_first_relevant_doc)

NDCG@K     Normalized Discounted Cumulative Gain，综合排名和相关性的指标。

用法
----
# 使用默认向量检索策略评估
python scripts/eval_recall.py

# 指定策略和 Top-K
python scripts/eval_recall.py --strategy hybrid --top-k 5

# 只评估特定产品线
python scripts/eval_recall.py --product meetings phone

# 输出详细失败案例
python scripts/eval_recall.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def load_eval_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def vector_search(query: str, collection: str, top_k: int) -> list[int]:
    """向量检索，返回 faq_id 列表（按相关性降序）。"""
    from app.llm.embedding_client import get_embedding_client
    from config.settings import get_settings
    from pymilvus import MilvusClient

    settings = get_settings()
    embeddings = get_embedding_client()
    vector = embeddings.embed_query(query)

    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    results = client.search(
        collection_name=collection,
        data=[vector],
        limit=top_k,
        output_fields=["faq_id"],
    )
    return [hit["entity"]["faq_id"] for hit in results[0]] if results else []


def bm25_search(query: str, collection: str, top_k: int) -> list[int]:
    """BM25 检索，返回 faq_id 列表。"""
    from app.retrieval.bm25_retriever import get_bm25_retriever
    docs = get_bm25_retriever(collection).search(query, top_k=top_k)
    return [d.metadata.get("faq_id") for d in docs]


def hybrid_search(query: str, collection: str, top_k: int) -> list[int]:
    """Hybrid Search，返回 faq_id 列表。"""
    from app.retrieval.bm25_retriever import get_bm25_retriever
    from app.llm.embedding_client import get_embedding_client
    from config.settings import get_settings
    from pymilvus import MilvusClient
    from langchain_core.documents import Document

    settings = get_settings()

    # 向量检索
    embeddings = get_embedding_client()
    vector = embeddings.embed_query(query)
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    vec_results = client.search(
        collection_name=collection,
        data=[vector],
        limit=top_k * 3,
        output_fields=["faq_id", "tags", "text"],
    )
    vec_docs = [
        Document(
            page_content=hit["entity"].get("text", ""),
            metadata={"faq_id": hit["entity"].get("faq_id"), "score": hit.get("distance", 0)},
        )
        for hit in (vec_results[0] if vec_results else [])
    ]

    # BM25 检索
    bm25_docs = get_bm25_retriever(collection).search(query, top_k=top_k * 3)

    # RRF 融合
    k = 60
    scores: dict[int, float] = {}
    faq_id_map: dict[int, int] = {}
    for rank, doc in enumerate(vec_docs):
        fid = doc.metadata.get("faq_id")
        if fid:
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(bm25_docs):
        fid = doc.metadata.get("faq_id")
        if fid:
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return sorted_ids


# collection 名映射
_PRODUCT_TO_COLLECTION = {
    "meetings": "zoo_faq_meetings",
    "phone":    "zoo_faq_phone",
    "earbuds":  "zoo_faq_earbuds",
    "mouse":    "zoo_faq_mouse",
    "screen":   "zoo_faq_screen",
}

_SEARCH_FNS = {
    "vector":  vector_search,
    "bm25":    bm25_search,
    "hybrid":  hybrid_search,
}


def compute_metrics(
    cases: list[dict],
    strategy: str,
    top_k: int,
    products: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """计算 Recall@K 和 MRR。"""
    search_fn = _SEARCH_FNS[strategy]

    recall_hits = []
    mrr_scores = []
    per_product: dict[str, list] = defaultdict(list)
    failures = []

    for case in cases:
        product = case.get("product", "meetings")
        if products and product not in products:
            continue

        collection = _PRODUCT_TO_COLLECTION.get(product, "zoo_faq_meetings")
        expected_ids = set(case.get("expected_faq_ids", []))
        question = case["question"]

        t0 = time.perf_counter()
        try:
            retrieved_ids = search_fn(question, collection, top_k)
        except Exception as e:
            print(f"  ⚠️  检索失败 [{product}] '{question[:30]}': {e}")
            recall_hits.append(0)
            mrr_scores.append(0.0)
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        # Recall@K：任意一个 expected_id 出现在 top-k 结果中
        hit = int(bool(expected_ids & set(retrieved_ids)))
        recall_hits.append(hit)
        per_product[product].append(hit)

        # MRR：第一个 expected_id 出现的排名的倒数
        mrr = 0.0
        for rank, fid in enumerate(retrieved_ids, 1):
            if fid in expected_ids:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        if not hit and verbose:
            failures.append({
                "question": question,
                "product": product,
                "expected": list(expected_ids),
                "retrieved": retrieved_ids[:5],
                "latency_ms": round(latency_ms, 1),
            })

    total = len(recall_hits)
    if total == 0:
        return {"error": "没有可评估的样本"}

    result = {
        "strategy": strategy,
        "top_k": top_k,
        "total_cases": total,
        f"recall@{top_k}": round(sum(recall_hits) / total, 4),
        "mrr": round(sum(mrr_scores) / total, 4),
        "per_product": {
            p: round(sum(hits) / len(hits), 4)
            for p, hits in per_product.items()
        },
    }

    if verbose and failures:
        result["failures"] = failures

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Zoo 召回质量评估")
    parser.add_argument("--strategy", choices=["vector", "bm25", "hybrid"], default="vector")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--product", nargs="*", choices=list(_PRODUCT_TO_COLLECTION.keys()))
    parser.add_argument("--verbose", action="store_true", help="显示失败案例详情")
    parser.add_argument(
        "--eval-file",
        default=str(_PROJECT_ROOT / "data" / "eval" / "recall_eval.jsonl"),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"找不到评估文件：{eval_path}")
        sys.exit(1)

    cases = load_eval_dataset(eval_path)
    print(f"加载评估集：{len(cases)} 条样本")
    print(f"策略：{args.strategy}  Top-K：{args.top_k}")
    print("-" * 50)

    t0 = time.perf_counter()
    metrics = compute_metrics(
        cases,
        strategy=args.strategy,
        top_k=args.top_k,
        products=args.product,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n评估耗时：{elapsed:.1f}s")

    if "failures" in metrics:
        print(f"\n=== 召回失败案例（共 {len(metrics['failures'])} 条）===")
        for f in metrics["failures"]:
            print(f"  [{f['product']}] Q: {f['question']}")
            print(f"    期望 FAQ IDs: {f['expected']}")
            print(f"    实际召回 IDs: {f['retrieved']}")
            print(f"    延迟: {f['latency_ms']}ms")


if __name__ == "__main__":
    main()
