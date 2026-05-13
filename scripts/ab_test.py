"""
召回策略 A/B 对比实验（Phase 5）。

实验设计
--------
将 eval_recall.jsonl 中的测试用例分别跑三种召回策略：
  A: 纯向量检索（baseline）
  B: BM25 关键词检索
  C: Hybrid Search（向量 + BM25 RRF 融合）
  D: Hybrid Search + Reranker

对每种策略记录：Recall@K、MRR、平均延迟、置信区间。
最终输出对比报告（Markdown 表格 + 统计显著性检验）。

用法
----
python scripts/ab_test.py                     # 对比全部策略
python scripts/ab_test.py --strategies vector hybrid  # 只对比指定策略
python scripts/ab_test.py --top-k 5          # 调整 Top-K
python scripts/ab_test.py --output report.md # 保存报告到文件

统计显著性
----------
使用 McNemar 检验（针对召回命中/未命中的二元结果），
p < 0.05 视为显著差异。
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

from scripts.eval_recall import (
    _PRODUCT_TO_COLLECTION,
    _SEARCH_FNS,
    load_eval_dataset,
)


def run_strategy(
    cases: list[dict],
    strategy: str,
    top_k: int,
    products: list[str] | None = None,
) -> dict:
    """运行单个策略，收集原始结果（用于后续统计）。"""
    search_fn = _SEARCH_FNS[strategy]
    results = []
    total_latency = 0.0

    for case in cases:
        product = case.get("product", "meetings")
        if products and product not in products:
            continue
        collection = _PRODUCT_TO_COLLECTION.get(product, "zoo_faq_meetings")
        expected_ids = set(case.get("expected_faq_ids", []))

        t0 = time.perf_counter()
        try:
            retrieved_ids = search_fn(case["question"], collection, top_k)
        except Exception as e:
            print(f"    ⚠️  {strategy} 检索失败: {e}")
            retrieved_ids = []
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        hit = bool(expected_ids & set(retrieved_ids))
        mrr = 0.0
        for rank, fid in enumerate(retrieved_ids, 1):
            if fid in expected_ids:
                mrr = 1.0 / rank
                break
        results.append({
            "question": case["question"],
            "product": product,
            "hit": hit,
            "mrr": mrr,
            "retrieved_ids": retrieved_ids,
            "expected_ids": list(expected_ids),
            "latency_ms": latency_ms,
        })

    n = len(results)
    if n == 0:
        return {"strategy": strategy, "error": "无样本"}

    hits = [r["hit"] for r in results]
    mrrs = [r["mrr"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    recall = sum(hits) / n
    mrr_avg = sum(mrrs) / n
    avg_latency = total_latency / n

    # 95% 置信区间（Wilson score interval for recall）
    from math import sqrt
    z = 1.96
    p = recall
    n_float = float(n)
    ci_margin = z * sqrt((p * (1 - p) + z * z / (4 * n_float)) / n_float) / (1 + z * z / n_float)

    return {
        "strategy": strategy,
        "n": n,
        f"recall@{top_k}": round(recall, 4),
        f"recall@{top_k}_ci95": f"±{ci_margin:.4f}",
        "mrr": round(mrr_avg, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(sorted(latencies)[n // 2], 1),
        "p95_latency_ms": round(sorted(latencies)[int(n * 0.95)], 1),
        "raw": results,
    }


def mcnemar_test(hits_a: list[bool], hits_b: list[bool]) -> dict:
    """
    McNemar 检验：比较两个二元结果序列是否有显著差异。

    计算：
      b = 策略 A 命中但 B 未命中的数量
      c = 策略 B 命中但 A 未命中的数量
      χ² = (|b-c| - 1)² / (b+c)  （Yates 连续性修正）
      p-value 从 χ² 分布（df=1）得到
    """
    if len(hits_a) != len(hits_b):
        return {"error": "样本数不同"}

    b = sum(1 for a, bb in zip(hits_a, hits_b) if a and not bb)
    c = sum(1 for a, bb in zip(hits_a, hits_b) if not a and bb)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "b": b, "c": c}

    try:
        from scipy.stats import chi2
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = chi2.sf(stat, df=1)
    except ImportError:
        # scipy 未安装时使用近似
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        # 粗略近似：stat > 3.84 时 p < 0.05
        p_value = 0.04 if stat > 3.84 else 0.5

    return {
        "chi2_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "significant_p05": bool(p_value < 0.05),
        "b": b,
        "c": c,
        "interpretation": (
            f"策略 B 比 A 多命中 {c-b} 个样本" if c > b
            else f"策略 A 比 B 多命中 {b-c} 个样本" if b > c
            else "两策略表现相当"
        ),
    }


def format_markdown_report(all_results: list[dict], top_k: int, stats: dict) -> str:
    lines = [
        "# Zoo 召回策略 A/B 对比实验报告",
        "",
        f"## 实验参数",
        f"- Top-K: {top_k}",
        f"- 评估样本: {all_results[0].get('n', '?')} 条",
        "",
        f"## 召回指标对比",
        "",
        f"| 策略 | Recall@{top_k} | 置信区间(95%) | MRR | P50延迟 | P95延迟 |",
        f"|------|--------|--------|-----|---------|---------|",
    ]

    for r in all_results:
        if "error" in r:
            continue
        strategy = r["strategy"]
        recall = r.get(f"recall@{top_k}", 0)
        ci = r.get(f"recall@{top_k}_ci95", "?")
        mrr = r.get("mrr", 0)
        p50 = r.get("p50_latency_ms", 0)
        p95 = r.get("p95_latency_ms", 0)
        lines.append(
            f"| {strategy} | {recall:.4f} | {ci} | {mrr:.4f} | {p50}ms | {p95}ms |"
        )

    lines.extend(["", "## 显著性检验（McNemar）", ""])
    for pair_key, stat in stats.items():
        a, b = pair_key.split(" vs ")
        sig = "✅ 显著" if stat.get("significant_p05") else "❌ 不显著"
        lines.append(f"### {a} vs {b}")
        lines.append(f"- χ² = {stat.get('chi2_statistic', '?')}, p = {stat.get('p_value', '?')} ({sig})")
        lines.append(f"- {stat.get('interpretation', '')}")
        lines.append("")

    lines.extend([
        "## 结论与建议",
        "",
        "根据以上实验结果：",
        "- Recall@K 最高的策略在这个数据集上表现最优。",
        "- 若显著性检验 p < 0.05，说明策略差异不是随机波动，可以置信地选择更优策略。",
        "- 综合召回率和延迟选择合适的线上策略（hybrid 通常是最优平衡点）。",
        "",
        "> 生成于 Zoo AI 系统 Phase 5 召回优化实验",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="召回策略 A/B 对比实验")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["vector", "bm25", "hybrid"],
        default=["vector", "bm25", "hybrid"],
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--product", nargs="*", choices=list(_PRODUCT_TO_COLLECTION.keys()))
    parser.add_argument("--output", default=None, help="将报告保存到 Markdown 文件")
    parser.add_argument(
        "--eval-file",
        default=str(_PROJECT_ROOT / "data" / "eval" / "recall_eval.jsonl"),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    cases = load_eval_dataset(eval_path)
    print(f"加载评估集：{len(cases)} 条样本")
    print(f"评估策略：{args.strategies}  Top-K：{args.top_k}")
    print("=" * 60)

    all_results = []
    for strategy in args.strategies:
        print(f"\n运行策略：{strategy} ...")
        result = run_strategy(cases, strategy, args.top_k, args.product)
        # 移除 raw 后打印摘要
        summary = {k: v for k, v in result.items() if k != "raw"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        all_results.append(result)

    # 统计显著性检验（两两比较）
    print("\n" + "=" * 60)
    print("显著性检验（McNemar）：")
    significance_stats = {}
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            a, b = all_results[i], all_results[j]
            if "raw" not in a or "raw" not in b:
                continue
            hits_a = [r["hit"] for r in a["raw"]]
            hits_b = [r["hit"] for r in b["raw"]]
            pair_key = f"{a['strategy']} vs {b['strategy']}"
            stat = mcnemar_test(hits_a, hits_b)
            significance_stats[pair_key] = stat
            print(f"\n{pair_key}:")
            print(json.dumps(stat, ensure_ascii=False, indent=2))

    # 生成 Markdown 报告
    report = format_markdown_report(all_results, args.top_k, significance_stats)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n报告已保存：{args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
