"""
端到端评估脚本：跑一批问题穿过完整 pipeline，评估意图准确率 + Recall@K + 关键词命中率。

使用方式：

    python -m scripts.eval_e2e [--dataset path] [--top-k 3] [--no-llm]

* ``--no-llm``：跳过 generation，只跑到 retrieval。Recall@K 仍然有效，节省 LLM token。
* 报告输出到 ``wiki/PHASE3_EVAL_REPORT.md``（追加到意图评估报告之后）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from app.intent.classifier import IntentClassifier
from app.pipeline.base import ChatContext, ChatPipeline
from app.pipeline.generation_step import GenerationStep
from app.pipeline.intent_step import IntentStep
from app.pipeline.retrieval_step import RetrievalStep
from app.pipeline.router_step import RouterStep
from app.pipeline.slot_filling_step import SlotFillingStep

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "eval" / "e2e_eval.jsonl"
_DEFAULT_REPORT = _PROJECT_ROOT / "wiki" / "PHASE3_EVAL_REPORT.md"


def _load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _build_pipeline(top_k: int, with_generation: bool, with_slot_filling: bool) -> ChatPipeline:
    """评估默认不挂 SlotFillingStep——目的是衡量"召回 + 生成"质量，
    多轮槽位流程已由单元测试覆盖。如需评估完整体验，传 ``--with-slots``。"""
    steps = [
        IntentStep(classifier=IntentClassifier()),
        RouterStep(),
    ]
    if with_slot_filling:
        steps.append(SlotFillingStep())
    steps.append(RetrievalStep(top_k=top_k))
    if with_generation:
        steps.append(GenerationStep())
    return ChatPipeline(steps=steps)


async def _run_one(pipeline: ChatPipeline, question: str) -> ChatContext:
    ctx = ChatContext(
        request_id="eval",
        session_id=f"eval_{int(time.time() * 1000)}",
        question=question,
    )
    await pipeline.run(ctx)
    return ctx


async def _evaluate(items: list[dict], top_k: int, with_generation: bool, with_slot_filling: bool, verbose: bool) -> dict:
    pipeline = _build_pipeline(top_k, with_generation, with_slot_filling)
    rows: list[dict] = []
    intent_correct = 0
    recall_hits = 0
    keyword_hits = 0
    keyword_total = 0

    for i, item in enumerate(items, start=1):
        question = item["question"]
        expected_intent = item.get("expected_intent")
        expected_faq = item.get("expected_faq_id")
        expected_kws = [str(k).lower() for k in item.get("expected_keywords", [])]

        ctx = await _run_one(pipeline, question)
        recalled_ids = [int(d.metadata.get("faq_id", -1)) for d in ctx.retrieved_docs]
        intent_match = ctx.intent_id == expected_intent
        recall_match = expected_faq in recalled_ids if expected_faq is not None else False
        if intent_match:
            intent_correct += 1
        if recall_match:
            recall_hits += 1

        kw_hit = False
        if with_generation and expected_kws:
            answer_lower = (ctx.answer or "").lower()
            kw_hit = any(kw in answer_lower for kw in expected_kws)
            keyword_total += 1
            if kw_hit:
                keyword_hits += 1

        rows.append({
            "question": question,
            "expected_intent": expected_intent,
            "got_intent": ctx.intent_id,
            "expected_faq": expected_faq,
            "recalled": recalled_ids,
            "recall_hit": recall_match,
            "intent_hit": intent_match,
            "keyword_hit": kw_hit,
            "answer_preview": (ctx.answer or "")[:120],
        })

        if verbose:
            mark_i = "i+" if intent_match else "i-"
            mark_r = "r+" if recall_match else "r-"
            print(f"[{i:>3}/{len(items)}] {mark_i} {mark_r} intent={ctx.intent_id} recalled={recalled_ids} q={question}")

    n = len(items)
    return {
        "n": n,
        "intent_accuracy": intent_correct / n if n else 0.0,
        "recall_at_k": recall_hits / n if n else 0.0,
        "keyword_hit_rate": (keyword_hits / keyword_total) if keyword_total else None,
        "rows": rows,
        "top_k": top_k,
        "with_generation": with_generation,
    }


def _format_report(result: dict) -> str:
    lines = [
        "## 端到端评估",
        "",
        f"- 数据集大小：**{result['n']}** 条",
        f"- Top-K：**{result['top_k']}**",
        f"- 是否走 generation：**{result['with_generation']}**",
        "",
        f"- 意图准确率：**{result['intent_accuracy']:.2%}**",
        f"- Recall@{result['top_k']}（召回包含期望 FAQ）：**{result['recall_at_k']:.2%}**",
    ]
    if result["keyword_hit_rate"] is not None:
        lines.append(f"- 关键词命中率：**{result['keyword_hit_rate']:.2%}**")
    lines.extend(["", "### 明细", ""])
    headers = ["question", "expected_intent", "got_intent", "expected_faq", "recalled", "intent_hit", "recall_hit", "keyword_hit"]
    rows = []
    for r in result["rows"]:
        rows.append([
            r["question"],
            str(r["expected_intent"]),
            str(r["got_intent"]),
            str(r["expected_faq"]),
            ",".join(str(x) for x in r["recalled"]),
            "OK" if r["intent_hit"] else "X",
            "OK" if r["recall_hit"] else "X",
            "OK" if r["keyword_hit"] else "-",
        ])
    lines.append(_format_table(headers, rows))
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-llm", action="store_true", help="Skip generation step")
    parser.add_argument("--with-slots", action="store_true", help="Include SlotFillingStep (will short-circuit on troubleshoot_* intents)")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    items = _load_dataset(args.dataset)
    print(f"Loaded {len(items)} samples from {args.dataset}; with_generation={not args.no_llm}, with_slots={args.with_slots}")

    t0 = time.perf_counter()
    result = asyncio.run(_evaluate(
        items,
        args.top_k,
        with_generation=not args.no_llm,
        with_slot_filling=args.with_slots,
        verbose=args.verbose,
    ))
    elapsed = time.perf_counter() - t0
    print(
        f"Eval done in {elapsed:.1f}s | intent_acc={result['intent_accuracy']:.2%} | "
        f"recall@{result['top_k']}={result['recall_at_k']:.2%}"
    )

    report = _format_report(result)
    print()
    print(report)

    if not args.no_write:
        # 追加到现有报告（如果存在）；否则新建
        args.report.parent.mkdir(parents=True, exist_ok=True)
        prev = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
        # 移除已有的端到端段落，避免重复
        marker = "## 端到端评估"
        if marker in prev:
            prev = prev.split(marker)[0].rstrip() + "\n\n"
        args.report.write_text(prev + report, encoding="utf-8")
        print(f"Report appended to {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
