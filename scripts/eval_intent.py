"""
意图分类评估脚本。

读取 ``data/eval/intent_eval.jsonl``，每条 ``{"question", "expected_intent"}``，
跑 ``IntentClassifier`` 拿预测结果，输出：
* overall accuracy
* per-intent precision / recall / F1
* confusion matrix (markdown 表格)
* 每条样本预测明细（可选）

输出会同时打到 stdout 与 ``wiki/PHASE3_EVAL_REPORT.md``（追加模式可控）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from app.intent.classifier import IntentClassifier
from app.intent.intents import INTENT_OUT_OF_SCOPE, list_intents

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "eval" / "intent_eval.jsonl"
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


def _evaluate(items: list[dict], classifier: IntentClassifier, verbose: bool = False) -> dict:
    intent_ids = [it.id for it in list_intents()]
    predictions: list[tuple[str, str, float, str]] = []

    for i, item in enumerate(items, start=1):
        question = item["question"]
        expected = item["expected_intent"]
        result = classifier.classify(question)
        predictions.append((expected, result.intent_id, result.confidence, question))
        if verbose:
            mark = "OK" if result.intent_id == expected else "FAIL"
            print(f"[{i:>3}/{len(items)}] {mark} expected={expected:<22} got={result.intent_id:<22} conf={result.confidence:.2f} q={question}")

    correct = sum(1 for exp, pred, *_ in predictions if exp == pred)
    accuracy = correct / len(items) if items else 0.0

    confusion: dict[tuple[str, str], int] = Counter()
    per_intent_total: dict[str, int] = Counter()
    per_intent_correct: dict[str, int] = Counter()
    per_intent_predicted: dict[str, int] = Counter()
    for exp, pred, *_ in predictions:
        confusion[(exp, pred)] += 1
        per_intent_total[exp] += 1
        per_intent_predicted[pred] += 1
        if exp == pred:
            per_intent_correct[exp] += 1

    metrics = []
    for iid in intent_ids:
        total = per_intent_total.get(iid, 0)
        predicted = per_intent_predicted.get(iid, 0)
        tp = per_intent_correct.get(iid, 0)
        precision = tp / predicted if predicted else 0.0
        recall = tp / total if total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics.append({
            "intent": iid,
            "total": total,
            "predicted": predicted,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return {
        "n": len(items),
        "accuracy": accuracy,
        "metrics": metrics,
        "confusion": confusion,
        "predictions": predictions,
    }


def _format_report(result: dict) -> str:
    intent_ids = [it.id for it in list_intents()]

    lines = [
        "# Phase 3 评估报告 - 意图分类",
        "",
        f"- 数据集大小：**{result['n']}** 条",
        f"- 整体准确率：**{result['accuracy']:.2%}**",
        "",
        "## 各意图指标",
        "",
        _format_table(
            ["intent", "total", "tp", "precision", "recall", "f1"],
            [
                [
                    m["intent"],
                    str(m["total"]),
                    str(m["tp"]),
                    f"{m['precision']:.2%}",
                    f"{m['recall']:.2%}",
                    f"{m['f1']:.2%}",
                ]
                for m in result["metrics"]
            ],
        ),
        "",
        "## 混淆矩阵",
        "",
        "行=实际意图，列=预测意图。",
        "",
    ]
    headers = ["actual \\ pred"] + intent_ids
    rows = []
    for actual in intent_ids:
        row = [actual]
        for pred in intent_ids:
            row.append(str(result["confusion"].get((actual, pred), 0)))
        rows.append(row)
    lines.append(_format_table(headers, rows))

    # 失败明细
    failures = [(exp, pred, conf, q) for exp, pred, conf, q in result["predictions"] if exp != pred]
    if failures:
        lines.extend(["", "## 错分明细", ""])
        lines.append(_format_table(
            ["expected", "got", "confidence", "question"],
            [[exp, pred, f"{conf:.2f}", q] for exp, pred, conf, q in failures],
        ))

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Intent classifier evaluation")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--no-write", action="store_true", help="Do not write the markdown report file")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample prediction")
    args = parser.parse_args(argv)

    items = _load_dataset(args.dataset)
    print(f"Loaded {len(items)} samples from {args.dataset}")

    classifier = IntentClassifier()
    t0 = time.perf_counter()
    result = _evaluate(items, classifier, verbose=args.verbose)
    elapsed = time.perf_counter() - t0
    print(f"Eval finished in {elapsed:.1f}s, accuracy = {result['accuracy']:.2%}")

    report = _format_report(result)
    print()
    print(report)

    if not args.no_write:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
