"""从逐题报告汇总一个分集的评测结果并生成本地归因图。"""

import json
from html import escape
from math import ceil
from pathlib import Path
from statistics import mean, median

_ATTRIBUTIONS = {
    "missing_material": "资料缺失",
    "retrieval_miss": "召回遗漏",
    "answer_strategy_error": "应答策略错误",
    "citation_error": "引用错误",
    "missing_fact": "事实遗漏",
    "service_error": "服务异常",
}
_SPLITS = {"debug": "工程调试集", "holdout": "正式留出集"}


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize_report(report_path: Path, config: dict, questions: list[dict]) -> dict:
    """读取已完成的逐题 JSONL，按冻结标注确定各指标的适用分母。"""
    with report_path.open(encoding="utf-8") as report:
        rows = [json.loads(line) for line in report]
    if [row["question_id"] for row in rows] != [question["id"] for question in questions] or not rows:
        raise ValueError("逐题报告与本次选定题目不匹配")

    group_count = sum(len(question["evidence_groups"]) for question in questions)
    groups = [group for row in rows for group in row["score"]["groups"]]
    answer_rows = [row for question, row in zip(questions, rows, strict=True)
                   if question["expected_error_code"] is None]
    attributions = dict.fromkeys(_ATTRIBUTIONS, 0)
    for row in rows:
        for attribution in set(row["score"]["attributions"]):
            attributions[attribution] += 1
    timings = sorted(row["timings_s"]["total"] for row in rows)
    count = len(rows)
    return {
        "run_id": config["run_id"],
        "dataset_version": config["dataset_version"],
        "split": config["split"],
        "question_id": config["question_id"],
        "question_count": count,
        "passed": sum(row["score"]["passed"] for row in rows),
        "evidence_coverage": _ratio(sum(group["in_context"] is True for group in groups), group_count),
        "citations_pass_rate": _ratio(
            sum(row["score"]["behavior_ok"] and row["score"]["citations_ok"] is True for row in answer_rows),
            len(answer_rows),
        ),
        "required_fact_pass_rate": _ratio(sum(group["fact_present"] is True for group in groups), group_count),
        "behavior_accuracy": _ratio(sum(row["score"]["behavior_ok"] for row in rows), count),
        "latency_s": {
            "total": sum(timings), "mean": mean(timings), "p50": median(timings),
            "p95": timings[ceil(count * 0.95) - 1],
        },
        "attributions": attributions,
    }


def render_attribution_chart(summary: dict) -> str:
    """生成独立 HTML 条形图，只显示分集、范围和归因计数。"""
    count = summary["question_count"]
    split = _SPLITS[summary["split"]]
    question_id = summary["question_id"]
    scope = f"指定题目 {escape(question_id)}" if question_id else "完整分集"
    bars = "\n".join(
        f'<li><span id="{key}">{_ATTRIBUTIONS[key]}</span>'
        f'<meter aria-labelledby="{key}" min="0" max="{count}" value="{value}"></meter>'
        f'<strong>{value}</strong></li>'
        for key, value in summary["attributions"].items()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>失败归因 · {split}</title>
<style>
body {{ color: #172a36; background: #f4f7f8; font: 16px/1.55 system-ui, sans-serif; margin: 0; padding: 24px; }}
main {{ max-width: 640px; margin: 24px auto; padding: clamp(16px, 5vw, 36px); background: white;
        border: 1px solid #d7e2e8; border-radius: 12px; }}
h1 {{ margin: 0; font-size: 1.75rem; }}
ul {{ list-style: none; padding: 0; }}
li {{ display: grid; grid-template-columns: 7em minmax(0, 1fr) 2em; gap: 8px;
      align-items: center; margin: 10px 0; }}
meter {{ width: 100%; height: 20px; accent-color: #126b70; }}
strong {{ text-align: right; }}
figcaption {{ color: #405968; }}
figure {{ margin: 24px 0 0; }}
</style>
<main>
<h1>失败归因</h1>
<p>{split} · {scope} · 样本量 {count} 题</p>
<figure>
<ul>{bars}</ul>
<figcaption>条形以本次样本量为满刻度。每题在同一类别最多计一次，可计入多个类别。</figcaption>
</figure>
</main>
</html>
"""
