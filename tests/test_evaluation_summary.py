"""验证报告汇总的统计口径和归因图。"""

import json

from manual_rag.evaluation_summary import render_attribution_chart, summarize_report


def test_summary_uses_annotated_denominators_and_counts_attributions_per_question(tmp_path):
    """缺失评分仍计入标注分母，重复归因只计一次，错答不会通过引用检查。"""
    config = {"run_id": "run", "dataset_version": "fixture", "split": "debug", "question_id": None}
    questions = [
        {"id": key, "evidence_groups": [{}] * count, "expected_error_code": code}
        for key, count, code in (("a", 2, None), ("b", 1, None), ("c", 1, None), ("d", 0, "clarify_fault"))
    ]
    scores = [
        (True, True, True, [
            {"in_context": True, "fact_present": True},
            {"in_context": False, "fact_present": None},
        ], ["retrieval_miss", "retrieval_miss", "service_error"]),
        (False, False, None, [], ["service_error"]),
        (False, False, True, [{"in_context": True, "fact_present": False}], ["answer_strategy_error"]),
        (True, True, True, [], []),
    ]
    report = tmp_path / "report.jsonl"
    report.write_text("".join(
        json.dumps({
            "question_id": question["id"], "timings_s": {"total": elapsed},
            "score": {"passed": passed, "behavior_ok": behavior, "citations_ok": citations,
                      "groups": groups, "attributions": attributions},
        }) + "\n"
        for question, elapsed, (passed, behavior, citations, groups, attributions) in
        zip(questions, [4, 1, 2, 3], scores, strict=True)
    ), encoding="utf-8")
    summary = summarize_report(report, config, questions)
    assert summary["question_count"] == 4
    assert summary["evidence_coverage"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert summary["citations_pass_rate"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert summary["required_fact_pass_rate"] == {"numerator": 1, "denominator": 4, "rate": 0.25}
    assert summary["behavior_accuracy"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert summary["latency_s"] == {"total": 10, "mean": 2.5, "p50": 2.5, "p95": 4}
    assert summary["attributions"] == {
        "missing_material": 0, "retrieval_miss": 1, "answer_strategy_error": 1,
        "citation_error": 0, "missing_fact": 0, "service_error": 2,
    }
    chart = render_attribution_chart(summary)
    assert "工程调试集 · 完整分集 · 样本量 4 题" in chart
    assert 'aria-labelledby="service_error" min="0" max="4" value="2"' in chart
    assert "可计入多个类别" in chart
