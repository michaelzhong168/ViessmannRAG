import hashlib
import json

import pymupdf
import pytest

from manual_rag.question_set import load_question_set, main


@pytest.fixture
def sample(tmp_path):
    pdf_path = tmp_path / "synthetic.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "Status A")
        page.insert_text((50, 90), "Repeat")
        page.insert_text((50, 130), "Repeat")
        pdf.new_page().insert_text((50, 50), "Status A")
        pdf.save(pdf_path)
        bbox = list(pdf[0].search_for("Repeat")[0] + (-1, -1, 1, 1))
    scopes = tmp_path / "scopes.json"
    scopes.write_text(json.dumps({"doc_id": "synthetic", "scopes": [
        {"first_page": 1, "last_page": 1, "control_type": "按键式"}
    ]}), encoding="utf-8")
    anchor = {
        "doc_id": "synthetic", "pdf_version": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "page_no": 1, "quote": "Status A",
    }
    common = {"dataset_version": "synthetic-v1", "intent_id": "status", "split": "debug"}
    answer = dict(common, id="q-01", question="怎么看状态？", control_type="按键式",
                  expected_error_code=None, evidence_groups=[
                      {"required_fact": "状态为 A", "anchors": [anchor, dict(anchor, quote="Status A", bbox=[40, 30, 110, 55])]},
                      {"required_fact": "复核标志", "anchors": [dict(anchor, quote="Repeat", bbox=bbox)]},
                  ])
    clarify = dict(common, id="q-02", question="状态在哪看？", control_type=None,
                   expected_error_code="missing_control_type", evidence_groups=[])
    refusal = dict(common, id="q-03", intent_id="unknown", split="holdout",
                   question="能查到耗电量吗？", control_type="按键式",
                   expected_error_code="insufficient_evidence", evidence_groups=[])
    return tmp_path / "questions.jsonl", pdf_path, scopes, [answer, clarify, refusal]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_reads_reordered_rows_and_checks_original_pages(sample, capsys):
    path, pdf, scopes, rows = sample
    rows.reverse()
    write_rows(path, rows)
    assert load_question_set(path) == rows
    assert main([str(path), "--pdf", str(pdf), "--scopes", str(scopes)]) == 0
    assert "调试集 2 题，留出集 1 题" in capsys.readouterr().out


@pytest.mark.parametrize(("index", "updates", "reason"), [
    (1, {"id": "q-01"}, "ID 重复"),
    (1, {"dataset_version": "synthetic-v2"}, "版本不一致"),
    (1, {"split": "holdout"}, "同一意图跨越"),
    (1, {"control_type": "未知"}, "control_type"),
    (1, {"question": " "}, "question"),
    (1, {"expected_error_code": "model_timeout"}, "expected_error_code"),
    (0, {"evidence_groups": []}, "应答题必须有证据组"),
    (0, {"evidence_groups": [{"required_fact": "状态为 A", "anchors": []}]}, "至少需要一个"),
])
def test_rejects_invalid_question_set(sample, index, updates, reason):
    path, _, _, rows = sample
    rows[index].update(updates)
    write_rows(path, rows)
    with pytest.raises(ValueError, match=reason):
        load_question_set(path)


@pytest.mark.parametrize(("anchor_update", "reason"), [
    ({"doc_id": "other"}, "来源或 PDF 版本不符"),
    ({"pdf_version": "0" * 64}, "来源或 PDF 版本不符"),
    ({"page_no": 2}, "物理页未核定"),
    ({"quote": "Repeat"}, "原文未唯一匹配"),
    ({"quote": "Status B"}, "原文未唯一匹配"),
    ({"bbox": [22, 22, 23, 23]}, "原文未唯一匹配"),
    ({"bbox": [-1, 0, 12, 12]}, "bbox 超出物理页"),
])
def test_cli_reports_unverifiable_anchor(sample, capsys, anchor_update, reason):
    path, pdf, scopes, rows = sample
    rows[0]["evidence_groups"][0]["anchors"][0].update(anchor_update)
    write_rows(path, rows)
    with pytest.raises(SystemExit) as result:
        main([str(path), "--pdf", str(pdf), "--scopes", str(scopes)])
    assert result.value.code == 1
    assert reason in capsys.readouterr().err


def test_cli_rejects_anchor_for_another_control_type(sample, capsys):
    path, pdf, scopes, rows = sample
    rows[0]["control_type"] = "触控式"
    write_rows(path, rows)
    with pytest.raises(SystemExit) as result:
        main([str(path), "--pdf", str(pdf), "--scopes", str(scopes)])
    assert result.value.code == 1
    assert "锚点操作类型不符" in capsys.readouterr().err
