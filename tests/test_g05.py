"""G05 验收使用自造的故障表和上游 EU。"""

import ast
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import lookup
from lookup import load_codes, lookup_code, verify_quotes
from pipeline.build_codes import build_codes, load_eus, load_source_metadata
from pipeline.build_eu import write_jsonl
from pipeline.layout import Table, load_layout, write_layout
from pipeline.parse import Document, Page, write_audit
from schema import EU, eu_hash

FIXTURES = ROOT / "tests" / "fixtures"
CODES = FIXTURES / "codes_fixture.json"
EUS = FIXTURES / "eu_for_codes.jsonl"
LAYOUT = FIXTURES / "layout_for_codes.json"
FIELDS = ("code", "model_code", "quote", "page_no", "eu_id", "action_level")


def test_exact_lookup():
    codes = load_codes(CODES)
    entry = lookup_code("F4", "B1KA", codes)
    assert isinstance(entry, dict)
    assert tuple(entry) == FIELDS
    assert [row for row in codes if row["code"].upper() == "F4"] == [entry]
    assert entry["action_level"] in {"SAFE", "CAUTION", "P0_DANGER"}
    assert entry["code"] == "f4"
    assert entry["quote"] == "f4 | B1KA |  记录故障 0.15 bar "


def test_model_isolation_and_normalization():
    codes = load_codes(CODES)
    assert lookup_code(" f4 ", " b1ka ", codes) == lookup_code("F4", "B1KA", codes)
    assert lookup_code("F4", "B1HA", codes) is None
    assert lookup_code("ZZ9", "B1KA", codes) is None
    assert lookup_code("F4", "", codes) is None
    assert lookup_code("E5", "B1KA", codes)["quote"] == "E5 | B1KA | 查看显示"
    assert lookup_code("E5", "B1HA", codes)["quote"] == "E5 | B1HA | 燃气相关操作"


def test_quote_verification():
    codes = load_codes(CODES)
    eus = [json.loads(line) for line in EUS.read_text(encoding="utf-8").splitlines()]
    assert verify_quotes(codes, eus) == {"checked": len(codes), "mismatched": [], "rate": 1.0}
    original = lookup_code("F4", "B1KA", codes)
    for changed_fields in (
        {"page_no": 2},
        {"quote": ""},
        {"quote": "  \t"},
        {"quote": original["quote"].replace("0.15", "0.16")},
        {"eu_id": "g05-fixture:p001:99"},
    ):
        invalid = original | changed_fields
        assert verify_quotes([original, invalid], eus) == {
            "checked": 2, "mismatched": [invalid], "rate": 0.5,
        }
    assert verify_quotes([], eus) == {"checked": 0, "mismatched": [], "rate": 0.0}


@pytest.mark.parametrize("variation", [{}, {"code": " F4 ", "model_code": " b1ka "}])
def test_duplicate_normalized_key(tmp_path, variation):
    row = lookup_code("F4", "B1KA", load_codes(CODES))
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([row, row | variation]), encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_CODE"):
        load_codes(path)


def test_default_index_is_reused_and_reloaded(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "codes.json"
    shutil.copyfile(CODES, path)
    monkeypatch.chdir(tmp_path)
    original_load = lookup.load_codes
    loads = []

    def counted_load(path):
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(lookup, "load_codes", counted_load)
    first = lookup_code("F4", "B1KA")
    first["quote"] = "修改返回值"
    assert lookup_code("F4", "B1KA") == lookup_code("F4", "B1KA", load_codes(CODES))
    assert len(loads) == 1

    replacement = data / "new.json"
    replacement.write_text("[]\n", encoding="utf-8")
    replacement.replace(path)
    assert lookup_code("F4", "B1KA") is None
    assert len(loads) == 2


def test_loaded_snapshot_reuses_index(monkeypatch):
    codes = load_codes(CODES)
    assert isinstance(codes, list)

    def unexpected_rebuild(rows):
        pytest.fail("已加载快照不应重新建表")

    monkeypatch.setattr(lookup, "_index_codes", unexpected_rebuild)
    assert lookup_code("F4", "B1KA", codes)["code"] == "f4"
    assert lookup_code("F4", "B1HA", codes) is None
    next(row for row in codes if row["code"] == "f4")["model_code"] = "B1HA"
    assert lookup_code("F4", "B1KA", codes) is None


@pytest.mark.parametrize("code, model", [
    ("A1", "B1KA"), ("A2", "B1KA"), ("A3", "B1KA"), ("A4", "B1KA"), ("E5", "B1HA"),
])
def test_danger_category(code, model):
    assert lookup.DANGER_WORDS is config.DANGER_WORDS
    generated = build_codes(load_layout(LAYOUT).tables, load_eus(EUS))
    assert lookup_code(code, model, generated)["action_level"] == "P0_DANGER"


def test_single_lookup_implementation():
    definitions = []
    for directory in (ROOT, ROOT / "pipeline"):
        for path in directory.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.FunctionDef) and node.name == "lookup_code":
                    definitions.append(path.relative_to(ROOT))
    assert definitions == [Path("lookup.py")]


def test_generator_requires_matching_table_row():
    tables = load_layout(LAYOUT).tables
    eus = load_eus(EUS)
    target = next(eu for eu in eus if eu.eu_id == "g05-fixture:p002:00")
    for changes in ({"page_no": 1}, {"bbox": [10, 25, 140, 30]}, {"is_table_row": False}):
        changed = [replace(eu, **changes) if eu is target else eu for eu in eus]
        with pytest.raises(ValueError, match="第 2 页的故障码 e7 缺少匹配"):
            build_codes(tables, changed)


def test_generator_does_not_infer_model_from_action():
    table = load_layout(LAYOUT).tables[1]
    source = next(eu for eu in load_eus(EUS) if eu.eu_id == "g05-fixture:p002:00")
    table = replace(table, chapter="故障信息", rows=[["e7", "与 B1KA 显示不一致"]])
    source = replace(source, chapter=table.chapter,
                     text_raw=table.header_text + "\ne7 | 与 B1KA 显示不一致")
    assert build_codes([table], [source]) == []


def test_generator_classifies_wiring_synonym():
    table = load_layout(LAYOUT).tables[1]
    source = next(eu for eu in load_eus(EUS) if eu.eu_id == "g05-fixture:p002:00")
    table = replace(table, rows=[["e7", "断开电源导线"]])
    source = replace(source, text_raw=table.header_text + "\ne7 | 断开电源导线")
    assert build_codes([table], [source])[0]["action_level"] == "P0_DANGER"


def _prepare_run(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copyfile(EUS, data / "eu.jsonl")
    audit = tmp_path / "语料审计.md"
    doc = Document(
        "g05-fixture", 2, 2.0,
        [Page(1, "", "page_001.png", True, 0, 140, 120),
         Page(2, "", "page_002.png", False, 0, 140, 120)],
        "g05-fixture.pdf",
    )
    write_audit(doc, audit, has_fault_code_table=True, model_codes_detected="",
                chapter_count=3, table_count=3, eu_count=10, cross_chapter_eu_count=0,
                old_extra="保留")
    return data, audit


def _run_cli(tmp_path, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("DASHSCOPE_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "pipeline.build_codes", *args], cwd=tmp_path,
        env=env, check=True, capture_output=True, text=True,
    )


def test_cli_rebuild_and_audit(tmp_path):
    data, audit = _prepare_run(tmp_path)
    shutil.copyfile(LAYOUT, data / "layout.json")
    prior = audit.read_text(encoding="utf-8").splitlines()
    assert "code_count=8" in _run_cli(tmp_path).stdout
    output = data / "codes.json"
    first_codes, first_audit = output.read_bytes(), audit.read_bytes()
    assert first_codes == CODES.read_bytes()
    assert [(row["code"].upper(), row["model_code"]) for row in load_codes(output)] == [
        ("A1", "B1KA"), ("A2", "B1KA"), ("A3", "B1KA"), ("A4", "B1KA"),
        ("E5", "B1HA"), ("E5", "B1KA"), ("E7", "B1KA"), ("F4", "B1KA"),
    ]
    expected = [
        '| model_codes_detected | ["B1HA", "B1KA"] |'
        if line.startswith("| model_codes_detected |") else line
        for line in prior[:-1]
    ]
    assert audit.read_text(encoding="utf-8").splitlines() == expected + [
        "| code_count | 8 |", "| quote_verify_rate | 1.0 |", prior[-1],
    ]
    _run_cli(tmp_path)
    assert (output.read_bytes(), audit.read_bytes()) == (first_codes, first_audit)


def test_cli_empty_fault_table(tmp_path):
    data, audit = _prepare_run(tmp_path)
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    layout["tables"] = [layout["tables"][-1]]
    (data / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    assert "code_count=0" in _run_cli(tmp_path).stdout
    assert load_codes(data / "codes.json") == []
    text = audit.read_text(encoding="utf-8")
    assert "| code_count | 0 |" in text
    assert "| quote_verify_rate | N/A |" in text
    assert "| model_codes_detected |  |" in text


def _grouped_source(page, action="记录显示。"):
    header = ["原因", "排除故障"]
    cells = ["显示故障代码 06、0C、CC、\nF04 或 E5 闪烁。", action]
    row_bbox = [10, 20, 140, 35]
    header_text = " | ".join(header)
    table = Table(page, [10, 10, 140, 35], header, [cells], [row_bbox], "室内太冷", header_text)
    text = header_text + "\n" + " | ".join(cells)
    doc_id = "g05-fixture"
    eu = EU(f"{doc_id}:p{page:03d}:00", doc_id, page, row_bbox, table.chapter, text, [],
            eu_hash(doc_id, page, row_bbox, text), True)
    return table, eu


def _write_model_scopes(tmp_path, scopes, doc_id="g05-fixture", choices=None):
    path = tmp_path / "model-scopes.json"
    document = {"doc_id": doc_id, "scopes": scopes}
    if choices is not None:
        document["preferred_evidence"] = choices
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_cli_grouped_rows_keep_original_evidence_and_model_scope(tmp_path):
    data, _ = _prepare_run(tmp_path)
    layout = load_layout(LAYOUT)
    table1, eu1 = _grouped_source(1)
    table2, eu2 = _grouped_source(2, "记录显示。\n提示\n仅记录故障码。")
    table3, eu3 = _grouped_source(3, "拆机检查。")
    # 一个普通原因行即便含有类似的编号，也不应生成代码。
    unrelated = replace(table1, rows=[["参见图 F09。", "记录显示。"]])
    layout.tables = [table3, table2, table1, unrelated]
    write_layout(layout, data / "layout.json")
    write_jsonl([eu1, eu2, eu3], data / "eu.jsonl")
    path = _write_model_scopes(
        tmp_path,
        [{"first_page": 1, "last_page": 2, "model_codes": ["B1KA", "L1P20-A1JF"]}],
        choices=[{"alternate_hash": eu1.content_hash, "preferred_hash": eu2.content_hash}],
    )

    assert "code_count=0" in _run_cli(tmp_path).stdout
    assert "code_count=10" in _run_cli(tmp_path, "--model-scopes", str(path)).stdout
    output = data / "codes.json"
    snapshot = output.read_bytes()
    rows = load_codes(output)
    assert {row["code"] for row in rows} == {"06", "0C", "CC", "F04", "E5"}
    assert {row["model_code"] for row in rows} == {"B1KA", "L1P20-A1JF"}
    assert lookup_code("F4", "B1KA", rows) is None
    assert lookup_code("F04", "B1HA", rows) is None
    assert lookup_code("F04", "B1KA", rows) == {
        "code": "F04", "model_code": "B1KA",
        "quote": " | ".join(table2.rows[0]), "page_no": 2, "eu_id": eu2.eu_id,
        "action_level": "SAFE",
    }
    assert verify_quotes(rows, [eu1, eu2, eu3])["rate"] == 1.0
    _run_cli(tmp_path, "--model-scopes", str(path))
    assert output.read_bytes() == snapshot


@pytest.mark.parametrize("second_action", ["允许继续运行。\n提示\n禁止继续运行。", "拆机检查。"])
def test_grouped_conflicting_evidence_fails(tmp_path, second_action):
    table1, eu1 = _grouped_source(1, "允许继续运行。")
    table2, eu2 = _grouped_source(2, second_action)
    path = _write_model_scopes(tmp_path, [{"first_page": 1, "last_page": 2,
                                         "model_codes": ["B1KA"]}])
    metadata = load_source_metadata(path, [eu1, eu2])
    with pytest.raises(ValueError, match=r"DUPLICATE_CODE.*B1KA.*1.*2"):
        build_codes([table1, table2], [eu1, eu2], metadata)


def test_grouped_evidence_cannot_override_different_action_level(tmp_path):
    table1, eu1 = _grouped_source(1)
    table2, eu2 = _grouped_source(2, "拆机检查。")
    path = _write_model_scopes(
        tmp_path, [{"first_page": 1, "last_page": 2, "model_codes": ["B1KA"]}],
        choices=[{"alternate_hash": eu2.content_hash, "preferred_hash": eu1.content_hash}],
    )
    metadata = load_source_metadata(path, [eu1, eu2])
    with pytest.raises(ValueError, match=r"DUPLICATE_CODE.*B1KA"):
        build_codes([table1, table2], [eu1, eu2], metadata)


def test_model_scopes_reject_wrong_document_and_overlapping_ranges(tmp_path):
    _, eu = _grouped_source(1)
    range_ = {"first_page": 1, "last_page": 2, "model_codes": ["B1KA"]}
    path = _write_model_scopes(tmp_path, [range_], doc_id="another")
    with pytest.raises(ValueError, match="doc_id"):
        load_source_metadata(path, [eu])

    path = _write_model_scopes(tmp_path, [
        range_, {"first_page": 2, "last_page": 3, "model_codes": ["B1HA"]},
    ])
    with pytest.raises(ValueError, match="重叠"):
        load_source_metadata(path, [eu])

    path = _write_model_scopes(tmp_path, [range_], choices=[
        {"alternate_hash": "0" * 64, "preferred_hash": eu.content_hash},
    ])
    with pytest.raises(ValueError, match="现有表格行哈希"):
        load_source_metadata(path, [eu])
