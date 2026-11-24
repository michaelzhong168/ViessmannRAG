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
from pipeline.build_codes import build_codes, load_eus
from pipeline.layout import load_layout
from pipeline.parse import Document, Page, write_audit

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


def _run_cli(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("DASHSCOPE_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "pipeline.build_codes"], cwd=tmp_path,
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
