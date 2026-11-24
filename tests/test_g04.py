"""G04 v1.4 和共享总纲 v4.2 §2.2 的离线验收。"""

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.build_eu import build_eu, write_jsonl
from pipeline.layout import Block, Layout, analyze_layout, load_layout
from pipeline.parse import parse_pdf
from schema import eu_hash, validate_eu

FIXTURES = ROOT / "tests/fixtures"
PDF = FIXTURES / "layout.pdf"


@pytest.fixture
def document(tmp_path):
    return parse_pdf(PDF, tmp_path / "pages")


@pytest.fixture
def layout():
    return load_layout(FIXTURES / "layout_fixture.json")


@pytest.fixture
def eus(layout, document):
    return build_eu(layout, document)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("eu_id", "layout:p001:0"), ("eu_id", "other:p001:00"),
        ("eu_id", "layout:p002:00"), ("doc_id", ""), ("page_no", 0),
        ("page_no", True), ("bbox", [0, 0, 1]), ("bbox", [2, 0, 1, 1]),
        ("bbox", [0, 1, 1, 1]), ("bbox", [0, 0, float("nan"), 1]),
        ("bbox", [0, 0, 1, float("inf")]), ("bbox", [False, 0, 1, 1]),
        ("chapter", ""), ("text_raw", ""), ("model_codes", "B1KA"),
        ("model_codes", [4]), ("content_hash", "A" * 64),
        ("content_hash", "a" * 63), ("is_table_row", 1),
    ],
)
def test_validate_eu_rejects_invalid_fields(eus, field, bad):
    """FR-001：每个坏值独立触发对应字段的错误。"""
    record = asdict(eus[0])
    assert validate_eu(record) == []
    record[field] = bad
    assert any(field in error for error in validate_eu(record))


def test_validate_eu_requires_all_fields(eus):
    for field in asdict(eus[0]):
        record = asdict(eus[0])
        del record[field]
        assert f"缺少字段 {field}" in validate_eu(record)


def test_ids_and_hashes(eus):
    """FR-002：固定向量、独立全集重算及同页索引。"""
    assert eu_hash("doc", 1, [72, 318, 240, 360], "中文") == (
        "4111da25a3f5c797140072812b14ed6d67a6dbc819f7976bf93dc1400d340951"
    )
    assert [eu.eu_id for eu in eus] == [
        *(f"layout:p001:{index:02d}" for index in range(6)),
        "layout:p002:00", "layout:p002:01",
    ]
    assert len({eu.eu_id for eu in eus}) == len(eus)
    for eu in eus:
        bbox = ",".join(format(value, ".2f") for value in eu.bbox)
        payload = f"{PDF.stem}|{eu.page_no}|[{bbox}]|{eu.text_raw}"
        assert eu.doc_id == PDF.stem
        assert re.fullmatch("[0-9a-f]{64}", eu.content_hash)
        assert eu.content_hash == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_rebuild_is_deterministic(layout, document, tmp_path):
    """FR-003：重载相同输入后，EU 对象及 JSONL 字节完全相同。"""
    path = tmp_path / "nested/eu.jsonl"
    first = build_eu(layout, document)
    write_jsonl(first, path)
    initial = path.read_bytes()
    second = build_eu(load_layout(FIXTURES / "layout_fixture.json"), document)
    assert first == second
    write_jsonl(second, path)
    assert path.read_bytes() == initial


def test_chapters_and_verbatim_rows(layout, eus):
    """FR-004、FR-005：正文逐块保真，表头仅前置一次。"""
    body = [eu for eu in eus if not eu.is_table_row]
    assert body[0].chapter != body[1].chapter
    assert [(eu.chapter, eu.text_raw, eu.bbox) for eu in body] == [
        (block.chapter, block.text, block.bbox) for block in layout.body_blocks
    ]
    rows = [eu for eu in eus if eu.is_table_row]
    assert len(rows) == sum(len(table.rows) for table in layout.tables)
    for eu, (table, cells, bbox) in zip(rows, (
        (table, cells, bbox)
        for table in layout.tables
        for cells, bbox in zip(table.rows, table.row_bboxes, strict=True)
    ), strict=True):
        assert eu.chapter == table.chapter
        assert eu.bbox == bbox
        assert eu.text_raw == table.header_text + "\n" + " | ".join(cells)
        assert eu.text_raw.count(table.header_text + "\n") == 1


@pytest.mark.parametrize(
    ("text", "lengths"),
    [
        ("甲" * 100 + "。" + "乙" * 640 + "！" + "\n" + "丙" * 200, [742, 201]),
        ("甲" * 600 + "\n" + "乙" * 100 + "\n" + "丙" * 200, [702, 200]),
        ("甲" * 800 + "。" + "乙" * 20, [800, 21]),
        ("甲" * 799 + "?" + "乙" * 20, [800, 20]),
        ("甲" * 798 + "12345" + "乙" * 20, [798, 25]),
        ("甲" * 798 + "12.45" + "乙" * 20, [798, 25]),
        ("甲" * 795 + "L1P20-A1JF" + "乙" * 20, [795, 30]),
        ("1" * 830 + "乙" * 20, [800, 50]),
        ("甲" * 800, [800]),
    ],
)
def test_long_blocks_preserve_text_and_break_rules(document, text, lengths):
    """FR-004：固定分隔符优先级、硬切和 token 保护，逐字复原。"""
    block = Block(1, [10, 10, 30, 20], text, 10, "原章节")
    eus = build_eu(Layout([], [], [block], 0), document)
    assert [len(eu.text_raw) for eu in eus] == lengths
    assert all(eu.chapter == block.chapter and eu.bbox == block.bbox for eu in eus)
    assert all(0 < len(eu.text_raw) <= 800 for eu in eus)
    assert "".join(eu.text_raw for eu in eus) == text


def test_model_codes_only_from_eu_sources(eus):
    """FR-008：标题、表头、正文和表格行中的型号严格限定作用域。"""
    known = json.loads((FIXTURES / "model_codes_fixture.json").read_text())
    assert known == ["B1KA", "B1HA", "L1P20-A1JF", "L1P24-A1JF"]
    assert [eu.model_codes for eu in eus] == [
        [known[0]], [], known[2:], [], [known[1]], [known[1]], [known[0]], [],
    ]


def test_rotated_boxes_match_layout(document):
    """FR-001：旋转页的行框按 G03 索引逐一匹配，正文框沿用块框。"""
    layout = analyze_layout(document)
    eus = [eu for eu in build_eu(layout, document) if eu.page_no == 2]
    with pymupdf.open(PDF) as pdf:
        assert pdf[1].rotation == 90
    table, = [table for table in layout.tables if table.page_no == 2]
    assert [eu.bbox for eu in eus if eu.is_table_row] == table.row_bboxes
    block, = [block for block in layout.body_blocks if block.page_no == 2]
    rotated, = [eu for eu in eus if not eu.is_table_row]
    assert rotated.text_raw == block.text
    assert rotated.bbox == block.bbox


def test_cli_jsonl_and_audit(tmp_path):
    """FR-001、FR-006：从 G03 CLI 到 G04 CLI 全链路，旧审计字段仍可消费。"""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    def run(*args):
        subprocess.run(
            [sys.executable, "-m", *args], cwd=tmp_path, env=env,
            check=True, capture_output=True,
        )

    run("pipeline.layout", "--pdf", str(PDF))
    audit = tmp_path / "语料审计.md"
    previous_audit = audit.read_text(encoding="utf-8")
    assert "| table_count | 2 |" in previous_audit
    assert "| chapter_count | 5 |" in previous_audit
    run("pipeline.build_eu", "--layout", "data/layout.json", "--pdf", str(PDF))
    output = tmp_path / "data/eu.jsonl"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    with pymupdf.open(PDF) as pdf:
        for record in records:
            assert len(record) == 9
            assert validate_eu(record) == []
            assert 1 <= record["page_no"] <= len(pdf)
            rect = pdf[record["page_no"] - 1].rect
            x0, y0, x1, y1 = record["bbox"]
            assert 0 <= x0 < x1 <= rect.width
            assert 0 <= y0 < y1 <= rect.height
    report = audit.read_text(encoding="utf-8")
    assert set(previous_audit.splitlines()) <= set(report.splitlines())
    assert f"| eu_count | {len(records)} |" in report
    assert "| cross_chapter_eu_count | 0 |" in report
    assert report == (
        previous_audit
        + f"| eu_count | {len(records)} |\n"
        + "| cross_chapter_eu_count | 0 |\n"
    )

    first = output.read_bytes(), audit.read_bytes()
    run("pipeline.build_eu", "--layout", "data/layout.json", "--pdf", str(PDF))
    assert (output.read_bytes(), audit.read_bytes()) == first
