"""G03 v1.4 和总纲 v4.2 §2.2 的离线验收测试。"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.layout import analyze_layout, chapter_tree, load_layout, write_layout
from pipeline.parse import RENDER_ZOOM, parse_pdf, write_audit

FIXTURE = ROOT / "tests" / "fixtures" / "layout.pdf"


@pytest.fixture
def document(tmp_path):
    return parse_pdf(FIXTURE, tmp_path / "pages")


@pytest.fixture
def layout(document):
    return analyze_layout(document)


def test_header_footer_removal(document, layout):
    """FR-001：剔除上下页边缘的块，按块计数。"""
    assert layout.header_footer_removed == 2
    assert all(
        "Table of Contents 69" not in item.text and "Running header" not in item.text
        for item in [*layout.headings, *layout.body_blocks]
    )
    unfiltered = analyze_layout(document, top_ratio=0, bottom_ratio=0)
    assert unfiltered.header_footer_removed == 0
    assert {"Table of Contents 69", "Running header"} <= {
        item.text for item in unfiltered.body_blocks
    }


def test_heading_levels(layout):
    """FR-002：验证四档字号的层级顺序和三级上限。"""
    assert [item.text for item in layout.headings] == [
        "Title A", "Title B", "Title C", "Title D", "Title E"
    ]
    levels = [item.level for item in layout.headings]
    assert levels[:4] == [1, 2, 3, 3]
    assert 2 <= len(set(levels)) <= max(levels) <= 3
    assert layout.headings[3].font_size != layout.headings[4].font_size
    assert round(layout.headings[3].font_size, 1) == round(layout.headings[4].font_size, 1)
    assert levels[3] == levels[4]


def test_rounded_sizes_share_a_level(tmp_path):
    """FR-002：在达到层级上限前验证字号舍入。"""
    path = tmp_path / "rounded.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for y, text, size in (
            (110, "First", 24.04), (160, "Second", 24), (220, "Body", 10)
        ):
            page.insert_text((50, y), text, fontsize=size)
        pdf.save(path)
    layout = analyze_layout(parse_pdf(path, tmp_path / "pages"))
    assert [item.level for item in layout.headings] == [1, 1]


def test_chapters_and_json_tree(layout):
    """FR-003：验证正文归入最近的前置标题，章节可跨页延续。"""
    assert {item.text: item.chapter for item in layout.body_blocks} == {
        "Opening body": "(未分章)",
        "Pressure 0.15 bar": "Title A",
        "Second body": "Title B",
        "Rotated body": "Title E",
    }
    assert json.loads(json.dumps(chapter_tree(layout))) == [
        {"title": title, "level": level, "page_no": 1}
        for title, level in (
            ("Title A", 1), ("Title B", 2), ("Title C", 3), ("Title D", 3), ("Title E", 3)
        )
    ]


def test_table_rows_and_header(layout):
    """FR-004：验证线框表的表头和数据行。"""
    assert len(layout.tables) == 2
    table = layout.tables[0]
    assert table.header == ["Fault Code", "Cause"]
    assert table.header_text == "Fault Code | Cause"
    assert table.chapter == "Title E"
    assert table.rows == [["F4", "Low"], ["E5", ""]]
    assert len(table.rows) == len(table.row_bboxes) >= 2
    assert all(len(row) == len(table.header) for row in table.rows)
    assert [dict(zip(table.header, row, strict=True)) for row in table.rows] == [
        {"Fault Code": "F4", "Cause": "Low"},
        {"Fault Code": "E5", "Cause": ""},
    ]
    assert table.row_bboxes == [[50, 370, 190, 400], [50, 400, 190, 430]]


def test_rotated_boxes_match_rendered_page(document, layout):
    """FR-005：对照页图像素及手算旋转结果验证 pt 坐标。"""
    with pymupdf.open(FIXTURE) as pdf:
        page = pdf[1]
        assert page.rotation == 90
        assert list(pymupdf.Rect(20, 400, 60, 430) * page.rotation_matrix) == [
            70, 20, 100, 60
        ]
        word_boxes = [
            pymupdf.Rect(word[:4]) for word in page.get_text("words")
            if word[4] in {"Rotated", "body"}
        ]
        assert len(word_boxes) == 2
        source = word_boxes[0] | word_boxes[1]

    rotated = next(block for block in layout.body_blocks if block.page_no == 2)
    assert rotated.bbox == pytest.approx(
        [500 - source.y1, source.x0, 500 - source.y0, source.x1], abs=0.99
    )
    table = layout.tables[1]
    assert table.bbox == pytest.approx([100, 70, 190, 210], abs=0.99)
    assert table.row_bboxes[0] == pytest.approx([130, 70, 160, 210], abs=0.99)
    assert table.row_bboxes[1] == pytest.approx([100, 70, 130, 210], abs=0.99)

    image = pymupdf.Pixmap(document.pages[1].image_path)
    assert (image.width, image.height) == (1000, 600)
    pixel_box = [round(value * RENDER_ZOOM) for value in rotated.bbox]
    samples = image.samples
    red_pixels = []
    for offset in range(0, len(samples), image.n):
        red, green, blue = samples[offset:offset + 3]
        if red > 200 and green < 80 and blue < 80:
            y, x = divmod(offset // image.n, image.width)
            red_pixels.append((x, y))
    assert red_pixels
    assert all(
        pixel_box[0] <= x <= pixel_box[2] and pixel_box[1] <= y <= pixel_box[3]
        for x, y in red_pixels
    )
    for x0, y0, x1, y1 in table.row_bboxes:
        # 在远离列分隔线和单元格文字的位置取样。
        py = round((y0 + (y1 - y0) / 4) * RENDER_ZOOM)
        for edge in (x0, x1):
            px = round(edge * RENDER_ZOOM)
            assert all(min(image.pixel(px + dx, py)) > 240 for dx in (-2, 2))
            assert min(image.pixel(px, py)) < 100

    for item in [*layout.headings, *layout.tables, *layout.body_blocks]:
        page = document.pages[item.page_no - 1]
        for x0, y0, x1, y1 in [item.bbox, *getattr(item, "row_bboxes", [])]:
            assert 0 <= x0 < x1 <= page.width_pt
            assert 0 <= y0 < y1 <= page.height_pt


def test_stable_layout_roundtrip(document, layout, tmp_path):
    """FR-006：验证列表排序，以及独立提取的 JSON 字节一致性。"""
    for items in (layout.headings, layout.tables, layout.body_blocks):
        keys = [
            (item.page_no, item.bbox[1], item.bbox[0],
             item.header_text if hasattr(item, "header_text") else item.text)
            for item in items
        ]
        assert keys == sorted(keys)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    write_layout(layout, first)
    write_layout(analyze_layout(document), second)
    expected = json.dumps(
        asdict(layout), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert first.read_bytes() == second.read_bytes() == expected
    assert load_layout(first) == layout


@pytest.mark.parametrize(
    ("pdf", "fault_table", "chapters", "tables"),
    [(FIXTURE, "true", 5, 2), (FIXTURE.with_name("mini.pdf"), "false", 0, 0)],
)
def test_cli_updates_audit(pdf, fault_table, chapters, tables, tmp_path, monkeypatch):
    """FR-006：验证 CLI 更新审计并保留原有字段。"""
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    audit = tmp_path / "语料审计.md"
    doc = parse_pdf(pdf, tmp_path / "data/pages")
    write_audit(doc, audit, model_codes_detected=["B1KA"], existing_metric=123)

    def run():
        return subprocess.run(
            [sys.executable, "-m", "pipeline.layout", "--pdf", str(pdf)],
            cwd=tmp_path, check=True, capture_output=True,
        )

    run()
    lines = audit.read_text(encoding="utf-8").splitlines()
    fields = dict(tuple(cell.strip() for cell in line[1:-1].split("|", 1))
                  for line in lines if line.startswith("|"))
    assert {key: fields[key] for key in (
        "doc_id", "page_count", "text_layer_pages", "ocr_required_pages",
        "render_zoom", "has_fault_code_table", "model_codes_detected",
        "chapter_count", "table_count", "existing_metric",
    )} == {
        "doc_id": pdf.stem, "page_count": "2", "text_layer_pages": "[1, 2]",
        "ocr_required_pages": "[]", "render_zoom": "2.0",
        "has_fault_code_table": fault_table, "model_codes_detected": '["B1KA"]',
        "chapter_count": str(chapters), "table_count": str(tables), "existing_metric": "123",
    }
    output = tmp_path / "data/layout.json"
    result = load_layout(output)
    assert len(result.headings) == chapters
    assert len(result.tables) == tables
    before = (output.read_bytes(), audit.read_bytes())
    run()
    assert (output.read_bytes(), audit.read_bytes()) == before
