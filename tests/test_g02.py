"""G02 验收测试：PDF 转文本与逐页图（对应规格 v1.3 FR-001…FR-006）。

门禁只用自造夹具 ``tests/fixtures/mini.pdf``，不依赖真实 PDF / 网络 / 密钥；
真实语料的抽页对照属 §七 人工确认项，不进入任何断言。
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
import socket
import struct
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.parse import (  # noqa: E402  （需要先把仓库根塞进 sys.path）
    RENDER_ZOOM,
    Document,
    Page,
    parse_pdf,
    write_audit,
)

FIXTURE = ROOT / "tests" / "fixtures" / "mini.pdf"

#: 夹具两页的期望文本（与 ``tests/fixtures/make_mini_pdf.py`` 逐字一致）。
S1 = "第一页的测试句子：中文标点，不应算乱码。"
S2 = "第二页的测试句子：按 2.0 倍渲染，宽度 595.276 pt。"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: FR-005.6 结构扫描的写文件 API（启发式；见计划 §6.h 备注）。
WRITE_APIS = ("open(", "write_text(", "write_bytes(")

#: FR-005.6 全仓扫描时跳过的目录（.venv / 文档 / 数据 / 缓存等）。
SCAN_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "docs",
    "notes",
    "data",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}


# ────────────────────────── 测试辅助 ──────────────────────────


def _png_size(path: str | Path) -> tuple[int, int]:
    """自写 IHDR 解析：PNG 签名 + 大端 ``bytes[16:24]`` 的宽高。

    不用 PIL / numpy（本关不引入这两个依赖）。
    """
    raw = Path(path).read_bytes()
    assert len(raw) >= 24, f"PNG 文件过短，无法读取 IHDR：{path}"
    assert raw[:8] == PNG_SIGNATURE, f"不是合法 PNG 签名：{path}"
    return struct.unpack(">II", raw[16:24])


def _read_audit(path: str | Path) -> dict[str, object]:
    """把 ``语料审计.md`` 的 markdown 表解析成 dict（列表/布尔按 JSON 还原）。"""
    values: dict[str, object] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if len(cells) != 2:
            continue
        key, raw = cells
        if not key or key == "字段":
            continue
        if set(key) <= {"-", ":", " "}:
            continue
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    return values


# ────────────────────── FR-001 文档级结构 ──────────────────────


def test_document_page_count_equals_len_pages(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    assert doc.page_count == 2, f"夹具 2 页，page_count 应为 2，实际 {doc.page_count}"
    assert len(doc.pages) == 2, f"夹具 2 页，len(pages) 应为 2，实际 {len(doc.pages)}"
    assert doc.page_count == len(doc.pages), "page_count 必须等于 len(pages)"


def test_page_numbers_are_one_based_and_continuous(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    assert [page.page_no for page in doc.pages] == [1, 2]


def test_document_contract_fields_and_render_zoom(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    names = tuple(field.name for field in fields(Document))
    assert names == ("doc_id", "page_count", "render_zoom", "pages", "pdf_path"), names
    assert doc.render_zoom == 2.0
    assert Path(doc.pdf_path).resolve() == FIXTURE.resolve()
    assert RENDER_ZOOM == 2.0
    assert doc.render_zoom == RENDER_ZOOM
    assert isinstance(doc.doc_id, str) and doc.doc_id
    assert isinstance(doc.pages, list)


def test_page_contract_fields(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    names = tuple(field.name for field in fields(Page))
    assert names == (
        "page_no",
        "text",
        "image_path",
        "text_layer",
        "garbled_ratio",
        "width_pt",
        "height_pt",
    ), names

    for page in doc.pages:
        assert type(page.page_no) is int
        assert isinstance(page.text, str)
        assert isinstance(page.image_path, str) and page.image_path
        assert type(page.text_layer) is bool
        assert isinstance(page.garbled_ratio, float)
        assert 0.0 <= page.garbled_ratio <= 1.0
        assert isinstance(page.width_pt, float) and page.width_pt > 0
        assert isinstance(page.height_pt, float) and page.height_pt > 0


# ────────────────────── FR-002 逐页渲染 PNG ──────────────────────


def test_page_png_files_exist_nonempty_and_png_decodable(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    for page in doc.pages:
        image = Path(page.image_path)
        assert image.is_file(), f"页图不存在：{image}"
        assert image.stat().st_size > 0, f"页图是空文件：{image}"
        width, height = _png_size(image)
        assert width > 0 and height > 0, f"IHDR 宽高非法：{width}x{height}"


def test_png_pixel_size_matches_pt_times_render_zoom(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    for page in doc.pages:
        width, height = _png_size(page.image_path)
        expected_width = page.width_pt * 2.0
        expected_height = page.height_pt * 2.0
        assert abs(width - expected_width) <= 1, (
            f"第 {page.page_no} 页宽 {width}px 与 pt×zoom={expected_width} 偏差 > 1px"
        )
        assert abs(height - expected_height) <= 1, (
            f"第 {page.page_no} 页高 {height}px 与 pt×zoom={expected_height} 偏差 > 1px"
        )


def test_default_out_dir_writes_data_pages_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)

    doc = parse_pdf(FIXTURE)

    for page in doc.pages:
        expected = ROOT / "data" / "pages" / f"page_{page.page_no:03d}.png"
        assert expected.is_file(), f"默认路径页图不存在：{expected}"
        assert page.image_path == f"data/pages/page_{page.page_no:03d}.png"
        assert Path(page.image_path).is_file()


# ────────────────────── FR-003 文本层提取 ──────────────────────


def test_fixture_sentence_on_its_own_page(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    assert S1 in doc.pages[0].text, "第 1 页缺少夹具句子 S1"
    assert S2 not in doc.pages[0].text, "S2 不应出现在第 1 页"
    assert S2 in doc.pages[1].text, "第 2 页缺少夹具句子 S2"
    assert S1 not in doc.pages[1].text, "S1 不应出现在第 2 页"


def test_text_layer_true_for_both_fixture_pages(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    for page in doc.pages:
        assert page.text_layer is True, f"第 {page.page_no} 页 text_layer 应为 True"


def test_garbled_ratio_within_range_and_zero_for_normal_text(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    # 规格 §五 的阈值是 0.15，不能只依赖夹具（正常文本的任何阈值都为 True）。
    parse_module = importlib.import_module("pipeline.parse")
    assert parse_module.GARBLED_RATIO_THRESHOLD == 0.15

    for page in doc.pages:
        assert 0.0 <= page.garbled_ratio <= 1.0
        # 中文标点（：，。）与常规空白属于白名单 W，不得算乱码。
        assert page.garbled_ratio == 0.0, (
            f"第 {page.page_no} 页正常中文文本的乱码率应为 0.0，"
            f"实际 {page.garbled_ratio}"
        )


def test_garbled_ratio_counts_replacement_and_control_chars():
    parse_module = importlib.import_module("pipeline.parse")
    ratio = parse_module._garbled_ratio

    assert ratio("a\ufffd") == 0.5
    assert ratio("中\x00") == 0.5
    assert ratio("") == 0.0
    assert ratio("中文，。：") == 0.0

    # 白名单 W 的正例（真实语料里出现过的字符）与反例（私用区）。
    assert ratio("■□●○") == 0.0
    assert ratio("“”„–—…°±×µ®") == 0.0
    assert ratio("Ⅰ Ⅱ ½ №") == 0.0
    assert ratio("中\ue000") == 0.5


def test_blank_page_is_ocr_required(tmp_path):
    pdf_path = tmp_path / "blank_page.pdf"
    pdf = pymupdf.open()
    try:
        first = pdf.new_page(width=595.276, height=841.89)
        first.insert_text((72, 72), S1, fontname="china-s", fontsize=12)
        pdf.new_page(width=595.276, height=841.89)  # 第 2 页空白
        pdf.save(str(pdf_path), no_new_id=True)
    finally:
        pdf.close()

    doc = parse_pdf(pdf_path, tmp_path / "pages")

    assert doc.pages[0].text_layer is True
    assert doc.pages[1].text == ""
    assert doc.pages[1].text_layer is False
    assert doc.pages[1].garbled_ratio == 0.0

    audit_path = tmp_path / "audit.md"
    write_audit(doc, path=audit_path)
    audit = _read_audit(audit_path)
    assert audit["ocr_required_pages"] == [2]
    assert audit["text_layer_pages"] == [1]


def test_text_layer_threshold_boundary_is_strict():
    parse_module = importlib.import_module("pipeline.parse")
    is_text_layer = parse_module._is_text_layer

    assert is_text_layer("字", 0.149) is True
    assert is_text_layer("字", 0.15) is False  # 规格是严格 < 0.15
    assert is_text_layer("", 0.0) is False
    assert is_text_layer(" \n\t", 0.0) is False


def test_high_garbled_page_is_ocr_required(tmp_path):
    """端到端咬住 0.15 阈值：0.1905 判 OCR、0.1429 判有文本层。"""
    pdf_path = tmp_path / "garbled_threshold.pdf"
    page_texts = ("a" * 16 + "€" * 4, "a" * 17 + "€" * 3)
    pdf = pymupdf.open()
    try:
        for content in page_texts:
            page = pdf.new_page(width=595.276, height=841.89)
            page.insert_text((72, 72), content, fontname="china-s", fontsize=12)
        pdf.save(str(pdf_path), no_new_id=True)
    finally:
        pdf.close()

    doc = parse_pdf(pdf_path, tmp_path / "pages")

    first, second = doc.pages
    assert len(first.text) == 21
    assert len(second.text) == 21
    assert first.garbled_ratio > 0.15
    assert second.garbled_ratio < 0.15
    assert first.text_layer is False
    assert second.text_layer is True

    audit_path = tmp_path / "audit.md"
    write_audit(doc, path=audit_path)
    audit = _read_audit(audit_path)
    assert audit["ocr_required_pages"] == [1]
    assert audit["text_layer_pages"] == [2]


# ────────────────────── FR-004 解析幂等 ──────────────────────


def test_parse_pdf_is_idempotent(tmp_path):
    first = parse_pdf(FIXTURE, tmp_path)
    second = parse_pdf(FIXTURE, tmp_path)

    assert first.doc_id == second.doc_id
    assert first.page_count == second.page_count == len(second.pages)
    for left, right in zip(first.pages, second.pages):
        assert left.page_no == right.page_no
        assert left.text == right.text
        assert left.garbled_ratio == right.garbled_ratio
        assert left.width_pt == right.width_pt
        assert left.height_pt == right.height_pt


# ────────────────────── FR-005 语料审计报告 ──────────────────────


def test_audit_contains_seven_required_fields(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    audit_path = tmp_path / "audit.md"

    write_audit(doc, path=audit_path)
    audit = _read_audit(audit_path)

    required = {
        "doc_id",
        "page_count",
        "text_layer_pages",
        "ocr_required_pages",
        "render_zoom",
        "has_fault_code_table",
        "model_codes_detected",
    }
    missing = required - set(audit)
    assert not missing, f"审计报告缺少字段：{sorted(missing)}"

    # 旧报告坏掉时 write_audit 必须忽略旧内容、全量重写，且不抛异常（计划 §6.c）。
    audit_path.write_text("这不是 markdown 表格\n| broken | row\n", encoding="utf-8")
    write_audit(doc, path=audit_path)
    assert not (required - set(_read_audit(audit_path)))


def test_audit_g02_placeholder_values(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    audit_path = tmp_path / "audit.md"

    write_audit(doc, path=audit_path)
    audit = _read_audit(audit_path)

    assert audit["has_fault_code_table"] == "unknown"
    assert isinstance(audit["model_codes_detected"], str)
    assert audit["model_codes_detected"].strip() == ""


def test_audit_doc_fields_match_document(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    audit_path = tmp_path / "audit.md"

    write_audit(doc, path=audit_path)
    audit = _read_audit(audit_path)

    assert audit["doc_id"] == doc.doc_id
    assert audit["page_count"] == doc.page_count
    assert audit["render_zoom"] == doc.render_zoom == RENDER_ZOOM
    assert audit["text_layer_pages"] == [1, 2]
    assert audit["ocr_required_pages"] == []


def test_audit_default_path_at_repo_root(tmp_path, monkeypatch):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    monkeypatch.chdir(ROOT)

    write_audit(doc)

    audit_path = ROOT / "语料审计.md"
    assert audit_path.is_file()
    content = audit_path.read_text(encoding="utf-8")
    for key in (
        "doc_id",
        "page_count",
        "text_layer_pages",
        "ocr_required_pages",
        "render_zoom",
        "has_fault_code_table",
        "model_codes_detected",
    ):
        assert key in content, f"默认审计报告缺少字段名：{key}"


def test_audit_merge_preserves_downstream_fields(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    audit_path = tmp_path / "audit.md"

    write_audit(doc, path=audit_path)
    first = _read_audit(audit_path)
    assert set(first) >= {
        "doc_id",
        "page_count",
        "text_layer_pages",
        "ocr_required_pages",
        "render_zoom",
        "has_fault_code_table",
        "model_codes_detected",
    }

    # ② 模拟 G03/G04 追加新字段
    write_audit(doc, path=audit_path, chapter_count=3, table_count=2)

    # ③ 不传任何字段重调：旧字段（含下游追加）必须仍在
    write_audit(doc, path=audit_path)
    third = _read_audit(audit_path)
    assert third["chapter_count"] == 3
    assert third["table_count"] == 2
    assert set(first) <= set(third)

    # ④ 显式覆盖（**fields 覆盖一切）
    write_audit(
        doc, path=audit_path, has_fault_code_table=True, model_codes_detected="F4"
    )
    fourth = _read_audit(audit_path)
    assert fourth["has_fault_code_table"] is True
    assert fourth["model_codes_detected"] == "F4"
    assert fourth["chapter_count"] == 3
    assert fourth["table_count"] == 2
    assert fourth["doc_id"] == doc.doc_id

    # **fields 连 doc 派生字段也覆盖；不传时下次重跑恢复派生值。
    write_audit(doc, path=audit_path, doc_id="override-doc")
    assert _read_audit(audit_path)["doc_id"] == "override-doc"
    write_audit(doc, path=audit_path)
    assert _read_audit(audit_path)["doc_id"] == doc.doc_id == "mini"


def test_audit_regenerates_over_hand_edits(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    audit_path = tmp_path / "audit.md"

    write_audit(doc, path=audit_path)

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    mutated: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("| doc_id |"):
            mutated.append("| doc_id | HACKED |")
            replaced = True
        else:
            mutated.append(line)
    assert replaced, "测试未能改坏 doc_id 行"
    audit_path.write_text("\n".join(mutated) + "\n", encoding="utf-8")
    assert "HACKED" in audit_path.read_text(encoding="utf-8")

    write_audit(doc, path=audit_path)

    recovered = audit_path.read_text(encoding="utf-8")
    assert "HACKED" not in recovered, "重跑 write_audit 后手改值必须被覆盖"
    assert _read_audit(audit_path)["doc_id"] == doc.doc_id == "mini"


def test_audit_filename_only_in_write_audit_default():
    default = inspect.signature(write_audit).parameters["path"].default
    assert default == "语料审计.md", (
        f"write_audit 的默认路径应为 语料审计.md，实际 {default!r}"
    )

    write_audit_source = (ROOT / "pipeline" / "parse.py").resolve()
    offenders: list[str] = []
    for source in sorted(ROOT.rglob("*.py")):
        relative = source.relative_to(ROOT)
        if any(part in SCAN_EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if source.resolve() == write_audit_source:
            continue  # 唯一生成者，允许出现默认文件名与写入代码
        for lineno, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "语料审计" in line and any(api in line for api in WRITE_APIS):
                offenders.append(f"{relative}:{lineno}")
    assert not offenders, f"发现 write_audit 之外的手写审计代码：{offenders}"


def test_audit_is_byte_deterministic(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")
    first = tmp_path / "audit_a.md"
    second = tmp_path / "audit_b.md"

    write_audit(doc, path=first)
    write_audit(doc, path=second)
    assert first.read_bytes() == second.read_bytes(), "同一 doc 两次输出字节不一致"

    write_audit(doc, path=first)
    assert first.read_bytes() == second.read_bytes(), "同一路径重写后字节不一致"


# ────────────────────── S-1…S-6 补充断言 ──────────────────────


def test_parse_pdf_does_not_open_network(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("G02 解析过程禁止联网")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    doc = parse_pdf(FIXTURE, tmp_path / "pages")

    assert doc.page_count == 2
    assert len(doc.pages) == 2


def test_parse_writes_only_expected_files(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "page_001.png",
        "page_002.png",
    }

    write_audit(doc, path=tmp_path / "audit.md")

    assert {path.name for path in tmp_path.iterdir()} == {
        "page_001.png",
        "page_002.png",
        "audit.md",
    }


def test_requirements_pins_installed_pymupdf():
    installed = importlib.metadata.version("pymupdf")
    pinned = f"pymupdf=={installed}"
    lines = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    }
    assert pinned in lines, f"requirements.txt 缺少精确钉版本行：{pinned}"


def test_make_mini_pdf_module_regenerates_fixture(tmp_path):
    out = tmp_path / "mini_generated.pdf"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "-m", "tests.fixtures.make_mini_pdf", "--out", str(out)],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"夹具生成器退出码 {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert out.is_file()

    pdf = pymupdf.open(str(out))
    try:
        assert pdf.page_count == 2
        assert pdf[0].get_text().strip() == S1
        assert pdf[1].get_text().strip() == S2
    finally:
        pdf.close()


def test_doc_id_derived_from_filename_stem(tmp_path):
    doc = parse_pdf(FIXTURE, tmp_path / "pages")

    assert isinstance(doc.doc_id, str) and doc.doc_id
    assert doc.doc_id == Path(FIXTURE).stem == "mini"


def test_pipeline_module_exports():
    parse_module = importlib.import_module("pipeline.parse")

    assert parse_module.Document is Document
    assert parse_module.Page is Page
    assert parse_module.parse_pdf is parse_pdf
    assert parse_module.write_audit is write_audit
    assert parse_module.RENDER_ZOOM == RENDER_ZOOM == 2.0
