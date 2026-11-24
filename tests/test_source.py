# 验证 PDF 加载的页段边界、原文定位和命令行契约。
import json
from hashlib import sha256

import pymupdf
import pytest
from llama_index.core.schema import Document, MetadataMode

from manual_rag.source import load_document, main


@pytest.fixture
def source_files(tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    # 第 2 页留作越界对照，第 4 页留空以验证空页仍保留物理页编号。
    with pymupdf.open() as pdf:
        for text in ("Approved operation.", "Unapproved section.", "Approved specification.", ""):
            page = pdf.new_page()
            if text:
                page.insert_text((80, 110), text)
        pdf.save(pdf_path)

    scopes_path = tmp_path / "scopes.json"
    scopes = {
        "doc_id": "manual",
        # 倒序登记页段，验证加载结果始终按物理页排序。
        "scopes": [
            {"first_page": 3, "last_page": 4, "model_codes": ["TYPE-A", "TYPE-B"]},
            {"first_page": 1, "last_page": 1, "model_codes": ["TYPE-A"]},
        ],
    }
    scopes_path.write_text(json.dumps(scopes), encoding="utf-8")
    return pdf_path, scopes_path, scopes


def test_load_document_preserves_approved_pages_and_text(source_files):
    pdf_path, scopes_path, _ = source_files
    document = load_document(pdf_path, scopes_path)
    assert isinstance(document, Document)
    assert document.id_ == "manual"
    assert document.metadata["source"] == "manual.pdf"
    assert document.metadata["page_count"] == 4
    pages = document.metadata["pages"]
    assert [page["page_no"] for page in pages] == [1, 3, 4]
    assert [page["model_codes"] for page in pages] == [
        ["TYPE-A"], ["TYPE-A", "TYPE-B"], ["TYPE-A", "TYPE-B"],
    ]
    # 逐页回读 PDF，确认字符范围没有把分隔符或相邻页算入本页原文。
    with pymupdf.open(pdf_path) as original:
        for page in pages:
            source_text = original[page["page_no"] - 1].get_text()
            assert document.text[page["start_char_idx"]:page["end_char_idx"]] == source_text
    assert "Unapproved section." not in document.text
    # 本地元数据用于追溯，提供给模型的内容只含已核定原文。
    assert document.get_content(metadata_mode=MetadataMode.EMBED) == document.text
    assert document.get_content(metadata_mode=MetadataMode.LLM) == document.text


def test_reloading_same_pdf_keeps_fingerprint_and_a_new_version_changes_it(source_files):
    pdf_path, scopes_path, _ = source_files
    initial = load_document(pdf_path, scopes_path)
    fingerprint = initial.metadata["pdf_version"]
    assert fingerprint == sha256(pdf_path.read_bytes()).hexdigest()
    assert load_document(pdf_path, scopes_path).metadata["pdf_version"] == fingerprint

    # 只修改 PDF 元数据，区分文件版本与提取文本的哈希。
    with pymupdf.open(pdf_path) as pdf:
        pdf.set_metadata({"title": "New edition"})
        pdf.saveIncr()

    revised = load_document(pdf_path, scopes_path)
    assert revised.id_ == initial.id_
    assert revised.metadata["pdf_version"] == sha256(pdf_path.read_bytes()).hexdigest()
    assert revised.metadata["pdf_version"] != fingerprint


@pytest.mark.parametrize(("changes", "error"), [
    ({"doc_id": "another-manual"}, "doc_id"),
    ({"scopes": []}, "至少一个"),
    ({"scopes": [None]}, "JSON 对象"),
    # Python 的 bool 是 int 的子类，常规整数检查可能误接收 true。
    ({"scopes": [{"first_page": True, "last_page": 1, "model_codes": ["A"]}]}, "物理页"),
    ({"scopes": [{"first_page": 0, "last_page": 1, "model_codes": ["A"]}]}, "物理页"),
    ({"scopes": [{"first_page": 1, "last_page": 5, "model_codes": ["A"]}]}, "物理页"),
    # 页段含两端，两个范围都包含第 2 页时即为重叠。
    ({"scopes": [{"first_page": 1, "last_page": 2, "model_codes": ["A"]},
                 {"first_page": 2, "last_page": 3, "model_codes": ["B"]}]}, "重叠"),
    ({"scopes": [{"first_page": 1, "last_page": 1, "model_codes": ["A", "A"]}]}, "唯一"),
    ({"scopes": [{"first_page": 1, "last_page": 1, "model_codes": []}]}, "非空"),
])
def test_refuses_invalid_scopes(source_files, changes, error):
    pdf_path, scopes_path, scopes = source_files
    scopes_path.write_text(json.dumps(scopes | changes), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_document(pdf_path, scopes_path)


def test_cli_reports_source_without_printing_text(source_files, capsys):
    pdf_path, scopes_path, _ = source_files
    assert main([str(pdf_path), "--scopes", str(scopes_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "doc_id": "manual",
        "pdf_version": sha256(pdf_path.read_bytes()).hexdigest(),
        "page_count": 4,
        "approved_pages": [1, 3, 4],
        "model_codes": ["TYPE-A", "TYPE-B"],
        "pages_without_text": [4],
    }


def test_cli_reports_a_malformed_pdf(source_files, capsys):
    pdf_path, scopes_path, _ = source_files
    pdf_path.write_bytes(b"invalid PDF")
    assert main([str(pdf_path), "--scopes", str(scopes_path)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("加载失败")
