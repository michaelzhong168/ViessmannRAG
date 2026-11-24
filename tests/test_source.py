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
            {"first_page": 3, "last_page": 4, "control_type": "触控式"},
            {"first_page": 1, "last_page": 1, "control_type": "按键式"},
        ],
    }
    scopes_path.write_text(json.dumps(scopes), encoding="utf-8")
    return pdf_path, scopes_path, scopes


def test_load_document_preserves_approved_pages_and_text(source_files):
    """验证只收录核定物理页，并保留各页操作类型、原文和字符范围。"""
    pdf_path, scopes_path, _ = source_files
    document = load_document(pdf_path, scopes_path)
    assert isinstance(document, Document)
    assert document.id_ == "manual"
    assert document.metadata["source"] == "manual.pdf"
    assert document.metadata["page_count"] == 4
    pages = document.metadata["pages"]
    assert [page["page_no"] for page in pages] == [1, 3, 4]
    assert [page["control_type"] for page in pages] == ["按键式", "触控式", "触控式"]
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
    """验证同一 PDF 的指纹可复现，文件修订后指纹变化且文档 ID 保持稳定。"""
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
    ({"scopes": [{"first_page": True, "last_page": 1, "control_type": "按键式"}]}, "物理页"),
    ({"scopes": [{"first_page": 0, "last_page": 1, "control_type": "按键式"}]}, "物理页"),
    ({"scopes": [{"first_page": 1, "last_page": 5, "control_type": "按键式"}]}, "物理页"),
    # 页段含两端，两个范围都包含第 2 页时即为重叠。
    ({"scopes": [{"first_page": 1, "last_page": 2, "control_type": "按键式"},
                 {"first_page": 2, "last_page": 3, "control_type": "旋钮式"}]}, "重叠"),
    ({"scopes": [{"first_page": 1, "last_page": 1, "control_type": "按钮"}]}, "control_type"),
    ({"scopes": [{"first_page": 1, "last_page": 1, "control_type": None}]}, "control_type"),
])
def test_refuses_invalid_scopes(source_files, changes, error):
    """验证加载器拒绝来源不符、页段无效或操作类型不合规的范围文件。"""
    pdf_path, scopes_path, scopes = source_files
    scopes_path.write_text(json.dumps(scopes | changes), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_document(pdf_path, scopes_path)


def test_cli_reports_source_without_printing_text(source_files, capsys):
    """验证 CLI 输出包含空页信息的来源摘要，且输出中没有手册原文。"""
    pdf_path, scopes_path, _ = source_files
    assert main([str(pdf_path), "--scopes", str(scopes_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "doc_id": "manual",
        "pdf_version": sha256(pdf_path.read_bytes()).hexdigest(),
        "page_count": 4,
        "approved_pages": [1, 3, 4],
        "control_types": ["按键式", "触控式"],
        "pages_without_text": [4],
    }


def test_cli_reports_a_malformed_pdf(source_files, capsys):
    """验证损坏的 PDF 使 CLI 返回输入错误，并把错误提示写入标准错误。"""
    pdf_path, scopes_path, _ = source_files
    pdf_path.write_bytes(b"invalid PDF")
    assert main([str(pdf_path), "--scopes", str(scopes_path)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("加载失败")


def test_whole_manual_can_use_control_type_without_scope_file(source_files, capsys):
    """验证整本说明书可直接声明操作类型，并按 PDF 实际页数加载。"""
    pdf_path, _, _ = source_files
    assert main([str(pdf_path), "--control-type", "旋钮式"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["approved_pages"] == [1, 2, 3, 4]
    assert result["control_types"] == ["旋钮式"]
    document = load_document(pdf_path, control_type="旋钮式")
    assert [page["control_type"] for page in document.metadata["pages"]] == ["旋钮式"] * 4
    assert "Unapproved section." in document.text


def test_cli_rejects_zero_page_pdf(tmp_path, capsys):
    """验证整本声明拒绝物理页树为空的 PDF。"""
    pdf_path = tmp_path / "zero-pages.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    )
    assert main([str(pdf_path), "--control-type", "旋钮式"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "PDF 至少须有一页" in output.err


def test_rejects_ambiguous_scope_options(source_files):
    """验证直接声明与页段文件互斥，防止整本声明绕过已有页段。"""
    pdf_path, scopes_path, _ = source_files
    with pytest.raises(ValueError, match="请选择"):
        load_document(pdf_path, scopes_path, control_type="按键式")
    with pytest.raises(ValueError, match="请选择"):
        load_document(pdf_path)
