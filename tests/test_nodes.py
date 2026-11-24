"""使用合成说明书验证 Node 的版面、来源和核定范围。"""

import json

import pymupdf
import pytest
from llama_index.core.schema import MetadataMode, TextNode

from manual_rag.nodes import build_nodes
from manual_rag.source import load_document


@pytest.fixture
def manual(tmp_path):
    path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 70), "Heating", fontsize=18)
        page.insert_text((60, 105), ["Check the display.", "Read the desired setting."])
        page.insert_text((60, 155), ["1. Select heating.", "2. Confirm the setting."])
        for x in (60, 160, 260, 360):
            page.draw_line((x, 220), (x, 280))
        for y in (220, 250, 280):
            page.draw_line((60, y), (360, y))
        page.insert_text((65, 240), "Parameter")
        page.insert_text((265, 240), "Unit")
        page.insert_text((65, 270), "Pressure")
        page.insert_text((165, 270), "1.25")
        page.insert_text((265, 270), "bar")
        pdf.new_page().insert_text((60, 70), "Unapproved title", fontsize=18)
        pdf.new_page().insert_text((60, 105), "Touchscreen only.")
        pdf.new_page().insert_text((60, 105), "Continued without a heading.")
        pdf.new_page()
        pdf.set_toc([[1, "Heating", 1], [1, "Unapproved title", 2]])
        pdf.save(path)
    scopes = tmp_path / "scopes.json"
    scopes.write_text(json.dumps({
        "doc_id": path.stem,
        "scopes": [
            {"first_page": 1, "last_page": 1, "control_type": "按键式"},
            {"first_page": 3, "last_page": 3, "control_type": "触控式"},
            {"first_page": 4, "last_page": 5, "control_type": "按键式"},
        ],
    }), encoding="utf-8")
    return path, scopes


def test_layout_nodes_preserve_text_units_and_locations(manual):
    """逐字核对段落、分立步骤、表格行、空单元格和原页坐标。"""
    path, scopes = manual
    nodes = build_nodes(load_document(path, scopes), path)
    first_page = [node for node in nodes if node.metadata["page_no"] == 1]
    assert [(node.metadata["kind"], node.text) for node in first_page] == [
        ("heading", "Heating"),
        ("paragraph", "Check the display.\nRead the desired setting."),
        ("step", "1. Select heating."),
        ("step", "2. Confirm the setting."),
        ("table_row", "Parameter\t\tUnit"),
        ("table_row", "Pressure\t1.25\tbar"),
    ]
    with pymupdf.open(path) as pdf:
        for node in first_page:
            assert node.metadata["section"] == "Heating"
            assert isinstance(node, TextNode)
            rect = pymupdf.Rect(node.metadata["bbox"])
            assert not rect.is_empty
            assert rect in pdf[0].rect
            excerpt = pdf[0].get_text(clip=rect)
            assert excerpt.split() == node.text.split()
    content = first_page[-1].get_content(metadata_mode=MetadataMode.LLM)
    assert content == "section: Heating\n\nPressure\t1.25\tbar"


def test_page_scope_and_section_context_are_isolated(manual):
    """核定空页不生成节点，缺口及类型切换不会携带其他页的标题。"""
    path, scopes = manual
    nodes = build_nodes(load_document(path, scopes), path)
    assert {node.metadata["page_no"] for node in nodes} == {1, 3, 4}
    assert [(node.text, node.metadata["section"], node.metadata["control_type"])
            for node in nodes if node.metadata["page_no"] > 1] == [
        ("Touchscreen only.", "", "触控式"),
        ("Continued without a heading.", "", "按键式"),
    ]
    assert {node.metadata["control_type"] for node in nodes if node.metadata["page_no"] == 1} == {"按键式"}


def test_conflicting_page_types_are_rejected(manual):
    """即使传入包含重复物理页和相异操作类型的 Document，也整体拒绝生成。"""
    path, scopes = manual
    document = load_document(path, scopes)
    document.metadata["pages"].append(document.metadata["pages"][0] | {"control_type": "旋钮式"})
    with pytest.raises(ValueError, match="物理页 1.*核定记录重复"):
        build_nodes(document, path)


def test_source_identity_and_node_ids_survive_round_trip(manual, tmp_path):
    """重建和序列化保留来源关系，另一来源或文件版本使用独立的 Node ID。"""
    path, scopes = manual
    document = load_document(path, scopes)
    nodes = build_nodes(document, path)
    ids = [node.node_id for node in nodes]
    assert len(set(ids)) == len(nodes)
    assert ids == [node.node_id for node in build_nodes(load_document(path, scopes), path)]
    for node in nodes:
        restored = TextNode.from_json(node.to_json())
        assert restored == node
        assert restored.source_node.node_id == document.id_
        assert restored.source_node.metadata == {
            "source": path.name,
            "pdf_version": document.metadata["pdf_version"],
        }

    other_path = tmp_path / "other.pdf"
    other_path.write_bytes(path.read_bytes())
    other = load_document(other_path, control_type="按键式")
    other_nodes = build_nodes(other, other_path)
    assert all(node.source_node.node_id == other.id_ for node in other_nodes)
    assert set(ids).isdisjoint(node.node_id for node in other_nodes)

    with pymupdf.open(path) as pdf:
        pdf.set_metadata({"title": "Second edition"})
        pdf.saveIncr()
    revised = build_nodes(load_document(path, scopes), path)
    assert [node.text for node in revised] == [node.text for node in nodes]
    assert set(ids).isdisjoint(node.node_id for node in revised)


def test_rejects_mismatched_source(manual, tmp_path):
    """验证同名新版 PDF 和复制为另一文件名的原版均不能冒充来源。"""
    path, scopes = manual
    document = load_document(path, scopes)
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="PDF 来源或文件指纹"):
        build_nodes(document, copy)

    with pymupdf.open(path) as pdf:
        pdf.set_metadata({"title": "Different fingerprint"})
        pdf.saveIncr()
    with pytest.raises(ValueError, match="PDF 来源或文件指纹"):
        build_nodes(document, path)


def test_rotated_page_coordinates_use_unrotated_pdf_space(manual):
    """旋转显示方向后，正文和表格仍能用原始未旋转坐标回查。"""
    path, scopes = manual
    original = build_nodes(load_document(path, scopes), path)
    with pymupdf.open(path) as pdf:
        pdf[0].set_rotation(90)
        pdf.saveIncr()
    rotated = build_nodes(load_document(path, scopes), path)
    assert [(node.text, node.metadata["bbox"]) for node in rotated] == [
        (node.text, node.metadata["bbox"]) for node in original
    ]


def test_same_page_columns_keep_their_own_section(tmp_path):
    """左右栏各沿用本栏标题，通栏标题可更新两栏上下文。"""
    path = tmp_path / "columns.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "Overview of all available operating controls", fontsize=18)
        page.insert_text((315, 68), "Read the display first.")
        page.insert_text((50, 90), "Heating", fontsize=16)
        page.insert_text((305, 95), "Water pressure", fontsize=16)
        page.insert_text((55, 120), "1. Select heating.")
        page.insert_text((315, 120), "1. Read the pressure.")
        page.insert_text((50, 160), "Common status messages for all operating controls", fontsize=18)
        page.insert_text((315, 185), "Read the current status.")
        pdf.set_toc([
            [1, "Overview of all available operating controls", 1],
            [2, "Heating", 1],
            [2, "Water pressure", 1],
            [1, "Common status messages for all operating controls", 1],
        ])
        pdf.save(path)

    nodes = build_nodes(load_document(path, control_type="按键式"), path)
    sections = {node.text: node.metadata["section"] for node in nodes}
    assert sections["Read the display first."] == "Overview of all available operating controls"
    assert sections["1. Select heating."] == "Heating"
    assert sections["1. Read the pressure."] == "Water pressure"
    assert sections["Read the current status."] == "Common status messages for all operating controls"
