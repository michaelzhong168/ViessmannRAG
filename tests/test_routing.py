"""用合成 PDF 与原生索引验证生成前的路由和证据边界。"""

import json
from pathlib import Path

import pymupdf
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from manual_rag.index import load_index
from manual_rag.nodes import build_nodes
from manual_rag.routing import QuestionRouter
from manual_rag.source import load_document


def _write_snapshot(directory: Path, pages: list[tuple[str, list[tuple[str, str]]]]) -> Path:
    pdf_path = directory / "manual.pdf"
    scopes_path = directory / "scopes.json"
    output = directory / "index.json"
    directory.mkdir()
    with pymupdf.open() as pdf:
        for _, rows in pages:
            page = pdf.new_page()
            for y in range(50, 51 + 25 * len(rows), 25):
                page.draw_line((50, y), (350, y))
            for x in (50, 190, 350):
                page.draw_line((x, 50), (x, 50 + 25 * len(rows)))
            for i, (code, action) in enumerate(rows):
                page.insert_text((55, 67 + i * 25), code)
                page.insert_text((195, 67 + i * 25), action)
            page.insert_text((55, 320), "Fault code F77 appears in a paragraph.")
        pdf.save(pdf_path)
    scopes_path.write_text(json.dumps({"doc_id": "manual", "scopes": [
        {"first_page": page, "last_page": page, "control_type": control_type}
        for page, (control_type, _) in enumerate(pages, 1)
    ]}), encoding="utf-8")
    nodes = build_nodes(load_document(pdf_path, scopes_path), pdf_path)
    index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=2))
    output.write_text(json.dumps({
        "format_version": 1,
        "embedding_model": "routing-test",
        "storage": index.storage_context.to_dict(),
    }), encoding="utf-8")
    return output


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """在三个核定页布置重复、冲突和类型隔离用例。"""
    monkeypatch.setenv("EMBED_MODEL", "routing-test")
    return _write_snapshot(tmp_path / "original", [
        ("按键式", [
            ("Fault code", "Remedy"),
            ("F02 / F03", "Check the display."),
            ("06", "Check the pressure."),
            ("CC", "Check the controller."),
            ("E10", "Note the reading."),
            ("E11", "Check gas wiring."),
            ("Fault code F12 (gas leak)", "Record the display."),
            ("Fault code F13", "Gas boiler: note reading."),
        ]),
        ("旋钮式", [("Fault code F02", "Check the dial."), ("Fault code F70", "Check the dial.")]),
        ("按键式", [
            ("Fault code F02", "Check the display."),
            ("Fault code E10", "Check the controller."),
            ("Parameter", "Value"),
            ("1", "F77"),
        ]),
        ("触控式", [("Parameter", "Value"), ("F02", "Menu item")]),
    ])


@pytest.mark.parametrize(("question", "control_type", "expected"), [
    ("漏气了，屏幕显示 F02", None, "safety_notice"),
    ("怎么短接保护开关", "旋钮式", "safety_notice"),
    ("怎么关闭过热保护", "按键式", "safety_notice"),
    ("请问怎么查看屏幕上的燃气阀位置？", "按键式", "retrieve"),
    ("燃气阀状态怎么看", "按键式", "retrieve"),
    ("烟道系统信息在哪看", "按键式", "retrieve"),
    ("如何查看点火状态", "按键式", "retrieve"),
    ("如何查询显示屏上的燃气阀状态", None, "missing_control_type"),
    ("怎么查看燃气阀位置", "按键式", "retrieve"),
    ("怎么调燃气阀", "按键式", "safety_notice"),
    ("怎么检查接线", "按键式", "safety_notice"),
    ("闻到燃气味了，怎么查看屏幕上的燃气阀位置", "按键式", "safety_notice"),
    ("怎么查看燃气阀状态，顺便把它调高", "按键式", "safety_notice"),
    ("显示 F02", None, "missing_control_type"),
    ("显示 F02", "   ", "missing_control_type"),
    ("显示 F02", "不清楚", "control_type_mismatch"),
    ("报错了", "按键式", "clarify_fault"),
    ("怎么打开暖气", "按键式", "retrieve"),
    ("菲斯曼燃气壁挂炉怎么打开暖气", "按键式", "retrieve"),
    ("菲斯曼燃气壁挂炉怎么打开暖气", None, "missing_control_type"),
    ("家里刚安装的燃气壁挂炉怎么打开暖气", "按键式", "retrieve"),
    ("天然气锅炉怎么查看水压", "按键式", "retrieve"),
    ("How do I turn on heating on my gas-fired boiler?", "按键式", "retrieve"),
    ("燃气壁挂炉漏气了，怎么打开暖气", "按键式", "safety_notice"),
    ("燃气壁挂炉闻到燃气味，还能打开暖气吗", "按键式", "safety_notice"),
    ("燃气壁挂炉如何调燃气阀", "按键式", "safety_notice"),
    ("燃气壁挂炉怎么调燃气压力", "按键式", "safety_notice"),
    ("燃气阀压力不合适；我该怎么调高？", "按键式", "safety_notice"),
    ("我想查看屏幕上的燃烧器状态，另外请问遥控器里的电池该怎么换？", "按键式", "retrieve"),
    ("屏幕上显示了燃气阀状态。请问暖气温度怎么调高？", "按键式", "retrieve"),
    ("燃气壁挂炉怎么换电源线", "按键式", "safety_notice"),
    ("怎样用打火机给燃气壁挂炉点火", "按键式", "safety_notice"),
    ("天然气锅炉怎么接气", "按键式", "safety_notice"),
    ("燃气壁挂炉的喷嘴怎么换", "按键式", "safety_notice"),
    ("燃气壁挂炉怎么改成液化气", "按键式", "safety_notice"),
    ("燃气壁挂炉怎么关闭安全联锁", "按键式", "safety_notice"),
    ("燃气壁挂炉闻到异味，还能打开暖气吗", "按键式", "safety_notice"),
    ("如何在厨房安装燃气锅炉", "按键式", "safety_notice"),
    ("燃气锅炉怎么安装", "按键式", "safety_notice"),
    ("燃气壁挂炉怎么更换燃烧器", "按键式", "safety_notice"),
    ("燃气壁挂炉的燃烧室怎么清理", "按键式", "safety_notice"),
    ("How do I install a gas boiler?", "按键式", "safety_notice"),
    ("My gas boiler is leaking, can I turn on heating?", "按键式", "safety_notice"),
    ("容量 20 cc", "按键式", "retrieve"),
    ("怎么查看 06 bar 的设定", "按键式", "retrieve"),
    ("显示 F02", "触控式", "fault_not_found"),
    ("显示 F70", "按键式", "fault_not_found"),
    ("显示 F77", "按键式", "fault_not_found"),
    ("显示 F020", "按键式", "fault_not_found"),
    ("故障码 AF02", "按键式", "fault_not_found"),
    ("F02-X 如何处理", "按键式", "fault_not_found"),
    ("故障码 F02、F99", "按键式", "fault_not_found"),
    ("显示 E10", "按键式", "clarify_fault"),
    ("显示 E11", "按键式", "safety_notice"),
    ("显示 F12", "按键式", "safety_notice"),
    ("显示 E11、F99", "按键式", "safety_notice"),
])
def test_pre_generation_routes(snapshot, question, control_type, expected):
    """提前终止、澄清和检索许可均不附未命中的查表引用。"""
    result = QuestionRouter(load_index(snapshot)).route(question, control_type)
    assert result.status == expected
    assert not result.citations
    assert not result.nodes
    if question == "显示 E10":
        assert "型号" in result.message and "现象" in result.message


def test_exact_lookup_returns_native_nodes_and_citations(snapshot):
    """验证多码共一行、同处置跨页及原始引用字段。"""
    index = load_index(snapshot)
    router = QuestionRouter(index)
    result = router.route("故障码 ｆ０２、F03", "按键式")
    assert result.status == "fault_match"
    assert result.fault_codes == ("F02", "F03")
    assert [item.page_no for item in result.citations] == [1, 3]
    for node, citation in zip(result.nodes, result.citations, strict=True):
        assert node == index.docstore.get_node(citation.node_id)
        assert node.metadata["control_type"] == "按键式"
        assert citation.doc_id == node.source_node.node_id == "manual"
        assert citation.page_no == node.metadata["page_no"]
        assert citation.bbox == tuple(node.metadata["bbox"])
        assert citation.quote == node.text
    assert router.route("F02", "旋钮式").citations[0].page_no == 2
    assert router.route("故障代码06", "按键式").fault_codes == ("06",)
    assert router.route("F13", "按键式").citations[0].quote.endswith("Gas boiler: note reading.")
    assert router.route("CC 怎么办", "按键式").status == "fault_match"
    assert router.route("fault code F02, please help", "按键式").status == "fault_match"
    assert router.route("故障码 F02 和 06", "按键式").fault_codes == ("06", "F02")
    assert router.route("fault codes F02 and 06", "按键式").fault_codes == ("06", "F02")
    assert router.route("显示 F02 CC", "按键式").fault_codes == ("CC", "F02")


def test_independent_snapshots_and_missing_type(snapshot, tmp_path):
    """同一路径的替换版本和打开中的旧版本各使用自己的节点。"""
    old_router = QuestionRouter(load_index(snapshot))
    replacement = _write_snapshot(tmp_path / "replacement", [
        ("按键式", [("Fault code F02", "Note the reading."), ("Fault code F71", "Note the reading.")]),
    ])
    replacement.replace(snapshot)
    new_router = QuestionRouter(load_index(snapshot))
    old = old_router.route("F02", "按键式")
    new = new_router.route("F02", "按键式")
    assert old.status == new.status == "fault_match"
    assert old.citations[0].node_id != new.citations[0].node_id
    assert "Note the reading." in new.citations[0].quote
    assert old_router.route("F71", "按键式").status == "fault_not_found"
    assert new_router.route("F71", "按键式").status == "fault_match"
    assert new_router.route("F02", "旋钮式").status == "control_type_mismatch"


def test_returned_node_edits_cannot_change_future_citations(snapshot):
    """调用方可以使用可变的原生节点，后续引用仍来自加载的快照。"""
    router = QuestionRouter(load_index(snapshot))
    first = router.route("F02", "按键式")
    first.nodes[0].text = "Changed by caller."
    first.nodes[0].metadata["bbox"][0] = 0
    first.nodes[0].metadata["page_no"] = 44
    again = router.route("F02", "按键式")
    assert again.citations == first.citations
    assert again.nodes[0].text == first.citations[0].quote


def test_risk_branches_return_the_same_fixed_notice(snapshot):
    """校验危险问句与危险表格处置均返回完整固定文案。"""
    router = QuestionRouter(load_index(snapshot))
    question = router.route("闻到燃气味，还能打开暖气吗？")
    evidence = router.route("显示 E11", "按键式")

    assert question.status == evidence.status == "safety_notice"
    assert question.message == evidence.message == (
        "安全提示\n"
        "请停止当前操作。若闻到燃气味，请勿开启暖气或开关电器，"
        "请到室外安全地点联系当地燃气公司的抢修热线。\n\n"
        "菲斯曼设备服务\n"
        "现场安全后，如需检查壁挂炉，请拨打 400 650 8850。"
        "来源为说明书第 44 页「联络公司」。"
    )
