"""在问答生成前判定风险、核对操作类型并精确查询故障表格。"""

import unicodedata
from dataclasses import dataclass
from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from manual_rag.fault_codes import codes_in_question, codes_in_table_row
from manual_rag.risk import requires_safety_notice
from manual_rag.source import CONTROL_TYPES

RouteStatus = Literal[
    "retrieve", "fault_match", "safety_notice", "missing_control_type",
    "control_type_mismatch", "clarify_fault", "fault_not_found",
]

_FAULT_SYMPTOMS = ("故障", "报错", "报警", "不工作", "不热", "不启动", "没热水")
_SAFETY_NOTICE_MESSAGE = (
    "安全提示\n"
    "请停止当前操作。若闻到燃气味，请勿开启暖气或开关电器，"
    "请到室外安全地点联系当地燃气公司的抢修热线。\n\n"
    "菲斯曼设备服务\n"
    "现场安全后，如需检查壁挂炉，请拨打 400 650 8850。"
)


@dataclass(frozen=True)
class Citation:
    """取自实际命中节点的来源、物理页、页内坐标和逐字原文。"""

    node_id: str
    doc_id: str
    page_no: int  # 从 1 开始的 PDF 物理页。
    bbox: tuple[float, float, float, float]  # PyMuPDF 页坐标，单位 pt。
    quote: str


@dataclass(frozen=True)
class RouteResult:
    """生成前的决定；仅 fault_match 携带证据节点与一一对应的引用。"""

    status: RouteStatus
    message: str = ""
    fault_codes: tuple[str, ...] = ()
    nodes: tuple[TextNode, ...] = ()
    citations: tuple[Citation, ...] = ()


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


class QuestionRouter:
    """从已加载的一个索引建立路由；重载快照后应创建新的路由实例。"""

    def __init__(self, index: VectorStoreIndex) -> None:
        self._table: dict[str, dict[str, list[TextNode]]] = {}
        for node in index.docstore.get_nodes(list(index.index_struct.nodes_dict.values())):
            control_type = node.metadata.get("control_type")
            if not isinstance(node, TextNode) or control_type not in CONTROL_TYPES:
                continue
            entries = self._table.setdefault(control_type, {})
            if node.metadata.get("kind") != "table_row":
                continue
            first_cell, separator, remedy = node.text.partition("\t")
            if separator and remedy.strip() and node.source_node is not None:
                context = node.metadata.get("table_header") or node.metadata.get("section", "")
                for code in dict.fromkeys(codes_in_table_row(_normalize(first_cell), _normalize(context))):
                    entries.setdefault(code, []).append(node)
        self._known_codes = {code for entries in self._table.values() for code in entries}

    def route(self, question: str, control_type: str | None = None) -> RouteResult:
        """先处理风险和操作类型，再返回查表结果、澄清提示或检索许可。"""
        query = _normalize(question).strip()
        if not query:
            raise ValueError("问句不能为空")
        if requires_safety_notice(query):
            return RouteResult("safety_notice", _SAFETY_NOTICE_MESSAGE)
        if control_type is None or not control_type.strip():
            return RouteResult("missing_control_type", "请确认设备是旋钮式、按键式还是触控式。")
        entries = self._table.get(control_type.strip())
        if entries is None:
            return RouteResult("control_type_mismatch", "当前说明书不适用于该操作类型，请核对设备类型。")

        codes = codes_in_question(query, self._known_codes)
        if not codes:
            if any(symptom in query for symptom in _FAULT_SYMPTOMS):
                return RouteResult("clarify_fault", "请描述具体现象，并提供屏幕上的故障码（如有）。")
            return RouteResult("retrieve")

        matches = [entries.get(code, []) for code in codes]
        if any(requires_safety_notice(_normalize(node.text)) for rows in matches for node in rows):
            return RouteResult("safety_notice", _SAFETY_NOTICE_MESSAGE, codes)
        if not all(matches):
            return RouteResult("fault_not_found", "当前操作类型的说明书缺少对应故障码，请核对报码。", codes)
        # 表格中不同的处置文字可能对应同类设备内的适用差异，交由用户澄清。
        for rows in matches:
            remedies = {
                tuple(" ".join(_normalize(cell).split()) for cell in node.text.split("\t")[1:])
                for node in rows
            }
            if len(remedies) > 1:
                return RouteResult("clarify_fault", "请提供设备型号或具体现象，以确认适用的故障条目。", codes)

        nodes = {node.node_id: node for rows in matches for node in rows}
        citations = tuple(
            Citation(node.node_id, node.source_node.node_id, node.metadata["page_no"],
                     tuple(node.metadata["bbox"]), node.text)
            for node in nodes.values()
        )
        # TextNode 可变，交给调用方的节点需隔离于下一次路由。
        evidence = tuple(node.model_copy(deep=True) for node in nodes.values())
        return RouteResult("fault_match", fault_codes=codes, nodes=evidence, citations=citations)
