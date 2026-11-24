"""在问答生成前判定风险、核对操作类型并精确查询故障表格。"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from manual_rag.source import CONTROL_TYPES

RouteStatus = Literal[
    "retrieve", "fault_match", "human_review", "missing_control_type",
    "control_type_mismatch", "clarify_fault", "fault_not_found",
]

_HAZARD = re.compile(
    r"燃气|煤气|天然气|漏气|气体泄漏|焦糊|冒烟|起火|一氧化碳|触电|漏电|"
    r"拆机|拆卸|拆开|拆除|打开外壳|点火|排烟|烟道|烟管|接线|布线|电气连接|电源线|导线|线束|端子|"
    r"短接|旁路|(?:关闭|禁用|取消|停用|屏蔽|绕过).{0,8}保护|"
    r"\b(?:gas|disassemble|dismantle|ignition|ignite|flue|wiring|wire)\b"
)
_DISPLAY = r"(?:屏幕|显示屏|界面|菜单)(?:[上里中]?的?)"
_STATUS_ITEM = (
    rf"(?:{_DISPLAY})?(?:燃气阀(?:开关)?状态|点火状态|烟道(?:系统)?(?:信息|状态))"
    rf"|{_DISPLAY}燃气阀位置"
)
_READ_ONLY_STATUS = re.compile(
    rf"(?:请问\s*)?(?:(?:怎么|如何|在哪(?:里)?)\s*(?:查看|查询|看)\s*(?:{_STATUS_ITEM})"
    rf"|(?:{_STATUS_ITEM})\s*(?:怎么|如何|在哪(?:里)?)\s*(?:查看|查询|看))[?。.!！]*"
)
_CODE_LABEL = r"(?:故障\s*(?:代\s*码|码)|错误\s*(?:代码|码)|报码|fault\s+codes?)"
_CODE = r"[a-z0-9]+(?:[.-][a-z0-9]+)*"
_CODE_SEPARATOR = r"\s*(?:[、,/;]|或(?:者)?|以及|和|及|\b(?:and|or)\b)\s*"
_CODE_LIST = rf"{_CODE}(?:{_CODE_SEPARATOR}{_CODE})*"
_LABELED_CODES = re.compile(rf"{_CODE_LABEL}\s*[:：]?\s*(?:为|是)?\s*({_CODE_LIST})")
_BARE_CODE_LIST = re.compile(_CODE_LIST)
_TOKEN = re.compile(_CODE)
_STANDARD_CODE = re.compile(r"[fe][0-9]+(?:[.-][a-z0-9]+)*")
_DISPLAY_CUE = re.compile(r"(?:显示|报|闪烁|跳出|提示|代码|code)\s*[:：]?\s*$")
_QUANTITY_BEFORE = re.compile(r"(?<![a-z0-9.])(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*$")
_SYMPTOM = re.compile(r"故障|报错|报警|不工作|不热|不启动|没热水")
_HUMAN_REVIEW_MESSAGE = "请联系菲斯曼售后或具备资质的专业人员处理。"


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


def _split_codes(code_list: str) -> list[str]:
    return [code.upper() for code in re.split(_CODE_SEPARATOR, code_list)]


def _row_codes(first_cell: str, context: str) -> list[str]:
    cell = _normalize(first_cell).strip()
    labeled = _LABELED_CODES.search(cell)
    if labeled:
        return _split_codes(labeled[1])
    if re.search(_CODE_LABEL, _normalize(context)) and _BARE_CODE_LIST.fullmatch(cell):
        return _split_codes(cell)
    return []


def _question_codes(question: str, known_codes: set[str]) -> tuple[str, ...]:
    codes = set()
    for labeled in _LABELED_CODES.finditer(question):
        first, *following = _split_codes(labeled[1])
        codes.add(first)
        codes.update(code for code in following if code in known_codes or any(c.isdigit() for c in code))
    for token in _TOKEN.finditer(question):
        value = token[0]
        code = value.upper()
        if _STANDARD_CODE.fullmatch(value):
            codes.add(code)
        elif code in known_codes:
            # 纯数字可能是读数，纯字母紧跟数字时可能是单位。
            if (
                (not value.isdigit() and not value.isalpha())
                or (value.isalpha() and not _QUANTITY_BEFORE.search(question, 0, token.start()))
                or question == value
                or _DISPLAY_CUE.search(question[:token.start()])
                or question[token.end():].lstrip().startswith("闪烁")
            ):
                codes.add(code)
    return tuple(sorted(codes))


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
                for code in dict.fromkeys(_row_codes(first_cell, context)):
                    entries.setdefault(code, []).append(node)
        self._known_codes = {code for entries in self._table.values() for code in entries}

    def route(self, question: str, control_type: str | None = None) -> RouteResult:
        """先处理风险和操作类型，再返回查表结果、澄清提示或检索许可。"""
        query = _normalize(question).strip()
        if not query:
            raise ValueError("问句不能为空")
        # 只放行完整的界面查询句式，避免混合请求用一句状态查询绕过转人工。
        if _HAZARD.search(query) and not _READ_ONLY_STATUS.fullmatch(query):
            return RouteResult("human_review", _HUMAN_REVIEW_MESSAGE)
        if control_type is None or not control_type.strip():
            return RouteResult("missing_control_type", "请确认设备是旋钮式、按键式还是触控式。")
        entries = self._table.get(control_type.strip())
        if entries is None:
            return RouteResult("control_type_mismatch", "当前说明书不适用于该操作类型，请核对设备类型。")

        codes = _question_codes(query, self._known_codes)
        if not codes:
            if _SYMPTOM.search(query):
                return RouteResult("clarify_fault", "请描述具体现象，并提供屏幕上的故障码（如有）。")
            return RouteResult("retrieve")

        matches = [entries.get(code, []) for code in codes]
        if any(_HAZARD.search(_normalize(node.text)) for rows in matches for node in rows):
            return RouteResult("human_review", _HUMAN_REVIEW_MESSAGE, codes)
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
