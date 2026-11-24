"""识别说明书表格和用户问句中的故障码格式。"""

import re

_CODE_LABEL = r"(?:故障\s*(?:代\s*码|码)|错误\s*(?:代码|码)|报码|fault\s+codes?)"
_CODE_TOKEN = r"[a-z0-9]+(?:[.-][a-z0-9]+)*"
_CODE_SEPARATOR = r"\s*(?:[、,/;]|或(?:者)?|以及|和|及|\b(?:and|or)\b)\s*"
_CODE_LIST = rf"{_CODE_TOKEN}(?:{_CODE_SEPARATOR}{_CODE_TOKEN})*"
_LABELED_CODES = re.compile(rf"{_CODE_LABEL}\s*[:：]?\s*(?:为|是)?\s*({_CODE_LIST})")
_LABEL = re.compile(_CODE_LABEL)
_BARE_CODE_LIST = re.compile(_CODE_LIST)
_TOKEN = re.compile(_CODE_TOKEN)
_STANDARD_CODE = re.compile(r"[fe][0-9]+(?:[.-][a-z0-9]+)*")
_DISPLAY_CUE = re.compile(r"(?:显示|报|闪烁|跳出|提示|代码|code)\s*[:：]?\s*$")
_QUANTITY_BEFORE = re.compile(r"(?<![a-z0-9.])(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*$")


def _split_codes(code_list: str) -> list[str]:
    return [code.upper() for code in re.split(_CODE_SEPARATOR, code_list)]


def codes_in_table_row(first_cell: str, context: str) -> list[str]:
    """从已规范化的故障表首列取出报码，表头或章节可提供故障码标签。"""
    cell = first_cell.strip()
    labeled = _LABELED_CODES.search(cell)
    if labeled:
        return _split_codes(labeled[1])
    if _LABEL.search(context) and _BARE_CODE_LIST.fullmatch(cell):
        return _split_codes(cell)
    return []


def codes_in_question(question: str, known_codes: set[str]) -> tuple[str, ...]:
    """识别规范化问句中的明确报码，返回排序后的唯一大写代码。"""
    codes = set()
    for labeled in _LABELED_CODES.finditer(question):
        first, *following = _split_codes(labeled[1])
        codes.add(first)
        # 英文逗号或连词后也可能是普通文字，只收录已知码或含数字的内容。
        codes.update(code for code in following if code in known_codes or any(c.isdigit() for c in code))
    for token in _TOKEN.finditer(question):
        value = token[0]
        code = value.upper()
        if _STANDARD_CODE.fullmatch(value):
            codes.add(code)
            continue
        if code not in known_codes:
            continue

        # 纯数字可能是读数，纯字母紧跟数字时可能是单位。
        mixed_code = not value.isdigit() and not value.isalpha()
        standalone_letters = value.isalpha() and not _QUANTITY_BEFORE.search(question, 0, token.start())
        display_context = (
            _DISPLAY_CUE.search(question[:token.start()])
            or question[token.end():].lstrip().startswith("闪烁")
        )
        if mixed_code or standalone_letters or question == value or display_context:
            codes.add(code)
    return tuple(sorted(codes))
