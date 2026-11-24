"""从表格和问句中读取故障码，匹配规则使用规范化后的文本。"""

import re
from collections.abc import Iterator

# 标签片段之间允许有空白，适配 PDF 的折行。
_LABEL_PARTS = (
    ("故障", "代", "码"), ("故障", "码"), ("错误", "代码"), ("错误", "码"), ("报码",),
)
_LABEL = re.compile("|".join(
    [r"\s*".join(map(re.escape, parts)) for parts in _LABEL_PARTS] + [r"fault\s+codes?"]
))
_CODE_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_LIST_SEPARATOR = re.compile(r"[、,/;]|或者|或|以及|和|及|\b(?:and|or)\b")
_DISPLAY_CUES = ("显示", "报", "闪烁", "跳出", "提示", "代码", "code")
_NUMBER_CHARS = "0123456789."


def _skip_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    return offset


def _read_code_list(text: str, offset: int) -> tuple[list[str], int]:
    """从指定位置读取第一个码和后续用分隔符连接的码，返回停止位置。"""
    codes = []
    end = offset
    while token := _CODE_TOKEN.match(text, offset):
        codes.append(token[0].upper())
        end = token.end()
        separator = _LIST_SEPARATOR.match(text, _skip_space(text, end))
        if separator is None:
            break
        offset = _skip_space(text, separator.end())
    # 有标点但后面没有报码时，停止位置仍在最后一个码之后。
    return codes, end


def _labeled_code_lists(text: str) -> Iterator[list[str]]:
    offset = 0
    while label := _LABEL.search(text, offset):
        start = _skip_space(text, label.end())
        if text[start:start + 1] in (":", "："):
            start = _skip_space(text, start + 1)
        if text[start:start + 1] in ("为", "是"):
            start = _skip_space(text, start + 1)
        codes, end = _read_code_list(text, start)
        if codes:
            yield codes
        offset = end if codes else label.end()


def codes_in_table_row(first_cell: str, context: str) -> list[str]:
    """读取表格首列；省略标签时，表头或章节须说明它是故障码列。"""
    cell = first_cell.strip()
    labeled_codes = next(_labeled_code_lists(cell), None)
    if labeled_codes is not None:
        return labeled_codes
    if _LABEL.search(context):
        codes, end = _read_code_list(cell, 0)
        if end == len(cell):
            return codes
    return []


def _preceded_by_quantity(prefix: str) -> bool:
    """识别紧邻的整数或小数，避免把 20 cc 里的单位认成报码。"""
    end = len(prefix)
    start = end
    while start and prefix[start - 1] in _NUMBER_CHARS:
        start -= 1
    if start and prefix[start - 1].isascii() and prefix[start - 1].isalnum():
        return False
    value = prefix[start:end]
    whole, _, fraction = value.partition(".")
    return value.isdigit() or (fraction.isdigit() and (not whole or whole.isdigit()))


def _known_token_is_code(question: str, token: re.Match[str]) -> bool:
    """判断已知短码在此处是否表示报码。"""
    value = token[0]
    if question == value or (not value.isdigit() and not value.isalpha()):
        return True
    before = question[:token.start()].rstrip()
    if value.isalpha() and not _preceded_by_quantity(before):
        return True

    if before.endswith((":", "：")):
        before = before[:-1].rstrip()
    after = question[token.end():].lstrip()
    return before.endswith(_DISPLAY_CUES) or after.startswith("闪烁")


def codes_in_question(question: str, known_codes: set[str]) -> tuple[str, ...]:
    """读明确报码与可确认的短码，返回排序后的唯一大写代码。"""
    codes = set()
    for first, *following in _labeled_code_lists(question):
        codes.add(first)
        # 逗号或连词后也可能是普通文字，只保留已知码或含数字的内容。
        codes.update(code for code in following if code in known_codes or any(c.isdigit() for c in code))

    for token in _CODE_TOKEN.finditer(question):
        value = token[0]
        head = value.split(".", 1)[0].split("-", 1)[0]
        code = value.upper()
        # F02、E10 及其带后缀形式即使未入库，也需要走精确查表。
        standard_code = head.startswith(("f", "e")) and head[1:].isdigit()
        if standard_code or (code in known_codes and _known_token_is_code(question, token)):
            codes.add(code)
    return tuple(sorted(codes))
