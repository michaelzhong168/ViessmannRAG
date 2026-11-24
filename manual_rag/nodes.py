"""从已核定的说明书 Document 和对应 PDF 构建原生 TextNode。"""

import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf
from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode

from manual_rag.source import CONTROL_TYPES

LAYOUT_VERSION = 2
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_CHARS = 60
_SECTION_INDENT_PT = 18  # 同列标题和正文可以有约两个汉字的缩进差。
_STEP_START = re.compile(r"^\s*(?:[0-9]{1,3}[.．、)）](?![0-9])|[（(][0-9]{1,3}[）)]|[①-⑳])")

_CellPosition = tuple[int, int, int]  # 表格、行、列的索引，均从 0 开始。
_Table = list[list[pymupdf.Rect | None]]  # 行中的 None 表示缺失或被合并的单元格。


@dataclass
class _Line:
    text: str
    bbox: pymupdf.Rect  # 未旋转的页坐标，左上为原点，单位 pt。
    size: float


@dataclass
class _Fragment:
    text: str
    bbox: pymupdf.Rect  # 和 _Line 使用同一页坐标系。
    kind: str
    table_header: str = ""  # 同一线框表格首行的第一列。


def _approved_pages(document: Document, page_count: int) -> dict[int, str]:
    pages = {}
    for scope in document.metadata["pages"]:
        page_no, control_type = scope["page_no"], scope["control_type"]
        if type(page_no) is not int or not 1 <= page_no <= page_count:
            raise ValueError(f"核定物理页须位于 1–{page_count}")
        if control_type not in CONTROL_TYPES or page_no in pages:
            raise ValueError(f"物理页 {page_no} 的操作类型无效或核定记录重复")
        pages[page_no] = control_type
    return pages


def _line(glyphs: list[tuple[str, pymupdf.Rect, float]]) -> _Line:
    bbox = pymupdf.Rect(glyphs[0][1])
    for _, rect, _ in glyphs:
        bbox |= rect
    return _Line("".join(char for char, _, _ in glyphs), bbox, max(size for _, _, size in glyphs))


def _tables(page: pymupdf.Page) -> list[_Table]:
    # 字形和行框统一使用未旋转坐标。识别完成后恢复页的显示方向。
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    try:
        return [
            [[pymupdf.Rect(cell) if cell is not None else None for cell in row.cells]
             for row in table.rows]
            for table in page.find_tables().tables
            if table.row_count >= 2 and table.col_count >= 2
        ]
    finally:
        if rotation:
            page.set_rotation(rotation)


def _page_lines(
    page: pymupdf.Page, tables: list[_Table],
) -> tuple[list[list[_Line]], dict[_CellPosition, list[_Line]]]:
    blocks = []
    cell_lines: dict[_CellPosition, list[_Line]] = {}
    cell_boxes = [
        ((table_no, row_no, col_no), cell)
        for table_no, table in enumerate(tables)
        for row_no, row in enumerate(table)
        for col_no, cell in enumerate(row)
        if cell is not None
    ]
    for block in page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        lines = []
        for line in block["lines"]:
            # 逐字归入线框单元格。跨列的 PDF 文本行仍按实际列边界切开。
            runs = []
            for span in line["spans"]:
                for char in span["chars"]:
                    rect = pymupdf.Rect(char["bbox"])
                    center = (rect.tl + rect.br) / 2
                    owner = next((key for key, cell in cell_boxes if center in cell), None)
                    if not runs or runs[-1][0] != owner:
                        runs.append((owner, []))
                    runs[-1][1].append((char["c"], rect, span["size"]))
            for owner, glyphs in runs:
                part = _line(glyphs)
                if not part.text.strip():
                    continue
                if owner is None:
                    lines.append(part)
                else:
                    cell_lines.setdefault(owner, []).append(part)
        if lines:
            blocks.append(lines)
    return blocks, cell_lines


def _fragment(lines: list[_Line], kind: str) -> _Fragment:
    bbox = pymupdf.Rect(lines[0].bbox)
    for line in lines:
        bbox |= line.bbox
    return _Fragment("\n".join(line.text for line in lines), bbox, kind)


def _heading_key(text: str) -> str:
    return "".join(text.split())


def _text_fragments(blocks: list[list[_Line]], bookmarks: set[str]) -> list[_Fragment]:
    sizes: Counter[float] = Counter()
    for block in blocks:
        for line in block:
            sizes[round(line.size, 1)] += len(line.text.strip())
    body_size = sizes.most_common(1)[0][0] if sizes else 0
    fragments = []
    for block in blocks:
        pending = []
        kind = "paragraph"
        for line in block:
            text = line.text.strip()
            is_step = _STEP_START.match(text)
            is_heading = _heading_key(text) in bookmarks or (
                not is_step
                and round(line.size, 1) >= body_size * _HEADING_SIZE_RATIO
                and len(text) <= _HEADING_MAX_CHARS
                and not text.endswith((".", "。", "!", "！", "?", "？", ";", "；"))
            )
            if is_heading or is_step:
                if pending:
                    fragments.append(_fragment(pending, kind))
                    pending = []
                kind = "paragraph"
            if is_heading:
                fragments.append(_fragment([line], "heading"))
            else:
                if is_step:
                    kind = "step"
                pending.append(line)
        if pending:
            fragments.append(_fragment(pending, kind))
    return fragments


def _table_fragments(
    tables: list[_Table], cell_lines: dict[_CellPosition, list[_Line]],
) -> list[_Fragment]:
    fragments = []
    for table_no, table in enumerate(tables):
        header = "\n".join(line.text for line in cell_lines.get((table_no, 0, 0), []))
        for row_no, row in enumerate(table):
            cells = []
            for col_no in range(len(row)):
                lines = cell_lines.get((table_no, row_no, col_no))
                cells.append(_fragment(lines, "table_row") if lines else None)
            populated = [cell for cell in cells if cell is not None]
            if populated:
                bbox = pymupdf.Rect(populated[0].bbox)
                for cell in populated:
                    bbox |= cell.bbox
                text = "\t".join(cell.text if cell is not None else "" for cell in cells)
                fragments.append(_Fragment(text, bbox, "table_row", header))
    return fragments


def _sections(fragments: list[_Fragment]) -> Iterator[tuple[_Fragment, str]]:
    headings: list[_Fragment] = []
    for fragment in sorted(fragments, key=lambda item: (item.bbox.y0, item.bbox.x0)):
        left = fragment.bbox.x0
        if fragment.kind == "heading":
            # 新标题只替换同列标题，通栏标题可以同时覆盖多列。
            right = max(left + _SECTION_INDENT_PT, fragment.bbox.x1)
            headings = [
                heading for heading in headings
                if not left - _SECTION_INDENT_PT <= heading.bbox.x0 <= right
            ]
            headings.append(fragment)
            yield fragment, fragment.text
            continue
        # 前置标题中，使用最靠近正文左缘的列。右侧栏的标题不会覆盖左侧正文。
        heading = max(
            (heading for heading in headings if heading.bbox.x0 <= left + _SECTION_INDENT_PT),
            key=lambda heading: heading.bbox.x0,
            default=None,
        )
        yield fragment, heading.text if heading is not None else ""


def build_nodes(document: Document, pdf_path: str | Path) -> list[TextNode]:
    """逐核定页生成 Node，坐标为未旋转的 PyMuPDF 页坐标，单位 pt。

    页内标题提供章节上下文。表格行使用制表符标出列，保留单元格原字符。
    PDF 与 Document 的来源、版本或核定信息不符时抛出 ValueError。
    """
    source = Path(pdf_path)
    contents = source.read_bytes()
    version = sha256(contents).hexdigest()
    if source.name != document.metadata["source"] or version != document.metadata["pdf_version"]:
        raise ValueError("PDF 来源或文件指纹与 Document 不符")

    provenance = {"source": source.name, "pdf_version": version}
    source_info = RelatedNodeInfo(
        node_id=document.id_, node_type=document.get_type(), metadata=provenance,
    )
    nodes = []
    with pymupdf.open(stream=contents, filetype="pdf") as pdf:
        approved = _approved_pages(document, len(pdf))
        bookmarks: dict[int, set[str]] = {}
        for _, title, page_no in pdf.get_toc():
            if page_no in approved:
                bookmarks.setdefault(page_no, set()).add(_heading_key(title))

        for page_no, control_type in sorted(approved.items()):
            page = pdf[page_no - 1]
            tables = _tables(page)
            blocks, cell_lines = _page_lines(page, tables)
            fragments = _text_fragments(blocks, bookmarks.get(page_no, set()))
            fragments.extend(_table_fragments(tables, cell_lines))
            # 仅使用当前物理页的标题，避免跨页传播章节和操作类型。
            for ordinal, (fragment, section) in enumerate(_sections(fragments)):
                metadata = {
                    **provenance,
                    "page_no": page_no,
                    "bbox": list(fragment.bbox),
                    "section": section,
                    "control_type": control_type,
                    "kind": fragment.kind,
                }
                if fragment.table_header:
                    metadata["table_header"] = fragment.table_header
                identity = json.dumps(
                    [LAYOUT_VERSION, document.id_, ordinal, metadata, fragment.text],
                    sort_keys=True, ensure_ascii=False,
                )
                node_id = sha256(identity.encode("utf-8")).hexdigest()
                excluded = [key for key in metadata if key != "section" or not section]
                nodes.append(TextNode(
                    id_=node_id, text=fragment.text, metadata=metadata,
                    relationships={NodeRelationship.SOURCE: source_info},
                    excluded_embed_metadata_keys=excluded, excluded_llm_metadata_keys=excluded,
                ))
    return nodes
