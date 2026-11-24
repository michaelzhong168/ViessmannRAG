"""提取 PDF 版面，统一使用页图方向的 pt 坐标。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

UNSECTIONED = "(未分章)"
HEADING_SIZE_RATIO = 1.15
HEADING_MAX_CHARS = 60
HEADING_MAX_LINES = 2
BODY_SIZE_MIN_SHARE = 0.25
BODY_TEXT_MARKER = re.compile(r"^(?:[■●▪•]|[0-9]+(?:[、)]|[.．](?!\d)))|[。！？；]|[.!?]\s")


@dataclass
class Heading:
    text: str  # 标题原文
    level: int  # 标题层级，1 为最高级，最多 3 级
    page_no: int  # PDF 物理页码，从 1 开始
    bbox: list[float]  # 页图方向的标题框 [x0, y0, x1, y1]，左上为原点，单位 pt
    font_size: float  # 提取的原始字号，分级时使用 round(font_size, 1)


@dataclass
class Table:
    page_no: int  # PDF 物理页码，从 1 开始
    bbox: list[float]  # 页图方向的整表框 [x0, y0, x1, y1]，左上为原点，单位 pt
    header: list[str]  # 表格首行的单元格，按列顺序保存
    rows: list[list[str]]  # 表头下的数据行，各行按表头列顺序保存，空单元格为 ""
    row_bboxes: list[list[float]]  # 与 rows 按索引一一对应，坐标格式同 bbox
    chapter: str  # 最近的前置标题，无前置标题时为 "(未分章)"
    header_text: str  # 由 " | ".join(header) 得到的表头文本，随 JSON 保存


@dataclass
class Block:
    page_no: int  # PDF 物理页码，从 1 开始
    bbox: list[float]  # 页图方向的正文块框 [x0, y0, x1, y1]，左上为原点，单位 pt
    text: str  # 正文原文，保留行界
    font_size: float  # 块中有文本的片段使用的最大字号
    chapter: str  # 最近的前置标题，无前置标题时为 "(未分章)"


@dataclass
class Layout:
    headings: list[Heading]  # 标题列表，按页码、纵坐标、横坐标、文本排序
    tables: list[Table]  # 表格列表，按页码、纵坐标、横坐标、表头文本排序
    body_blocks: list[Block]  # 正文块列表，按页码、纵坐标、横坐标、文本排序
    header_footer_removed: int  # 按页眉页脚规则剔除的文本块数量


def _page_bbox(
    bbox: pymupdf.Rect, matrix: pymupdf.Matrix, bounds: pymupdf.Rect
) -> list[float]:
    return list((pymupdf.Rect(bbox) * matrix) & bounds)


def _extract_tables(page: pymupdf.Page) -> list[Table]:
    tables = []
    # 在未旋转的页面上识别表格，再用原旋转矩阵转换表格框和行框。
    matrix, bounds = page.rotation_matrix, page.rect
    rotation = page.rotation
    page.set_rotation(0)
    try:
        for grid in page.find_tables(strategy="lines").tables:
            extracted = grid.extract()
            if not extracted:
                continue
            header, *rows = [
                [cell if cell is not None else "" for cell in row]
                for row in extracted
            ]
            tables.append(
                Table(
                    page_no=page.number + 1,
                    bbox=_page_bbox(grid.bbox, matrix, bounds),
                    header=header,
                    rows=rows,
                    row_bboxes=[
                        _page_bbox(row.bbox, matrix, bounds) for row in grid.rows[1:]
                    ],
                    chapter=UNSECTIONED,
                    header_text=" | ".join(header),
                )
            )
    finally:
        page.set_rotation(rotation)
    return tables


def _layout_order(item: Heading | Block | Table) -> tuple[int, float, float, str]:
    text = item.header_text if isinstance(item, Table) else item.text
    return item.page_no, item.bbox[1], item.bbox[0], text


def _extract_blocks(
    page: pymupdf.Page, tables: list[Table], top_ratio: float, bottom_ratio: float
) -> tuple[list[Block], int]:
    matrix, bounds = page.rotation_matrix, page.rect
    top = bounds.height * top_ratio
    bottom = bounds.height * (1 - bottom_ratio)
    table_boxes = [pymupdf.Rect(table.bbox) for table in tables]
    blocks: list[Block] = []
    removed = 0
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        lines = block["lines"]
        spans = [span for line in lines for span in line["spans"]]
        if not spans:
            continue
        bbox = _page_bbox(block["bbox"], matrix, bounds)
        rect = pymupdf.Rect(bbox)
        if rect.is_empty:
            continue
        center = (rect.tl + rect.br) * 0.5
        if not top <= center.y <= bottom:
            removed += 1
            continue
        # 表格文字已随单元格提取，避免重复进入正文。
        if any(center in table_box for table_box in table_boxes):
            continue
        text = "\n".join(
            "".join(span["text"] for span in line["spans"]) for line in lines
        )
        if not text.strip():
            continue
        blocks.append(
            Block(
                page_no=page.number + 1,
                bbox=bbox,
                text=text,
                font_size=max(span["size"] for span in spans if span["text"]),
                chapter=UNSECTIONED,
            )
        )
    return blocks, removed


def _heading_text_key(text: str) -> str:
    return "".join(text.split())


def _heading_candidates(blocks: list[Block], bookmarks: set[tuple[int, str]]) -> set[int]:
    confirmed = {
        id(block) for block in blocks
        if (block.page_no, _heading_text_key(block.text)) in bookmarks
    }
    body_weights: Counter[float] = Counter()
    for block in blocks:
        if id(block) not in confirmed:
            body_weights[round(block.font_size, 1)] += len(_heading_text_key(block.text))
    if not body_weights:
        return confirmed
    # 取有足够文字的较小字号，避免零散小字或短文档里的大标题主导基准。
    weight_floor = max(body_weights.values()) * BODY_SIZE_MIN_SHARE
    body_size = min(size for size, weight in body_weights.items() if weight >= weight_floor)

    # 同页书签能确定标题，目录没列出的块按字号和文本形态继续判断。
    for block in blocks:
        text = block.text.strip()
        if (
            id(block) not in confirmed
            and round(block.font_size, 1) >= body_size * HEADING_SIZE_RATIO
            and len(text) <= HEADING_MAX_CHARS
            and len(text.splitlines()) <= HEADING_MAX_LINES
            and not BODY_TEXT_MARKER.search(text)
            and not text.endswith((".", "!", "?", ";", ":", "："))
        ):
            confirmed.add(id(block))
    return confirmed


def _assign_chapters(
    blocks: list[Block], tables: list[Table], removed: int, bookmarks: set[tuple[int, str]]
) -> Layout:
    candidates = _heading_candidates(blocks, bookmarks)
    sizes = sorted(
        {round(block.font_size, 1) for block in blocks if id(block) in candidates},
        reverse=True,
    )
    levels = {size: min(level, 3) for level, size in enumerate(sizes, start=1)}
    layout = Layout([], [], [], removed)
    chapter = UNSECTIONED
    for item in sorted([*blocks, *tables], key=_layout_order):
        if isinstance(item, Table):
            item.chapter = chapter
            layout.tables.append(item)
            continue
        if id(item) not in candidates:
            item.chapter = chapter
            layout.body_blocks.append(item)
        else:
            chapter = item.text
            level = levels[round(item.font_size, 1)]
            layout.headings.append(
                Heading(item.text, level, item.page_no, item.bbox, item.font_size)
            )
    return layout


def analyze_layout(
    pdf: pymupdf.Document, first_page: int = 1, last_page: int | None = None,
) -> Layout:
    """提取同一核定页段的版面，保持物理页码和页图方向的坐标。"""
    last_page = len(pdf) if last_page is None else last_page
    if not 1 <= first_page <= last_page <= len(pdf):
        raise ValueError("页段须位于 PDF 物理页范围内")
    blocks: list[Block] = []
    tables: list[Table] = []
    removed = 0
    bookmarks = {
        (page_no, _heading_text_key(title)) for _, title, page_no in pdf.get_toc()
        if first_page <= page_no <= last_page
    }
    for number in range(first_page - 1, last_page):
        page = pdf[number]
        page_tables = _extract_tables(page)
        page_blocks, page_removed = _extract_blocks(page, page_tables, 0.08, 0.08)
        tables.extend(page_tables)
        blocks.extend(page_blocks)
        removed += page_removed
    return _assign_chapters(blocks, tables, removed, bookmarks)
