"""提取 PDF 版面，统一使用页图方向的 pt 坐标。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from pipeline.parse import (
    Document,
    _OutputPathConflict,
    _parse_pdf,
    _require_distinct_files,
    _require_regular_output,
    write_audit,
)

UNSECTIONED = "(未分章)"
DEFAULT_LAYOUT_PATH = "data/layout.json"


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


def _assign_chapters(
    blocks: list[Block], tables: list[Table], removed: int
) -> Layout:
    sizes = sorted({round(block.font_size, 1) for block in blocks}, reverse=True)
    # 以全文最小字号作为正文基准，其余字号按降序分级。
    levels = {size: min(level, 3) for level, size in enumerate(sizes[:-1], start=1)}
    layout = Layout([], [], [], removed)
    chapter = UNSECTIONED
    for item in sorted([*blocks, *tables], key=_layout_order):
        if isinstance(item, Table):
            item.chapter = chapter
            layout.tables.append(item)
            continue
        level = levels.get(round(item.font_size, 1))
        if level is None:
            item.chapter = chapter
            layout.body_blocks.append(item)
        else:
            chapter = item.text
            layout.headings.append(
                Heading(item.text, level, item.page_no, item.bbox, item.font_size)
            )
    return layout


def analyze_layout(
    doc: Document, top_ratio: float = 0.08, bottom_ratio: float = 0.08
) -> Layout:
    """逐页提取正文和线框表格，再按全篇字号和阅读顺序划分章节。"""
    blocks: list[Block] = []
    tables: list[Table] = []
    removed = 0
    with pymupdf.open(doc.pdf_path) as pdf:
        for page in pdf:
            page_tables = _extract_tables(page)
            page_blocks, page_removed = _extract_blocks(
                page, page_tables, top_ratio, bottom_ratio
            )
            blocks.extend(page_blocks)
            tables.extend(page_tables)
            removed += page_removed
    return _assign_chapters(blocks, tables, removed)


def write_layout(layout: Layout, path: str | Path = DEFAULT_LAYOUT_PATH) -> None:
    path = Path(path)
    content = json.dumps(
        asdict(layout), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".layout-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(content, encoding="utf-8")
        staged.replace(path)


def load_layout(path: str | Path = DEFAULT_LAYOUT_PATH) -> Layout:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Layout(
        headings=[Heading(**item) for item in data["headings"]],
        tables=[Table(**item) for item in data["tables"]],
        body_blocks=[Block(**item) for item in data["body_blocks"]],
        header_footer_removed=data["header_footer_removed"],
    )


def chapter_tree(layout: Layout) -> list[dict]:
    return [
        {"title": heading.text, "level": heading.level, "page_no": heading.page_no}
        for heading in layout.headings
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="识别 PDF 版面、章节与线框表格")
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args(argv)

    layout_path = Path(DEFAULT_LAYOUT_PATH)
    pages_path = Path("data/pages")
    audit_path = Path("语料审计.md")
    try:
        _require_distinct_files(args.pdf, audit_path, layout_path)
        _require_regular_output(layout_path)
        doc = _parse_pdf(args.pdf, pages_path, audit_path)
        _require_distinct_files(layout_path, *(Path(page.image_path) for page in doc.pages))
        layout = analyze_layout(doc)
        write_layout(layout, layout_path)
        write_audit(
            doc,
            audit_path,
            chapter_count=len(layout.headings),
            table_count=len(layout.tables),
            has_fault_code_table=any(
                keyword in table.header_text.casefold()
                for table in layout.tables
                for keyword in ("代码", "故障", "code", "fault")
            ),
        )
    except (_OutputPathConflict, UnicodeError) as exc:
        parser.error(str(exc))
    print(f"wrote {layout_path} and {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
