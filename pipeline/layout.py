"""Extract G03 layout in the coordinate system of the rendered page."""

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
    text: str
    level: int
    page_no: int
    bbox: list[float]
    font_size: float


@dataclass
class Table:
    page_no: int
    bbox: list[float]
    header: list[str]
    rows: list[list[str]]
    row_bboxes: list[list[float]]
    chapter: str
    header_text: str


@dataclass
class Block:
    page_no: int
    bbox: list[float]
    text: str
    font_size: float
    chapter: str


@dataclass
class Layout:
    headings: list[Heading]
    tables: list[Table]
    body_blocks: list[Block]
    header_footer_removed: int


def _page_bbox(
    bbox: pymupdf.Rect, matrix: pymupdf.Matrix, bounds: pymupdf.Rect
) -> list[float]:
    return list((pymupdf.Rect(bbox) * matrix) & bounds)


def _extract_tables(
    page: pymupdf.Page, matrix: pymupdf.Matrix, bounds: pymupdf.Rect
) -> list[Table]:
    tables = []
    # find_tables expects the unrotated grid. The input PDF stays read-only.
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


def analyze_layout(
    doc: Document, top_ratio: float = 0.08, bottom_ratio: float = 0.08
) -> Layout:
    """Find text blocks and ruled tables, then assign heading levels and chapters."""
    blocks: list[Block] = []
    tables: list[Table] = []
    removed = 0
    with pymupdf.open(doc.pdf_path) as pdf:
        for page in pdf:
            matrix, bounds = page.rotation_matrix, page.rect
            page_tables = _extract_tables(page, matrix, bounds)
            tables.extend(page_tables)
            table_boxes = [pymupdf.Rect(table.bbox) for table in page_tables]
            text = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
            for block in text["blocks"]:
                lines = block["lines"]
                spans = [span for line in lines for span in line["spans"]]
                if not spans:
                    continue
                bbox = _page_bbox(block["bbox"], matrix, bounds)
                rect = pymupdf.Rect(bbox)
                if rect.is_empty:
                    continue
                center = (rect.tl + rect.br) * 0.5
                if not bounds.height * top_ratio <= center.y <= bounds.height * (1 - bottom_ratio):
                    removed += 1
                    continue
                if any(center in table_box for table_box in table_boxes):
                    continue
                block_text = "\n".join(
                    "".join(span["text"] for span in line["spans"]) for line in lines
                )
                if block_text.strip():
                    blocks.append(
                        Block(
                            page_no=page.number + 1,
                            bbox=bbox,
                            text=block_text,
                            font_size=max(span["size"] for span in spans if span["text"]),
                            chapter=UNSECTIONED,
                        )
                    )

    sizes = sorted({round(block.font_size, 1) for block in blocks}, reverse=True)
    # The smallest text size supplies the body baseline for this document.
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
