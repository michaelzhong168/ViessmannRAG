"""将 G03 正文块与表格数据行转换为可溯源的证据单元。"""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.layout import Block, Layout, Table, _layout_order, load_layout
from pipeline.parse import (
    Document,
    _parse_pdf,
    _require_distinct_files,
    _require_regular_output,
    write_audit,
)
from schema import EU, eu_hash, validate_eu

DEFAULT_EU_PATH = "data/eu.jsonl"
MAX_BODY_LENGTH = 800
SENTENCE_ENDS = "。！？.!?"
KNOWN_MODEL_CODES = ("B1KA", "B1HA", "L1P20-A1JF", "L1P24-A1JF")
MODEL_CODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    + "|".join(map(re.escape, KNOWN_MODEL_CODES))
    + r")(?![A-Za-z0-9_-])"
)
TOKEN_PATTERN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"(?![A-Za-z0-9_-])|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"
)


def _body_segments(text: str) -> Iterator[str]:
    # 只保护含数字的 token，普通拉丁字母串可以硬切。
    tokens = iter(TOKEN_PATTERN.finditer(text))
    token = next(tokens, None)
    start = 0
    while len(text) - start > MAX_BODY_LENGTH:
        limit = start + MAX_BODY_LENGTH
        window = text[start:limit]
        boundary = max(window.rfind(mark) for mark in SENTENCE_ENDS)
        if boundary == -1:
            boundary = window.rfind("\n")

        if boundary >= 0:
            end = start + boundary + 1
        else:
            end = limit
            while token is not None and token.end() <= limit:
                token = next(tokens, None)
            if (
                token is not None
                and start < token.start() < limit < token.end()
                and token.end() - token.start() <= MAX_BODY_LENGTH
                and any(char.isdigit() for char in token.group())
            ):
                end = token.start()
        yield text[start:end]
        start = end

    if start < len(text):
        yield text[start:]


def _item_segments(item: Block | Table) -> Iterator[tuple[list[float], str]]:
    if isinstance(item, Table):
        for bbox, cells in zip(item.row_bboxes, item.rows, strict=True):
            yield bbox, item.header_text + "\n" + " | ".join(cells)
    else:
        for text in _body_segments(item.text):
            yield item.bbox, text


def build_eu(layout: Layout, doc: Document) -> list[EU]:
    """每条 EU 只取同一正文块或同一表格的数据行与表头。"""
    doc_id = Path(doc.pdf_path).stem
    pages = {page.page_no: page for page in doc.pages}
    counts: dict[int, int] = defaultdict(int)
    eus: list[EU] = []
    for item in sorted([*layout.body_blocks, *layout.tables], key=_layout_order):
        page_no = item.page_no
        page = pages.get(page_no)
        if page is None:
            raise ValueError(f"版面引用了不存在的 PDF 页码 {page_no}")
        for bbox, text in _item_segments(item):
            # 按 G03 的章节和本条原文提取，表格原文已包含且只前置一次表头。
            matched_codes = MODEL_CODE_PATTERN.findall(item.chapter + "\n" + text)
            model_codes = list(dict.fromkeys(matched_codes))
            eu = EU(
                eu_id=f"{doc_id}:p{page_no:03d}:{counts[page_no]:02d}",
                doc_id=doc_id,
                page_no=page_no,
                bbox=list(bbox),
                chapter=item.chapter,
                text_raw=text,
                model_codes=model_codes,
                content_hash=eu_hash(doc_id, page_no, bbox, text),
                is_table_row=isinstance(item, Table),
            )
            errors = validate_eu(eu)
            if errors:
                raise ValueError(f"{eu.eu_id}：{'；'.join(errors)}")
            x0, y0, x1, y1 = eu.bbox
            if not (0 <= x0 < x1 <= page.width_pt and 0 <= y0 < y1 <= page.height_pt):
                raise ValueError(f"{eu.eu_id} 的 bbox 超出渲染页范围：{eu.bbox}")
            eus.append(eu)
            counts[page_no] += 1
    return eus


def write_jsonl(eus: list[EU], path: str | Path = DEFAULT_EU_PATH) -> None:
    path = Path(path)
    content = "".join(
        json.dumps(asdict(eu), ensure_ascii=False, allow_nan=False) + "\n"
        for eu in eus
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".eu-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_text(content, encoding="utf-8")
        staged.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成证据单元 JSONL 和语料审计")
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args(argv)

    eu_path = Path(DEFAULT_EU_PATH)
    audit_path = Path("语料审计.md")
    try:
        _require_distinct_files(args.pdf, args.layout, eu_path, audit_path)
        _require_regular_output(eu_path)
        layout = load_layout(args.layout)
        # G03 已交付页图。G04 使用临时页图取得 Document。
        with TemporaryDirectory(prefix="g04-pages-") as directory:
            doc = _parse_pdf(args.pdf, Path(directory), audit_path)
            eus = build_eu(layout, doc)
        if not eus:
            raise ValueError("版面中没有可构建的证据单元")
        write_jsonl(eus, eu_path)
        # 每条 EU 只从一个 G03 源构造，该源只有一个章节。
        write_audit(doc, audit_path, eu_count=len(eus), cross_chapter_eu_count=0)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {eu_path} and {audit_path}, eu_count={len(eus)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
