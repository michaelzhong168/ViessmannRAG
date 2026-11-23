"""G02：PDF → 逐页文本 + 逐页 PNG + 语料审计。

契约（G02 §四 / §五）：
- ``parse_pdf(path, out_dir="data/pages") -> Document``
- ``write_audit(doc, path="语料审计.md", **fields) -> None``（全项目唯一生成者）
- ``RENDER_ZOOM = 2.0``

本关不做文本清洗 / 改写 / 单位换算 / EU 切分 / OCR；``page.get_text()`` 原样保留。
import 用 ``pymupdf``（不是 ``fitz``，后者在 1.28.2 会向 stderr 打 deprecation）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pymupdf

#: 像素换算常量：页图宽高 = pt × RENDER_ZOOM（规格 §五）。
RENDER_ZOOM = 2.0

#: ``text_layer`` 的乱码率阈值（规格 §五）。
GARBLED_RATIO_THRESHOLD = 0.15

#: `语料审计.md` 的 7 个基础字段，顺序即写出顺序（G02 §3.5）。
BASE_AUDIT_FIELDS = (
    "doc_id",
    "page_count",
    "text_layer_pages",
    "ocr_required_pages",
    "render_zoom",
    "has_fault_code_table",
    "model_codes_detected",
)

#: 非 doc 派生字段的 G02 默认值；下游未传且旧报告没有时才使用。
AUDIT_DEFAULTS = {
    "has_fault_code_table": "unknown",
    "model_codes_detected": "",
}

#: 计划 §6.b 的白名单 W：这些字符**不计入**乱码分子。
#:
#: 其余字符（U+FFFD、C0/C1 控制符、私用区/代理区/未分配码位、白名单外的文种
#: 等）一律计入。分母为 ``len(text)``（含空白与换行），空文本返回 0.0。
ALLOWED_CODEPOINT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0020, 0x007E),  # ASCII 可打印
    (0x00A0, 0x00FF),  # Latin-1 补充（° ± × µ ® 与 NBSP）
    (0x0370, 0x03FF),  # 希腊字母
    (0x2000, 0x206F),  # 通用标点（“ ” „ – — …）
    (0x2100, 0x214F),  # 字母式符号
    (0x2150, 0x218F),  # 数字形式（Ⅰ Ⅱ ½ №）
    (0x2190, 0x21FF),  # 箭头
    (0x2200, 0x22FF),  # 数学运算符
    (0x2500, 0x25FF),  # 制表 / 几何图形（■ □ ● ○）
    (0x2600, 0x26FF),  # 杂项符号
    (0x3000, 0x303F),  # CJK 标点（。、《》）
    (0x3400, 0x4DBF),  # CJK 扩展 A
    (0x4E00, 0x9FFF),  # CJK 统一表意
    (0xF900, 0xFAFF),  # CJK 兼容表意
    (0xFE10, 0xFE1F),  # 竖排形式
    (0xFE30, 0xFE4F),  # CJK 兼容形式
    (0xFF00, 0xFFEF),  # 全角 / 半角形式
    (0x20000, 0x2FA1F),  # CJK 扩展 B–F + 兼容补充
)

#: 空白字符显式放行（ASCII 控制区的 \t / \n / \r 不在可打印区间里）。
ALLOWED_WHITESPACE = frozenset(" \t\n\r\u3000\u00a0")


@dataclass
class Page:
    """单页解析结果（7 字段，顺序即契约顺序，G02 §五）。"""

    page_no: int
    text: str
    image_path: str
    text_layer: bool
    garbled_ratio: float
    width_pt: float
    height_pt: float


@dataclass
class Document:
    """整份 PDF 的解析结果（5 字段，含 ``pdf_path`` 供 G03 重取页对象）。"""

    doc_id: str
    page_count: int
    render_zoom: float
    pages: list[Page]
    pdf_path: str


def _garbled_ratio(text: str) -> float:
    """计算乱码率：非白名单字符数 / ``len(text)``。

    ``text`` 为空时返回 ``0.0``（避免除零；空文本的 ``text_layer`` 由
    ``len(text.strip()) > 0`` 另行判 False）。
    """
    if not text:
        return 0.0

    garbled = 0
    for char in text:
        if char in ALLOWED_WHITESPACE:
            continue
        codepoint = ord(char)
        for low, high in ALLOWED_CODEPOINT_RANGES:
            if low <= codepoint <= high:
                break
        else:
            garbled += 1
    return garbled / len(text)


def _is_text_layer(text: str, ratio: float) -> bool:
    """``text_layer`` 判定：严格 ``ratio < 0.15`` 且 strip 后非空（规格 §五）。"""
    return ratio < GARBLED_RATIO_THRESHOLD and len(text.strip()) > 0


def parse_pdf(path: str | Path, out_dir: str | Path = "data/pages") -> Document:
    """逐页提取文本并渲染 PNG，返回 :class:`Document`。

    ``out_dir`` 不存在则创建；页图命名 ``page_{page_no:03d}.png``（1-based）。
    ``text`` 保持 ``page.get_text()`` 原文，不 strip、不清洗。
    """
    pdf_path = Path(path)
    image_dir = Path(out_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    matrix = pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)
    pages: list[Page] = []
    pdf = pymupdf.open(pdf_path)
    try:
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text()
            image_path = image_dir / f"page_{page_no:03d}.png"
            pixmap = page.get_pixmap(matrix=matrix)
            pixmap.save(str(image_path))

            ratio = _garbled_ratio(text)
            pages.append(
                Page(
                    page_no=page_no,
                    text=text,
                    image_path=str(image_path),
                    text_layer=_is_text_layer(text, ratio),
                    garbled_ratio=ratio,
                    width_pt=page.rect.width,
                    height_pt=page.rect.height,
                )
            )
        return Document(
            doc_id=pdf_path.stem,
            page_count=pdf.page_count,
            render_zoom=RENDER_ZOOM,
            pages=pages,
            pdf_path=str(pdf_path),
        )
    finally:
        pdf.close()


def _doc_audit_fields(doc: Document) -> dict[str, object]:
    """由 ``doc`` 派生的审计字段（每次 ``write_audit`` 都重算）。

    计划 §6.c 文字写「6 个」但只枚举 5 个；``has_fault_code_table`` /
    ``model_codes_detected`` 属调用方字段，只在缺失时回落默认值，不能每次
    重置——否则 G03/G04/G05 追加的旧字段会被清掉（违反「旧字段不丢」）。
    """
    return {
        "doc_id": doc.doc_id,
        "page_count": doc.page_count,
        "text_layer_pages": [page.page_no for page in doc.pages if page.text_layer],
        "ocr_required_pages": [
            page.page_no for page in doc.pages if not page.text_layer
        ],
        "render_zoom": doc.render_zoom,
    }


def _read_audit_table(path: str | Path) -> dict[str, object]:
    """读取旧 ``语料审计.md`` 的 markdown 表；不存在 / 坏掉时返回空 dict。"""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}

    values: dict[str, object] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if len(cells) != 2:
            continue
        key, raw = cells
        if not key or key == "字段":
            continue
        if set(key) <= {"-", ":", " "}:
            continue
        try:
            values[key] = json.loads(raw)
        except (ValueError, TypeError):
            values[key] = raw
    return values


def _format_audit_value(value: object) -> str:
    """列表 / 布尔 / None 用 JSON 字面量，字符串原样，其余 ``str()``。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_audit(
    doc: Document, path: str | Path = "语料审计.md", **fields: object
) -> None:
    """生成 / 重写 ``语料审计.md``（全项目唯一生成者）。

    合并策略（计划 §6.c）：

    1. 由 ``doc`` 派生的字段每次重算，覆盖旧报告中的同名值；
    2. ``**fields`` 显式传入的键覆盖一切（下游 G03/G04/G05 只许调用本函数）；
    3. 旧报告里的其它字段（如下游追加的 ``chapter_count``）保留；
    4. 7 个基础字段缺失时回落内置默认值（``unknown`` / 空串）；
    5. 固定字段顺序、不写时间戳 → 同一 ``doc`` 重跑字节一致。
    """
    merged = _read_audit_table(path)
    merged.update(_doc_audit_fields(doc))
    for key, default in AUDIT_DEFAULTS.items():
        merged.setdefault(key, default)
    merged.update(fields)

    ordered: dict[str, object] = {key: merged[key] for key in BASE_AUDIT_FIELDS}
    ordered.update(
        {key: merged[key] for key in sorted(merged) if key not in BASE_AUDIT_FIELDS}
    )

    lines = ["# 语料审计", "", "| 字段 | 值 |", "| --- | --- |"]
    lines.extend(
        f"| {key} | {_format_audit_value(value)} |" for key, value in ordered.items()
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PDF 转逐页文本 + 逐页 PNG + 语料审计")
    parser.add_argument("pdf", type=Path, help="输入 PDF 路径")
    parser.add_argument(
        "--out-dir", default="data/pages", help="页图输出目录（默认 data/pages）"
    )
    parser.add_argument(
        "--audit", default="语料审计.md", help="审计报告路径（默认 语料审计.md）"
    )
    args = parser.parse_args(argv)

    doc = parse_pdf(args.pdf, args.out_dir)
    write_audit(doc, args.audit)
    print(
        f"parsed doc_id={doc.doc_id} page_count={doc.page_count} "
        f"out_dir={args.out_dir} audit={args.audit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
