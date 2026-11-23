"""生成 G02 的 2 页确定性 PDF 夹具（``tests/fixtures/mini.pdf``）。

为什么需要确定性：
- 门禁命令 ``python -m tests.fixtures.make_mini_pdf`` 重跑后不能让 ``git status`` 变脏；
- PyMuPDF 默认会给 PDF 生成随机 ``/ID``，因此必须固定 metadata 并
  ``save(..., no_new_id=True)``。

用法::

    .venv/bin/python -m tests.fixtures.make_mini_pdf [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

#: A4（pt），与真实说明书一致；像素尺寸由 ``RENDER_ZOOM=2.0`` 换算。
PAGE_WIDTH = 595.276
PAGE_HEIGHT = 841.89

#: 两页夹具的期望文本（测试逐字比对，见 ``tests/test_g02.py``）。
S1 = "第一页的测试句子：中文标点，不应算乱码。"
S2 = "第二页的测试句子：按 2.0 倍渲染，宽度 595.276 pt。"

#: 固定 metadata：不写时间戳/随机值，保证同一脚本两次输出字节一致。
FIXED_METADATA = {
    "title": "G02 mini fixture",
    "author": "ViessmannRAG tests",
    "subject": "deterministic 2-page fixture",
    "keywords": "g02, fixture",
    "creator": "tests/fixtures/make_mini_pdf.py",
    "producer": "PyMuPDF",
    "creationDate": "D:20260101000000Z",
    "modDate": "D:20260101000000Z",
}

DEFAULT_OUT = Path(__file__).with_name("mini.pdf")


def build(out_path: str | Path = DEFAULT_OUT) -> Path:
    """生成 2 页 A4 夹具并返回落盘路径。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    try:
        for sentence in (S1, S2):
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            page.insert_text((72, 72), sentence, fontname="china-s", fontsize=12)
        doc.set_metadata(FIXED_METADATA)
        doc.save(str(out), no_new_id=True)
    finally:
        doc.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 G02 的 2 页确定性 PDF 夹具")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"输出路径（默认 {DEFAULT_OUT}）",
    )
    args = parser.parse_args(argv)

    path = build(args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
