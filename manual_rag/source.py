"""校验核定页段，并从本地 PDF 加载一个 LlamaIndex Document。"""
import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import pymupdf
from llama_index.core.schema import Document

CONTROL_TYPES = ("旋钮式", "按键式", "触控式")
_PAGE_SEPARATOR = "\n"


def _approved_scopes(path: Path, doc_id: str, page_count: int) -> list[tuple[int, int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("doc_id") != doc_id:
        raise ValueError("核定文件的 doc_id 须与 PDF 文件名去掉扩展名后相同")
    ranges = data.get("scopes")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("scopes 须包含至少一个核定页段")

    scopes = []
    for entry in ranges:
        if not isinstance(entry, dict):
            raise ValueError("核定页段须为 JSON 对象")
        first, last, control_type = (entry.get(key) for key in ("first_page", "last_page", "control_type"))
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= page_count:
            raise ValueError(f"核定页段须位于 PDF 物理页 1–{page_count}")
        if control_type not in CONTROL_TYPES:
            raise ValueError("核定页段的 control_type 须为旋钮式、按键式或触控式")
        scopes.append((first, last, control_type))

    scopes.sort(key=lambda scope: scope[0])
    if any(current[1] >= following[0] for current, following in zip(scopes, scopes[1:])):
        raise ValueError("核定页段不得重叠")
    return scopes


def load_document(
    pdf_path: str | Path,
    scopes_path: str | Path | None = None,
    *,
    control_type: str | None = None,
) -> Document:
    """按核定 JSON 页段或整本操作类型加载一份 PDF。

    页号从 1 开始，字符区间可直接切片 Document.text。声明或页段无效时抛出 ValueError。
    """
    if (scopes_path is None) == (control_type is None):
        raise ValueError("请选择整本操作类型或核定页段文件")
    if control_type is not None and control_type not in CONTROL_TYPES:
        raise ValueError("操作类型须为旋钮式、按键式或触控式")
    source = Path(pdf_path)
    contents = source.read_bytes()
    doc_id = source.stem
    parts, pages = [], []
    offset = 0

    with pymupdf.open(stream=contents, filetype="pdf") as pdf:
        if not pdf.is_pdf or pdf.needs_pass:
            raise ValueError("请输入可直接读取的 PDF 文件")
        page_count = len(pdf)
        if page_count == 0:
            raise ValueError("PDF 至少须有一页")
        scopes = (
            _approved_scopes(Path(scopes_path), doc_id, page_count)
            if scopes_path is not None else [(1, page_count, control_type)]
        )
        for first_page, last_page, page_type in scopes:
            for page_no in range(first_page, last_page + 1):
                text = pdf[page_no - 1].get_text()
                parts.append(text)
                pages.append({
                    "page_no": page_no, "control_type": page_type,
                    "start_char_idx": offset, "end_char_idx": offset + len(text),
                })
                offset += len(text) + len(_PAGE_SEPARATOR)

    metadata = {
        "source": source.name,
        "pdf_version": sha256(contents).hexdigest(),
        "page_count": page_count,
        "pages": pages,
    }
    return Document(
        id_=doc_id, text=_PAGE_SEPARATOR.join(parts), metadata=metadata,
        excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--scopes", type=Path, help="按物理页段声明操作类型")
    scope.add_argument("--control-type", choices=CONTROL_TYPES, help="声明整本说明书的操作类型")
    args = parser.parse_args(argv)
    try:
        document = load_document(args.pdf, args.scopes, control_type=args.control_type)
    except (OSError, ValueError, pymupdf.FileDataError) as exc:
        print(f"加载失败：{exc}", file=sys.stderr)
        return 2

    metadata = document.metadata
    summary = {
        "doc_id": document.id_,
        "pdf_version": metadata["pdf_version"],
        "page_count": metadata["page_count"],
        "approved_pages": [page["page_no"] for page in metadata["pages"]],
        "control_types": sorted({page["control_type"] for page in metadata["pages"]}),
        "pages_without_text": [
            page["page_no"] for page in metadata["pages"]
            if not document.text[page["start_char_idx"]:page["end_char_idx"]].strip()
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
