"""校验核定页段，并从本地 PDF 加载一个 LlamaIndex Document。"""
import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path

import pymupdf
from llama_index.core.schema import Document

_MODEL_CODE = re.compile(r"[A-Z0-9]+(?:[-_][A-Z0-9]+)*")
_PAGE_SEPARATOR = "\n"


def _approved_scopes(path: Path, doc_id: str, page_count: int) -> list[tuple[int, int, list[str]]]:
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
        first, last, models = (entry.get(key) for key in ("first_page", "last_page", "model_codes"))
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= page_count:
            raise ValueError(f"核定页段须位于 PDF 物理页 1–{page_count}")
        if (
            not isinstance(models, list) or not models
            or any(not isinstance(code, str) or not _MODEL_CODE.fullmatch(code) for code in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError("核定页段须有非空、唯一的大写型号列表")
        scopes.append((first, last, models))

    scopes.sort(key=lambda scope: scope[0])
    if any(current[1] >= following[0] for current, following in zip(scopes, scopes[1:])):
        raise ValueError("核定页段不得重叠")
    return scopes


def load_document(pdf_path: str | Path, scopes_path: str | Path) -> Document:
    """从批准的物理页提取文本，记录每页在 Document.text 中的字符范围。"""
    source = Path(pdf_path)
    contents = source.read_bytes()
    doc_id = source.stem
    parts, pages = [], []
    offset = 0

    with pymupdf.open(stream=contents, filetype="pdf") as pdf:
        if not pdf.is_pdf or pdf.needs_pass:
            raise ValueError("请输入可直接读取的 PDF 文件")
        page_count = len(pdf)
        for first_page, last_page, models in _approved_scopes(Path(scopes_path), doc_id, page_count):
            for page_no in range(first_page, last_page + 1):
                text = pdf[page_no - 1].get_text()
                parts.append(text)
                pages.append({
                    "page_no": page_no, "model_codes": list(models),
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
    parser.add_argument("--scopes", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = load_document(args.pdf, args.scopes)
    except (OSError, ValueError, pymupdf.FileDataError) as exc:
        print(f"加载失败：{exc}", file=sys.stderr)
        return 2

    metadata = document.metadata
    summary = {
        "doc_id": document.id_,
        "pdf_version": metadata["pdf_version"],
        "page_count": metadata["page_count"],
        "approved_pages": [page["page_no"] for page in metadata["pages"]],
        "model_codes": sorted({model for page in metadata["pages"] for model in page["model_codes"]}),
        "pages_without_text": [
            page["page_no"] for page in metadata["pages"]
            if not document.text[page["start_char_idx"]:page["end_char_idx"]].strip()
        ],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
