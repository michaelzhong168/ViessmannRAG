"""验证 R02-01 JSONL 标注和核定 PDF 原页锚点。"""

import argparse
import json
import math
import re
from hashlib import sha256
from pathlib import Path

import pymupdf

from manual_rag.source import CONTROL_TYPES, load_document

_EXPECTED_CODES = (
    None, "missing_control_type", "clarify_fault", "control_type_mismatch",
    "insufficient_evidence", "safety_notice",
)


def normalize_evidence_text(text: str) -> str:
    """忽略 PDF 排版空白，保留引文中的其余字符。"""
    return "".join(text.split())


def is_identifier(value: object) -> bool:
    """检查可作为本地文件名的题集标识符。"""
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is not None


def _required_text(record: dict, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须为非空字符串")


def _check_anchor(anchor) -> None:
    if not isinstance(anchor, dict):
        raise ValueError("anchor 必须为对象")
    for key in ("doc_id", "pdf_version", "quote"):
        _required_text(anchor, key)
    if not re.fullmatch(r"[a-f0-9]{64}", anchor["pdf_version"]):
        raise ValueError("pdf_version 必须为 PDF 的 SHA-256")
    if type(anchor.get("page_no")) is not int or anchor["page_no"] < 1:
        raise ValueError("page_no 必须为从 1 开始的物理页码")
    if "bbox" in anchor:
        box = anchor["bbox"]
        if (
            not isinstance(box, list) or len(box) != 4
            or any(type(value) not in (int, float) or not -math.inf < value < math.inf for value in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError("bbox 必须为有限坐标组成的非空矩形")


def _check_question(row) -> None:
    if not isinstance(row, dict):
        raise ValueError("每行必须为一个题目对象")
    for key in ("dataset_version", "id", "intent_id"):
        if not is_identifier(row.get(key)):
            raise ValueError(f"{key} 必须为小写 ASCII 标识符")
    _required_text(row, "question")
    if row.get("split") not in ("debug", "holdout"):
        raise ValueError("split 必须为 debug 或 holdout")
    if "control_type" not in row or row["control_type"] not in (*CONTROL_TYPES, None):
        raise ValueError("control_type 须为操作类型或 null")
    if "expected_error_code" not in row or row["expected_error_code"] not in _EXPECTED_CODES:
        raise ValueError("expected_error_code 须为核定问答状态或 null")
    groups = row.get("evidence_groups")
    if not isinstance(groups, list) or bool(groups) != (row["expected_error_code"] is None):
        raise ValueError("应答题必须有证据组，其余题填写空数组")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("证据组必须为对象")
        _required_text(group, "required_fact")
        anchors = group.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("每个必要事实至少需要一个原页锚点")
        for anchor in anchors:
            _check_anchor(anchor)


def load_question_set(path: str | Path) -> list[dict]:
    """读取同一版本题集，检查内容结构和意图分集。"""
    rows = []
    ids = set()
    splits = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                _check_question(row)
                if row["id"] in ids:
                    raise ValueError("题目 ID 重复")
                if rows and row["dataset_version"] != rows[0]["dataset_version"]:
                    raise ValueError("题集版本不一致")
                if splits.setdefault(row["intent_id"], row["split"]) != row["split"]:
                    raise ValueError("同一意图跨越调试集和留出集")
                ids.add(row["id"])
                rows.append(row)
            except ValueError as exc:
                reason = "无效 JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(f"第 {line_no} 行：{reason}") from exc
    if not rows:
        raise ValueError("题集不能为空")
    return rows


def check_pdf_anchors(rows: list[dict], pdf_path: Path, scopes_path: Path) -> None:
    """按单份已核定 PDF 的版本、页段、类型和原文验证锚点。"""
    document = load_document(pdf_path, scopes_path)
    contents = pdf_path.read_bytes()
    version = document.metadata["pdf_version"]
    if sha256(contents).hexdigest() != version:
        raise ValueError("PDF 在校验期间发生变化")
    approved = {page["page_no"]: page["control_type"] for page in document.metadata["pages"]}
    with pymupdf.open(stream=contents, filetype="pdf") as pdf:
        for row in rows:
            for group in row["evidence_groups"]:
                for anchor in group["anchors"]:
                    page_no = anchor["page_no"]
                    location = f"题目 {row['id']}，物理页 {page_no}"
                    if anchor["doc_id"] != document.id_ or anchor["pdf_version"] != version:
                        raise ValueError(f"{location}：来源或 PDF 版本不符")
                    if page_no not in approved:
                        raise ValueError(f"{location}：物理页未核定")
                    if row["control_type"] is not None and row["control_type"] != approved[page_no]:
                        raise ValueError(f"{location}：锚点操作类型不符")
                    page = pdf[page_no - 1]
                    clip = None
                    if "bbox" in anchor:
                        clip = pymupdf.Rect(anchor["bbox"])
                        page_bounds = page.rect * page.derotation_matrix
                        if not page_bounds.contains(clip):
                            raise ValueError(f"{location}：bbox 超出物理页")
                    text = normalize_evidence_text(page.get_text(clip=clip))
                    quote = normalize_evidence_text(anchor["quote"])
                    first = text.find(quote)
                    if first < 0 or text.find(quote, first + 1) >= 0:
                        raise ValueError(f"{location}：原文未唯一匹配，请核对引文或用 bbox 消歧")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, help="题集 JSONL 路径")
    parser.add_argument("--pdf", type=Path, help="本地核定 PDF")
    parser.add_argument("--scopes", type=Path, help="核定物理页段 JSON")
    args = parser.parse_args(argv)
    if (args.pdf is None) != (args.scopes is None):
        parser.error("--pdf 和 --scopes 须同时提供")
    try:
        rows = load_question_set(args.questions)
        if args.pdf is not None:
            check_pdf_anchors(rows, args.pdf, args.scopes)
    except (OSError, ValueError, pymupdf.FileDataError) as exc:
        parser.exit(1, f"校验失败：{exc}\n")
    print(f"题集 {rows[0]['dataset_version']}，共 {len(rows)} 题，"
          f"调试集 {sum(row['split'] == 'debug' for row in rows)} 题，"
          f"留出集 {sum(row['split'] == 'holdout' for row in rows)} 题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
