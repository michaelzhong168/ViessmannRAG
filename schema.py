"""G04 证据单元及共享契约中的内容哈希。"""

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass
class EU:
    eu_id: str
    doc_id: str
    page_no: int
    bbox: list[float]
    chapter: str
    text_raw: str
    model_codes: list[str]
    content_hash: str
    is_table_row: bool


def bbox_str(bbox: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.2f}" for value in bbox) + "]"


def eu_hash(doc_id: str, page_no: int, bbox: Sequence[float], text_raw: str) -> str:
    payload = f"{doc_id}|{page_no}|{bbox_str(bbox)}|{text_raw}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(value: object) -> bool:
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def validate_eu(eu: EU | Mapping[str, object]) -> list[str]:
    """返回单条 EU 的违规列表，兼容构建对象和读回的 JSON 对象。"""
    data = asdict(eu) if isinstance(eu, EU) else eu
    missing = EU.__dataclass_fields__.keys() - data.keys()
    errors = [f"缺少字段 {field}" for field in sorted(missing)]

    doc_id = data.get("doc_id")
    page_no = data.get("page_no")
    if not isinstance(doc_id, str) or not doc_id.strip():
        errors.append("doc_id 必须是非空字符串")
    if type(page_no) is not int or page_no < 1:
        errors.append("page_no 必须是正整数")

    eu_id = data.get("eu_id")
    if (
        not isinstance(eu_id, str)
        or not isinstance(doc_id, str)
        or type(page_no) is not int
        or not re.fullmatch(
            rf"{re.escape(doc_id)}:p{page_no:03d}:(?:[0-9]{{2}}|[1-9][0-9]{{2,}})",
            eu_id,
        )
    ):
        errors.append("eu_id 必须匹配 doc_id:pNNN:idx")

    bbox = data.get("bbox")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(_finite_number(value) for value in bbox)
        or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
    ):
        errors.append("bbox 必须包含四个有限数，且坐标严格递增")

    chapter = data.get("chapter")
    if not isinstance(chapter, str) or not chapter.strip():
        errors.append("chapter 必须是非空字符串")
    text_raw = data.get("text_raw")
    if not isinstance(text_raw, str) or not text_raw:
        errors.append("text_raw 必须是非空字符串")

    model_codes = data.get("model_codes")
    if not isinstance(model_codes, list) or not all(
        isinstance(code, str) for code in model_codes
    ):
        errors.append("model_codes 必须是字符串列表")

    content_hash = data.get("content_hash")
    if not isinstance(content_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", content_hash
    ):
        errors.append("content_hash 必须是 64 位小写十六进制字符串")
    if type(data.get("is_table_row")) is not bool:
        errors.append("is_table_row 必须是 bool")
    return errors
