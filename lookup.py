"""按故障码与型号做精确匹配，并核对对应的 EU 引文。"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from config import DANGER_WORDS as DANGER_WORDS
from schema import EU

DEFAULT_CODES_PATH = "data/codes.json"
ACTION_LEVELS = frozenset({"SAFE", "CAUTION", "P0_DANGER"})
_QUERY_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")
_STANDARD_SHORT_CODE = re.compile(r"[FE][0-9]{1,2}")
_CODE_LABEL = re.compile(r"(?:故障(?:代码|码)|错误(?:代码|码)|报码)[：: \t]*(?:为|是)?[：: \t]*$")


def _normalized_key(code: str, model: str) -> tuple[str, str]:
    return code.strip().upper(), model.strip().upper()


def _index_codes(codes: Sequence[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in codes:
        key = _normalized_key(row["code"], row["model_code"])
        if key in indexed:
            raise ValueError(f"DUPLICATE_CODE: {key[0]} / {key[1]}")
        indexed[key] = row
    return indexed


class _LoadedCodes(list[dict]):
    """文件行快照与同一快照的预建索引。"""

    def __init__(self, rows: list[dict]) -> None:
        super().__init__(rows)
        self.index_by_key = _index_codes(rows)


def load_codes(path: str | Path = DEFAULT_CODES_PATH) -> list[dict]:
    """读取供查表或核验的只读行快照；重复归一化键抛出 ValueError。"""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}：故障码表必须是 JSON 数组")
    for number, row in enumerate(rows, 1):
        if (
            not isinstance(row, dict)
            or row.keys() != {"code", "model_code", "quote", "page_no", "eu_id", "action_level"}
            or any(not isinstance(row[field], str) for field in (
                "code", "model_code", "quote", "eu_id", "action_level",
            ))
            or not row["code"].strip()
            or not row["model_code"].strip()
            or not row["quote"].strip()
            or not row["eu_id"].strip()
            or type(row["page_no"]) is not int
            or row["page_no"] < 1
            or row["action_level"] not in ACTION_LEVELS
        ):
            raise ValueError(f"{path}：故障码表第 {number} 条不符合六字段契约")
    return _LoadedCodes(rows)


@lru_cache(maxsize=1)
def _load_default_codes(
    path: Path, file_version: tuple[int, int, int, int, int]
) -> Sequence[dict]:
    # 文件被脚本原子替换后，进程会在下一次查表时重建字典。
    return load_codes(path)


def _default_codes() -> Sequence[dict]:
    path = Path(DEFAULT_CODES_PATH).resolve()
    stat = path.stat()
    version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _load_default_codes(path, version)


def fault_codes_in_query(query: str) -> list[str]:
    """识别明确故障码，其他两位短码只在 G05 表内确认。"""
    explicit = set()
    candidates = set()
    for token in _QUERY_TOKEN.finditer(query):
        code = token[0].upper()
        prefix = query[:token.start()]
        if _STANDARD_SHORT_CODE.fullmatch(code) or _CODE_LABEL.search(prefix):
            explicit.add(code)
        elif len(code) == 2 and code.isalnum():
            # 避免把「20 cc」等带数值的单位读成短码。
            if not (code.isalpha() and re.search(r"\d[ \t]*$", prefix)):
                candidates.add(code)

    if candidates - explicit:
        try:
            known = {row["code"].strip().upper() for row in _default_codes()}
        except FileNotFoundError:
            # 没有查表产物时，普通检索仍可独立运行。
            known = set()
        explicit.update(candidates & known)
    return sorted(explicit)


def lookup_code(code: str, model: str, codes: Sequence[dict] | None = None) -> dict | None:
    """按归一化双键查表，复用已加载快照的字典。普通行序列会临时建表。"""
    key = _normalized_key(code, model)
    if not all(key):
        return None
    if codes is None:
        codes = _default_codes()
    indexed = codes.index_by_key if isinstance(codes, _LoadedCodes) else _index_codes(codes)
    row = indexed.get(key)
    if row is None or _normalized_key(row["code"], row["model_code"]) != key:
        return None
    return row.copy()


def verify_quotes(
    codes: Sequence[dict], eus: Sequence[EU | Mapping[str, object]]
) -> dict:
    """按 EU ID、物理页和逐字子串核验；mismatched 保留对应的输入条目。"""
    by_id = {}
    for eu in eus:
        data = asdict(eu) if isinstance(eu, EU) else eu
        eu_id = data["eu_id"]
        if eu_id in by_id:
            raise ValueError(f"DUPLICATE_EU: {eu_id}")
        by_id[eu_id] = data

    mismatched = []
    for row in codes:
        eu = by_id.get(row["eu_id"])
        quote = row["quote"]
        if (
            eu is None
            or type(row["page_no"]) is not int
            or row["page_no"] != eu["page_no"]
            or not isinstance(quote, str)
            or not quote.strip()
            or quote not in eu["text_raw"]
        ):
            mismatched.append(row)
    checked = len(codes)
    return {
        "checked": checked,
        "mismatched": mismatched,
        "rate": (checked - len(mismatched)) / checked if checked else 0.0,
    }
