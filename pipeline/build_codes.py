"""从版面表格与 G04 证据单元生成故障码表。"""

import argparse
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from config import DANGER_WORDS
from lookup import DEFAULT_CODES_PATH, _index_codes, _normalized_key, verify_quotes
from pipeline.build_eu import DEFAULT_EU_PATH, MODEL_CODE_PATTERN
from pipeline.layout import DEFAULT_LAYOUT_PATH, Table, load_layout
from pipeline.parse import _require_distinct_files, _require_regular_output, write_audit
from schema import EU, validate_eu

CODE_HEADERS = {"故障代码", "故障码", "错误代码", "错误码", "fault code", "error code"}
MODEL_HEADERS = {"型号", "适用型号", "设备型号", "model", "model code"}
CODE_TOKEN = r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*"
CODE_PATTERN = re.compile(CODE_TOKEN)
GROUPED_HEADERS = ["原因", "排除故障"]
CODE_SEPARATOR = r"\s*(?:[、,，]|或|和|及)\s*"
GROUPED_CODES = re.compile(
    rf"故障\s*(?:代\s*码|码)\s*[：:]?\s*"
    rf"({CODE_TOKEN}(?:{CODE_SEPARATOR}{CODE_TOKEN})*)"
    r"(?=\s*(?:闪烁|亮起|显示|[。.;；！!？?]|$))"
)
MODEL_IDENTIFIER = re.compile(r"[A-Z0-9]+(?:[-_][A-Z0-9]+)*")


@dataclass(frozen=True)
class ModelScope:
    """经核定的型号适用范围，起止都为 PDF 物理页码且包含端点。"""

    doc_id: str
    first_page: int
    last_page: int
    model_codes: tuple[str, ...]


@dataclass
class _FaultRow:
    codes: tuple[str, ...]
    quote: str
    bbox: list[float]
    explicit_models: list[str] | None
    grouped: bool


def load_eus(path: str | Path) -> list[EU]:
    """读取 G04 JSONL，并逐行验证 EU schema。"""
    eus = []
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}：EU 必须是 JSON 对象")
            errors = validate_eu(item)
            if errors:
                raise ValueError(f"{path}:{number}：{'；'.join(errors)}")
            eus.append(EU(**{field: item[field] for field in EU.__dataclass_fields__}))
    return eus


def load_model_scopes(path: str | Path, eus: list[EU]) -> list[ModelScope]:
    """读取文档专用型号范围，拒绝不匹配的文档、无效型号和重叠页段。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("scopes"), list):
        raise ValueError(f"{path}：型号范围需要包含 doc_id 和 scopes 数组")
    doc_id = data.get("doc_id")
    if not isinstance(doc_id, str) or {eu.doc_id for eu in eus} != {doc_id}:
        raise ValueError(f"{path}：型号范围 doc_id 与 EU 文档不一致")
    scopes = []
    for position, entry in enumerate(data["scopes"], 1):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}：第 {position} 个型号范围必须为对象")
        first, last, models = (entry.get(key) for key in ("first_page", "last_page", "model_codes"))
        if (
            type(first) is not int or type(last) is not int or not 1 <= first <= last
            or not isinstance(models, list) or not models
            or any(not isinstance(model, str) or not MODEL_IDENTIFIER.fullmatch(model) for model in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError(f"{path}：第 {position} 个型号范围的页码或型号无效")
        scopes.append(ModelScope(doc_id, first, last, tuple(models)))

    scopes.sort(key=lambda scope: scope.first_page)
    for earlier, later in zip(scopes, scopes[1:]):
        if earlier.last_page >= later.first_page:
            raise ValueError(f"{path}：型号范围的物理页段重叠")
    return scopes


def _grouped_codes(cause: str) -> tuple[str, ...]:
    codes = [
        code
        for match in GROUPED_CODES.finditer(cause)
        for code in re.split(CODE_SEPARATOR, match[1])
    ]
    # 有标识的分组允许纯十六进制两位码，例如 CC。
    if not all(any(char.isdigit() for char in code) or re.fullmatch(r"[A-Fa-f]{2}", code)
               for code in codes):
        return ()
    return tuple(dict.fromkeys(codes))


def _fault_rows(table: Table) -> Iterator[_FaultRow]:
    headers = [" ".join(cell.casefold().split()) for cell in table.header]
    code_columns = [column for column, header in enumerate(headers) if header in CODE_HEADERS]
    grouped = [header.replace(" ", "") for header in headers] == GROUPED_HEADERS
    if not code_columns and not grouped:
        return
    model_columns = [column for column, header in enumerate(headers) if header in MODEL_HEADERS]
    if len(code_columns) > 1 or len(model_columns) > 1:
        raise ValueError(f"第 {table.page_no} 页的故障码表列名有歧义")

    for cells, bbox in zip(table.rows, table.row_bboxes, strict=True):
        if len(cells) != len(headers):
            raise ValueError(f"第 {table.page_no} 页的故障码表数据与表头列数不同")
        if grouped:
            codes = _grouped_codes(cells[0])
        else:
            code = cells[code_columns[0]].strip()
            codes = (code,) if CODE_PATTERN.fullmatch(code) else ()
        if codes:
            models = MODEL_CODE_PATTERN.findall(cells[model_columns[0]]) if model_columns else None
            yield _FaultRow(codes, " | ".join(cells), bbox, models, grouped)


def _merge_grouped_evidence(previous: dict, current: dict) -> dict:
    if previous["quote"] == current["quote"]:
        return min((previous, current), key=lambda row: (row["page_no"], row["eu_id"]))
    shorter, longer = sorted((previous, current), key=lambda row: len(row["quote"]))
    # 同一条排障说明在不同章节重复时，保留明确带有附注的那条原文。
    if (
        previous["action_level"] == current["action_level"]
        and longer["quote"].startswith(shorter["quote"].rstrip() + "\n提示\n")
    ):
        return longer
    pages = sorted((previous["page_no"], current["page_no"]))
    raise ValueError(
        f"DUPLICATE_CODE: {current['code']} / {current['model_code']}，"
        f"物理页 {pages[0]}、{pages[1]} 的分组故障码证据冲突"
    )


def build_codes(
    tables: list[Table], eus: list[EU], model_scopes: list[ModelScope] | None = None
) -> list[dict]:
    """按页码、行框和逐字文本关联 EU，构建并验证按双键排序的故障码条目。"""
    by_row = {}
    for eu in eus:
        if eu.is_table_row:
            key = (eu.page_no, tuple(eu.bbox), eu.text_raw)
            if key in by_row:
                raise ValueError(f"重复表格行 EU：{eu.eu_id}")
            by_row[key] = eu

    codes = []
    grouped_entries: dict[tuple[str, str], dict] = {}
    for table in tables:
        table_models = set(MODEL_CODE_PATTERN.findall(table.chapter + "\n" + table.header_text))
        for row in _fault_rows(table):
            key = (table.page_no, tuple(row.bbox), table.header_text + "\n" + row.quote)
            eu = by_row.get(key)
            if eu is None:
                raise ValueError(f"第 {table.page_no} 页的故障码 {row.codes[0]} 缺少匹配的表格行 EU")
            models = row.explicit_models
            if models is None:
                scope_models = next((
                    scope.model_codes for scope in model_scopes or ()
                    if scope.doc_id == eu.doc_id and scope.first_page <= eu.page_no <= scope.last_page
                ), ())
                models = (
                    [model for model in eu.model_codes
                     if model in table_models and (not scope_models or model in scope_models)]
                    if table_models else list(scope_models)
                )
            folded_quote = row.quote.casefold()
            level = "P0_DANGER" if any(word in folded_quote for word in DANGER_WORDS) else "SAFE"
            for code in row.codes:
                for model in dict.fromkeys(models):
                    entry = {
                        "code": code,
                        "model_code": model,
                        "quote": row.quote,
                        "page_no": eu.page_no,
                        "eu_id": eu.eu_id,
                        "action_level": level,
                    }
                    if row.grouped:
                        code_key = _normalized_key(code, model)
                        previous = grouped_entries.get(code_key)
                        grouped_entries[code_key] = (
                            _merge_grouped_evidence(previous, entry) if previous is not None else entry
                        )
                    else:
                        codes.append(entry)
    indexed = _index_codes([*codes, *grouped_entries.values()])
    return [indexed[key] for key in sorted(indexed)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成故障码 JSON 和语料审计")
    parser.add_argument("--layout", type=Path, default=Path(DEFAULT_LAYOUT_PATH))
    parser.add_argument("--eus", type=Path, default=Path(DEFAULT_EU_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_CODES_PATH))
    parser.add_argument("--audit", type=Path, default=Path("语料审计.md"))
    parser.add_argument("--model-scopes", type=Path, help="已核定的文档物理页段与适用型号 JSON")
    args = parser.parse_args(argv)

    try:
        paths = [args.layout, args.eus, args.output, args.audit]
        if args.model_scopes is not None:
            paths.append(args.model_scopes)
        _require_distinct_files(*paths)
        _require_regular_output(args.output)
        _require_regular_output(args.audit)
        layout = load_layout(args.layout)
        eus = load_eus(args.eus)
        model_scopes = load_model_scopes(args.model_scopes, eus) if args.model_scopes is not None else None
        codes = build_codes(layout.tables, eus, model_scopes)
        verification = verify_quotes(codes, eus)
        if verification["mismatched"]:
            raise ValueError("生成的故障码引文未通过逐字核验")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".codes-", dir=args.output.parent) as directory:
            staged = Path(directory) / args.output.name
            staged.write_text(
                json.dumps(codes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            write_audit(
                None, args.audit,
                eu_count=len(eus),
                cross_chapter_eu_count=0,
                model_codes_detected=sorted({row["model_code"] for row in codes}) if codes else "",
                code_count=len(codes),
                quote_verify_rate=verification["rate"] if verification["checked"] else "N/A",
            )
            staged.replace(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output} and {args.audit}, code_count={len(codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
