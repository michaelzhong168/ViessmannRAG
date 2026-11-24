"""从版面表格与 G04 证据单元生成故障码表。"""

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from config import DANGER_WORDS
from lookup import DEFAULT_CODES_PATH, _index_codes, verify_quotes
from pipeline.build_eu import DEFAULT_EU_PATH, MODEL_CODE_PATTERN
from pipeline.layout import DEFAULT_LAYOUT_PATH, Table, load_layout
from pipeline.parse import _require_distinct_files, _require_regular_output, write_audit
from schema import EU, validate_eu

CODE_HEADERS = {"故障代码", "故障码", "错误代码", "错误码", "fault code", "error code"}
MODEL_HEADERS = {"型号", "适用型号", "设备型号", "model", "model code"}
CODE_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*")


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


def _fault_rows(table: Table) -> Iterator[tuple[str, str, list[float], list[str] | None]]:
    headers = [" ".join(cell.casefold().split()) for cell in table.header]
    code_columns = [column for column, header in enumerate(headers) if header in CODE_HEADERS]
    if not code_columns:
        return
    model_columns = [column for column, header in enumerate(headers) if header in MODEL_HEADERS]
    if len(code_columns) != 1 or len(model_columns) > 1:
        raise ValueError(f"第 {table.page_no} 页的故障码表列名有歧义")
    code_column = code_columns[0]

    for cells, bbox in zip(table.rows, table.row_bboxes, strict=True):
        if len(cells) != len(headers):
            raise ValueError(f"第 {table.page_no} 页的故障码表数据与表头列数不同")
        code = cells[code_column].strip()
        if not CODE_PATTERN.fullmatch(code):
            continue
        models = None
        if model_columns:
            models = MODEL_CODE_PATTERN.findall(cells[model_columns[0]])
        yield code, " | ".join(cells), bbox, models


def build_codes(tables: list[Table], eus: list[EU]) -> list[dict]:
    """按页码、行框和逐字文本关联 EU，构建并验证按双键排序的故障码条目。"""
    by_row = {}
    for eu in eus:
        if eu.is_table_row:
            key = (eu.page_no, tuple(eu.bbox), eu.text_raw)
            if key in by_row:
                raise ValueError(f"重复表格行 EU：{eu.eu_id}")
            by_row[key] = eu

    codes = []
    for table in tables:
        table_models = set(MODEL_CODE_PATTERN.findall(table.chapter + "\n" + table.header_text))
        for code, quote, bbox, explicit_models in _fault_rows(table):
            key = (table.page_no, tuple(bbox), table.header_text + "\n" + quote)
            eu = by_row.get(key)
            if eu is None:
                raise ValueError(f"第 {table.page_no} 页的故障码 {code} 缺少匹配的表格行 EU")
            # 处理说明可能提到其他型号。只有明确型号列或表格范围能确定适用型号。
            models = (
                [model for model in eu.model_codes if model in table_models]
                if explicit_models is None
                else explicit_models
            )
            folded_quote = quote.casefold()
            level = "P0_DANGER" if any(word in folded_quote for word in DANGER_WORDS) else "SAFE"
            for model in dict.fromkeys(models):
                codes.append({
                    "code": code,
                    "model_code": model,
                    "quote": quote,
                    "page_no": eu.page_no,
                    "eu_id": eu.eu_id,
                    "action_level": level,
                })
    indexed = _index_codes(codes)
    return [indexed[key] for key in sorted(indexed)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成故障码 JSON 和语料审计")
    parser.add_argument("--layout", type=Path, default=Path(DEFAULT_LAYOUT_PATH))
    parser.add_argument("--eus", type=Path, default=Path(DEFAULT_EU_PATH))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_CODES_PATH))
    parser.add_argument("--audit", type=Path, default=Path("语料审计.md"))
    args = parser.parse_args(argv)

    try:
        _require_distinct_files(args.layout, args.eus, args.output, args.audit)
        _require_regular_output(args.output)
        _require_regular_output(args.audit)
        layout = load_layout(args.layout)
        eus = load_eus(args.eus)
        codes = build_codes(layout.tables, eus)
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
