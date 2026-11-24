"""从证据索引检索原文及引用所需元数据。"""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from schema import EU

DEFAULT_EU_PATH = "data/eu.jsonl"
DEFAULT_INDEX_PATH = "data/fts5.db"


def load_eu(path: str | Path = DEFAULT_EU_PATH) -> list[EU]:
    with Path(path).open(encoding="utf-8") as source:
        return [EU(**json.loads(line)) for line in source if line.strip()]


def retrieve(
    query: str, model: str | None = None, k: int = 6, *,
    db_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[EU]:
    terms = re.findall(r"[^\W_]+", query)
    if not terms or k <= 0:
        return []
    match = " OR ".join(f'"{term}"' for term in terms)
    model = model.strip().upper() if model else None
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT meta.*
            FROM eu_fts JOIN eu_meta AS meta ON meta.eu_id = eu_fts.eu_id
            WHERE eu_fts MATCH ?
              AND (? IS NULL OR json_array_length(meta.model_codes) = 0
                   OR EXISTS (SELECT 1 FROM json_each(meta.model_codes) WHERE value = ?))
            ORDER BY bm25(eu_fts), meta.eu_id
            LIMIT ?
            """,
            (match, model, model, k),
        )
        evidence = []
        for row in rows:
            fields = dict(row)
            fields["bbox"] = json.loads(fields["bbox"])
            fields["model_codes"] = json.loads(fields["model_codes"])
            fields["is_table_row"] = bool(fields["is_table_row"])
            evidence.append(EU(**fields))
        return evidence
