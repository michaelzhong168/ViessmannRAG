"""从证据索引检索原文及引用所需元数据。"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from schema import EU
from search_terms import WORD, word_query

DEFAULT_EU_PATH = "data/eu.jsonl"
DEFAULT_INDEX_PATH = "data/fts5.db"


def load_eu(path: str | Path = DEFAULT_EU_PATH) -> list[EU]:
    with Path(path).open(encoding="utf-8") as source:
        return [EU(**json.loads(line)) for line in source if line.strip()]


def _ranked_rows(
    db: sqlite3.Connection, index: str, match: str, model: str | None, k: int,
) -> list[sqlite3.Row]:
    # 受控章节匹配的权重较高，使属于目标章节的短步骤和正文一起召回。
    rank = "bm25(eu_words, 0.0, 3.0, 1.0)" if index == "eu_words" else "bm25(eu_fts)"
    # index 只由内部调用方指定。查询词和型号通过 SQL 参数绑定。
    return db.execute(
        f"""
        SELECT meta.*
        FROM {index} JOIN eu_meta AS meta ON meta.eu_id = {index}.eu_id
        WHERE {index} MATCH ?
          AND (? IS NULL OR json_array_length(meta.model_codes) = 0
               OR EXISTS (SELECT 1 FROM json_each(meta.model_codes) WHERE value = ?))
        ORDER BY {rank}, meta.eu_id
        LIMIT ?
        """,
        (match, model, model, k),
    ).fetchall()


def retrieve(
    query: str, model: str | None = None, k: int = 6, *,
    db_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[EU]:
    parts = WORD.findall(query)
    if not parts or k <= 0:
        return []
    match = " OR ".join(f'"{part}"' for part in parts)
    model = model.strip().upper() if model else None
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        rows = _ranked_rows(db, "eu_fts", match, model, k)
        if not rows and db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'eu_words' AND type = 'table'"
        ).fetchone():
            segmented_match = word_query(query, parts)
            if segmented_match:
                rows = _ranked_rows(db, "eu_words", segmented_match, model, k)

    evidence = []
    for row in rows:
        fields = dict(row)
        fields["bbox"] = json.loads(fields["bbox"])
        fields["model_codes"] = json.loads(fields["model_codes"])
        fields["is_table_row"] = bool(fields["is_table_row"])
        evidence.append(EU(**fields))
    return evidence
