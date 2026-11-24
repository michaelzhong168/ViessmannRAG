"""由 EU JSONL 重建 SQLite FTS5 索引。"""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from retrieval import DEFAULT_EU_PATH, DEFAULT_INDEX_PATH, load_eu
from search_terms import word_tokens


def build_index(
    eu_path: str | Path = DEFAULT_EU_PATH, db_path: str | Path = DEFAULT_INDEX_PATH,
) -> int:
    evidence = load_eu(eu_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DROP TABLE IF EXISTS eu_fts")
        db.execute("DROP TABLE IF EXISTS eu_words")
        db.execute("DROP TABLE IF EXISTS eu_meta")
        db.execute("""
            CREATE TABLE eu_meta (
                eu_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, page_no INTEGER NOT NULL,
                bbox TEXT NOT NULL, chapter TEXT NOT NULL, text_raw TEXT NOT NULL,
                model_codes TEXT NOT NULL, content_hash TEXT NOT NULL, is_table_row INTEGER NOT NULL
            )
        """)
        db.execute("CREATE VIRTUAL TABLE eu_fts USING fts5(eu_id UNINDEXED, text, tokenize='trigram')")
        db.executemany(
            "INSERT INTO eu_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    eu.eu_id, eu.doc_id, eu.page_no, json.dumps(eu.bbox), eu.chapter,
                    eu.text_raw, json.dumps([code.strip().upper() for code in eu.model_codes]),
                    eu.content_hash, eu.is_table_row,
                )
                for eu in evidence
            ),
        )
        db.execute("INSERT INTO eu_fts SELECT eu_id, text_raw FROM eu_meta ORDER BY eu_id")
        db.execute(
            "CREATE VIRTUAL TABLE eu_words USING "
            "fts5(eu_id UNINDEXED, chapter, text, tokenize='unicode61')"
        )
        db.executemany(
            "INSERT INTO eu_words VALUES (?, ?, ?)",
            (
                (eu.eu_id, " ".join(word_tokens(eu.chapter)), " ".join(word_tokens(eu.text_raw)))
                for eu in sorted(evidence, key=lambda eu: eu.eu_id)
            ),
        )
    return len(evidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eu", default=DEFAULT_EU_PATH)
    parser.add_argument("--db", default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()
    print(f"已索引 {build_index(args.eu, args.db)} 条证据单元")


if __name__ == "__main__":
    main()
