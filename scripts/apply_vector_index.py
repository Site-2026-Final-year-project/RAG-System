#!/usr/bin/env python3
"""
Create a pgvector index on rag_kb_chunks.embedding for fast RAG retrieval.

After many manuals are synced, queries without this index scan the whole table.

Usage:
  export DATABASE_URL="postgresql+psycopg://..."
  python scripts/apply_vector_index.py

Tries HNSW first; falls back to IVFFlat if the server/pgvector build lacks HNSW.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from database import DATABASE_URL, is_postgres_url

HNSW_SQL = """
CREATE INDEX IF NOT EXISTS ix_rag_kb_chunks_embedding_hnsw
  ON rag_kb_chunks
  USING hnsw (embedding vector_l2_ops)
"""

IVFFLAT_SQL = """
CREATE INDEX IF NOT EXISTS ix_rag_kb_chunks_embedding_ivfflat
  ON rag_kb_chunks
  USING ivfflat (embedding vector_l2_ops)
  WITH (lists = 200)
"""


def main() -> int:
    if not is_postgres_url(DATABASE_URL):
        print("DATABASE_URL must be a PostgreSQL URL (pgvector).", file=sys.stderr)
        return 1

    engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT", future=True, pool_pre_ping=True)

    with engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'rag_kb_chunks'"
            )
        ).scalar()
        if not exists:
            print(
                "Table rag_kb_chunks not found. Apply docs/schema_rag_pgvector.sql or start Docker "
                "with docker/initdb/02-rag_kb.sql, then sync manuals.",
                file=sys.stderr,
            )
            return 1

        try:
            conn.execute(text(HNSW_SQL))
            print("OK: ix_rag_kb_chunks_embedding_hnsw (HNSW)")
        except ProgrammingError as e:
            orig = getattr(e, "orig", None)
            msg = str(orig) if orig is not None else str(e)
            print(f"HNSW unavailable ({msg}); trying IVFFlat...", file=sys.stderr)
            try:
                conn.execute(text(IVFFLAT_SQL))
                print("OK: ix_rag_kb_chunks_embedding_ivfflat (IVFFlat)")
            except ProgrammingError as e2:
                print(f"IVFFlat failed: {e2}", file=sys.stderr)
                return 1

        conn.execute(text("ANALYZE rag_kb_chunks"))
        print("OK: ANALYZE rag_kb_chunks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
