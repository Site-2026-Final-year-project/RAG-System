"""pgvector-backed knowledge retrieval (same Postgres DB as Express admin + chat API)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session

from database import DATABASE_URL, RagKbChunkModel, is_postgres_url


def kb_pgvector_enabled() -> bool:
    """True when DATABASE_URL is Postgres (KB tables + pgvector expected)."""
    return is_postgres_url(DATABASE_URL)


def search_kb_l2(
    db: Session,
    *,
    scope: str,
    owner_user_id: str | None,
    query_embedding: np.ndarray,
    k: int,
    manual_ids: Sequence[str] | None = None,
) -> List[str]:
    """Return up to k chunk texts by L2 distance (matches former FAISS IndexFlatL2)."""
    q = query_embedding.astype(np.float64).flatten().tolist()
    if len(q) != 384:
        raise ValueError(f"Expected embedding dim 384, got {len(q)}")

    stmt: Select[tuple[str]] = select(RagKbChunkModel.content).where(RagKbChunkModel.scope == scope)
    if owner_user_id is None:
        stmt = stmt.where(RagKbChunkModel.owner_user_id.is_(None))
    else:
        stmt = stmt.where(RagKbChunkModel.owner_user_id == owner_user_id)
    if manual_ids:
        stmt = stmt.where(RagKbChunkModel.manual_id.in_(list(manual_ids)))

    stmt = stmt.order_by(RagKbChunkModel.embedding.l2_distance(q)).limit(k)
    rows = db.scalars(stmt).all()
    return [str(r) for r in rows if r]


def count_user_chunks(db: Session, user_id: str) -> int:
    n = db.scalar(
        select(func.count())
        .select_from(RagKbChunkModel)
        .where(RagKbChunkModel.scope == "user", RagKbChunkModel.owner_user_id == user_id)
    )
    return int(n or 0)


def delete_global_chunks(db: Session) -> None:
    """Remove unified global KB only (build_index). Keeps education manual chunks (manual_id set)."""
    db.execute(
        delete(RagKbChunkModel).where(
            RagKbChunkModel.scope == "global",
            RagKbChunkModel.owner_user_id.is_(None),
            RagKbChunkModel.manual_id.is_(None),
        )
    )


def delete_chunks_for_education_manual(db: Session, education_content_id: str) -> None:
    """Replace vectors for one admin EducationContent (MANUALS) row."""
    db.execute(
        delete(RagKbChunkModel).where(
            RagKbChunkModel.scope == "global",
            RagKbChunkModel.manual_id == education_content_id,
        )
    )


def delete_user_chunks(db: Session, user_id: str) -> None:
    db.execute(
        delete(RagKbChunkModel).where(
            RagKbChunkModel.scope == "user", RagKbChunkModel.owner_user_id == user_id
        )
    )
