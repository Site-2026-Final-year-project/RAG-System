"""
Index admin-uploaded education PDFs (Prisma `EducationContent`, category MANUALS) into `rag_kb_chunks`.

Writes vectors to `DATABASE_URL` (RAG/pgvector). Reads `EducationContent` from
`PRISMA_DATABASE_URL` when set, otherwise `DATABASE_URL` (use one DB for both after
`prisma migrate deploy`, or split URLs when RAG and Express use different Postgres instances).
Resolves files from Express project: `EXPRESS_PROJECT_ROOT/uploads/education-manuals/<file>`
or downloads `pdf_url` over HTTP if the file is not on disk (Express must serve `/uploads`).

Usage:
  export DATABASE_URL="postgresql+psycopg://..."
  export EXPRESS_PROJECT_ROOT="/path/to/driver-garage-backend"
  python scripts/sync_education_manuals.py

Optional:
  python scripts/sync_education_manuals.py --content-id <uuid>   # one manual only
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import requests
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from database import RagKbChunkModel, SessionLocal, is_postgres_url, session_for_prisma_reads
from rag_kb import delete_chunks_for_education_manual, kb_pgvector_enabled

torch.set_num_threads(1)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS_DEFAULT = 200


def chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def resolve_local_pdf_path(pdf_url: str, express_root: Path) -> Path | None:
    """Map e.g. http://localhost:4000/uploads/education-manuals/foo.pdf -> express_root/uploads/..."""
    parsed = urlparse(pdf_url)
    rel = parsed.path.lstrip("/")
    if not rel.startswith("uploads/"):
        return None
    candidate = express_root / rel
    return candidate if candidate.is_file() else None


def load_pdf_bytes(pdf_url: str, express_root: Path | None) -> bytes:
    if express_root:
        local = resolve_local_pdf_path(pdf_url, express_root)
        if local is not None:
            return local.read_bytes()
    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        resp = requests.get(pdf_url, timeout=120)
        resp.raise_for_status()
        return resp.content
    raise FileNotFoundError(
        f"Could not load PDF. Set EXPRESS_PROJECT_ROOT to the Express repo root, or use an http(s) pdf_url. url={pdf_url!r}"
    )


def fetch_manual_rows(content_id: str | None) -> list[tuple[str, str, str]]:
    """Returns list of (id, title, pdf_url)."""
    if not is_postgres_url(os.environ.get("DATABASE_URL", "")):
        raise RuntimeError("DATABASE_URL must be PostgreSQL (same as Express).")
    if not kb_pgvector_enabled():
        raise RuntimeError("pgvector KB is disabled; set DATABASE_URL to Postgres.")

    sql = """
    SELECT id, title, pdf_url
    FROM "EducationContent"
    WHERE pdf_url IS NOT NULL
      AND category::text = 'MANUALS'
    """
    params: dict[str, str] = {}
    if content_id:
        sql += " AND id = :cid"
        params["cid"] = content_id

    try:
        with session_for_prisma_reads() as db:
            rows = db.execute(text(sql), params).fetchall()
    except ProgrammingError as e:
        if "EducationContent" in str(e.orig):
            raise RuntimeError(
                'Table "EducationContent" is missing. Either run Prisma migrations against the DB '
                "you read from (driver-garage-backend: prisma migrate deploy), or set "
                "PRISMA_DATABASE_URL to the Postgres URL where Express created that table "
                "(RAG can keep using DATABASE_URL for rag_kb_chunks)."
            ) from e
        raise

    out: list[tuple[str, str, str]] = []
    for row in rows:
        rid, title, pdf_url = row[0], row[1], row[2]
        if pdf_url:
            out.append((str(rid), str(title), str(pdf_url)))
    return out


def sync_one(
    db,
    content_id: str,
    title: str,
    pdf_url: str,
    *,
    express_root: Path | None,
    chunk_words: int,
    embed_model: SentenceTransformer,
) -> int:
    data = load_pdf_bytes(pdf_url, express_root)
    text_body = extract_text_from_pdf_bytes(data)
    if not text_body:
        raise ValueError(f"No extractable text for EducationContent id={content_id} ({title})")

    chunks = chunk_text(text_body, chunk_words)
    if not chunks:
        return 0

    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embeddings = embeddings.astype(np.float32)

    delete_chunks_for_education_manual(db, content_id)
    for i, chunk in enumerate(chunks):
        emb = embeddings[i].astype(np.float64).flatten().tolist()
        db.add(
            RagKbChunkModel(
                id=str(uuid4()),
                scope="global",
                owner_user_id=None,
                manual_id=content_id,
                chunk_index=i,
                content=f"[Education: {title}]\n{chunk}",
                embedding=emb,
                embedding_model=EMBEDDING_MODEL,
            )
        )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Express EducationContent MANUALS PDFs into rag_kb_chunks.")
    parser.add_argument(
        "--express-root",
        type=Path,
        default=None,
        help="Path to driver-garage-backend (contains uploads/education-manuals/). "
        "Default: EXPRESS_PROJECT_ROOT env.",
    )
    parser.add_argument("--content-id", type=str, default=None, help="Sync only this EducationContent id.")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS_DEFAULT)
    args = parser.parse_args()

    express_root = args.express_root or (
        Path(os.environ["EXPRESS_PROJECT_ROOT"]).resolve()
        if os.environ.get("EXPRESS_PROJECT_ROOT")
        else None
    )

    rows = fetch_manual_rows(args.content_id)
    if not rows:
        print("No MANUALS rows with pdf_url found.")
        return

    print(f"Loading embedder: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    for cid, title, pdf_url in rows:
        print(f"Indexing {cid} — {title[:60]}...")
        try:
            with SessionLocal() as db:
                n = sync_one(
                    db,
                    cid,
                    title,
                    pdf_url,
                    express_root=express_root,
                    chunk_words=args.chunk_words,
                    embed_model=embed_model,
                )
                db.commit()
            print(f"  -> {n} chunks stored (manual_id={cid})")
        except Exception as e:
            print(f"  !! FAILED: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
