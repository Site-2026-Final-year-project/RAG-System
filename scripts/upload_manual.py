from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from database import RagKbChunkModel, RagUserProfileModel, SessionLocal
from rag_kb import delete_user_chunks, kb_pgvector_enabled

MODEL = "all-MiniLM-L6-v2"


def detect_car_model(text: str) -> str:
    patterns = (
        r"\b(Toyota\s+[A-Za-z0-9\-]+)\b",
        r"\b(Honda\s+[A-Za-z0-9\-]+)\b",
        r"\b(Hyundai\s+[A-Za-z0-9\-]+)\b",
        r"\b(Ford\s+[A-Za-z0-9\-]+)\b",
        r"\b(BMW\s+[A-Za-z0-9\-]+)\b",
        r"\b(Mercedes[- ]Benz\s+[A-Za-z0-9\-]+)\b",
        r"\b(Maruti(?:\s+Suzuki)?\s+[A-Za-z0-9\-]+)\b",
        r"\b(Kia\s+[A-Za-z0-9\-]+)\b",
        r"\b(Tata\s+[A-Za-z0-9\-]+)\b",
        r"\b(Mahindra\s+[A-Za-z0-9\-]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return "Unknown Model"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, size: int = 200) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def _use_postgres_for_kb() -> bool:
    if os.environ.get("RAG_KB_BACKEND", "auto").lower().strip() == "faiss":
        return False
    return kb_pgvector_enabled()


def _sync_user_to_postgres(
    user_id: str,
    chunks: list[str],
    embeddings: np.ndarray,
    model_name: str,
    manual_id: str | None,
) -> None:
    with SessionLocal() as db:
        delete_user_chunks(db, user_id)
        for i, text in enumerate(chunks):
            emb = embeddings[i].astype(np.float64).flatten().tolist()
            db.add(
                RagKbChunkModel(
                    id=str(uuid4()),
                    scope="user",
                    owner_user_id=user_id,
                    manual_id=manual_id,
                    chunk_index=i,
                    content=text,
                    embedding=emb,
                    embedding_model=MODEL,
                )
            )
        row = db.get(RagUserProfileModel, user_id)
        if row is None:
            db.add(RagUserProfileModel(user_id=user_id, vehicle_meta=model_name))
        else:
            row.vehicle_meta = model_name
        db.commit()


def build_user_index(
    pdf_path: Path,
    user_id: str = "user1",
    chunk_words: int = 200,
    manual_id: str | None = None,
) -> dict[str, str | int]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Extracting text from PDF: {pdf_path}")
    text = extract_text(pdf_path)
    if not text:
        raise ValueError("No extractable text found in PDF. Try another PDF or add OCR later.")

    chunks = chunk_text(text, size=chunk_words)
    if not chunks:
        raise ValueError("No chunks produced from PDF text.")
    print(f"Total chunks: {len(chunks)}")
    model_name = detect_car_model(text)
    print(f"Detected car model: {model_name}")

    st_model = SentenceTransformer(MODEL)
    embeddings = st_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    if _use_postgres_for_kb():
        print("Writing user manual chunks to PostgreSQL (rag_kb_chunks, scope=user)...")
        _sync_user_to_postgres(user_id, chunks, embeddings, model_name, manual_id)
        print(f"User manual indexed in DB for user_id='{user_id}'")
        return {
            "user_id": user_id,
            "model_name": model_name,
            "chunks": len(chunks),
            "backend": "postgres",
        }

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    user_dir = Path("models") / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(user_dir / "faiss_index"))
    with open(user_dir / "docs.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")
    with open(user_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(model_name + "\n")

    print(f"User manual indexed successfully for user_id='{user_id}'")
    print(f"- Index: {user_dir / 'faiss_index'}")
    print(f"- Docs : {user_dir / 'docs.txt'}")
    print(f"- Meta : {user_dir / 'meta.txt'}")
    return {
        "user_id": user_id,
        "model_name": model_name,
        "chunks": len(chunks),
        "index_path": str(user_dir / "faiss_index"),
        "docs_path": str(user_dir / "docs.txt"),
        "backend": "faiss",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a user-specific KB from a manual PDF.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to manual PDF")
    parser.add_argument("--user-id", default="user1", help="User id (Postgres owner_user_id or models/<id>/)")
    parser.add_argument("--chunk-words", type=int, default=200, help="Words per chunk")
    parser.add_argument(
        "--manual-id",
        default=None,
        help="Optional UUID of manual row from Express DB (stored on rag_kb_chunks.manual_id)",
    )
    args = parser.parse_args()

    build_user_index(
        args.pdf,
        user_id=args.user_id,
        chunk_words=args.chunk_words,
        manual_id=args.manual_id,
    )


if __name__ == "__main__":
    main()
