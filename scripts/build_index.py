from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

from database import RagKbChunkModel, SessionLocal
from rag_kb import delete_global_chunks, kb_pgvector_enabled

DATA_PATH = "data/processed/unified_docs.txt"
INDEX_PATH = "models/faiss_index"
DOCS_OUT_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _write_faiss(docs: list[str], embeddings: np.ndarray) -> None:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")


def _sync_global_to_postgres(docs: list[str], embeddings: np.ndarray) -> None:
    with SessionLocal() as db:
        delete_global_chunks(db)
        for i, text in enumerate(docs):
            emb = embeddings[i].astype(np.float64).flatten().tolist()
            db.add(
                RagKbChunkModel(
                    id=str(uuid4()),
                    scope="global",
                    owner_user_id=None,
                    manual_id=None,
                    chunk_index=i,
                    content=text,
                    embedding=emb,
                    embedding_model=EMBEDDING_MODEL,
                )
            )
        db.commit()


def _use_postgres_for_kb() -> bool:
    if os.environ.get("RAG_KB_BACKEND", "auto").lower().strip() == "faiss":
        return False
    return kb_pgvector_enabled()


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing dataset file: {DATA_PATH}. Run `python scripts/build_knowledge_base.py` "
            f"(place CSVs under data/raw/; uses DelucionQA unless --skip-qa) or update DATA_PATH."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    if not docs:
        raise ValueError(f"No documents found in {DATA_PATH}.")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(docs)} documents...")
    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    if _use_postgres_for_kb():
        print("Writing global KB to PostgreSQL (rag_kb_chunks, scope=global)...")
        _sync_global_to_postgres(docs, embeddings)
        print("PostgreSQL sync OK.")
    else:
        print("Writing FAISS index + models/docs.txt ...")
        _write_faiss(docs, embeddings)
        print("Index built successfully!")
        print(f"- FAISS index: {INDEX_PATH}")
        print(f"- Stored docs : {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
