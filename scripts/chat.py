from __future__ import annotations

import os
import warnings
from typing import List

# Must run before importing torch/numpy/faiss: duplicate OpenMP runtimes often segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

INDEX_PATH = "models/faiss_index"
DOCS_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_faiss_and_docs() -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_PATH}. Run `python scripts/build_index.py` first."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Missing docs file: {DOCS_PATH}. Run `python scripts/build_index.py` first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    return index, docs


def get_llm_device() -> str:
    """Pick device for the causal LM. Apple GPU off by default — TinyLlama + MPS often segfaults."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        if os.environ.get("RAG_USE_MPS", "").lower() in ("1", "true", "yes"):
            return "mps"
        return "cpu"
    return "cpu"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = get_llm_device()
    if torch.backends.mps.is_available() and device == "cpu":
        print("Using device: cpu (LLM; MPS off by default — set RAG_USE_MPS=1 to try Apple GPU)")
    else:
        print(f"Using device: {device}")

    # Load LLM before SentenceTransformer + FAISS to reduce native-library init clashes on macOS.
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        clean_up_tokenization_spaces=False,
    )

    if device == "cuda":
        torch_dtype = torch.float16
    elif device == "mps":
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch_dtype,
    ).to(device)
    llm.eval()

    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    index, docs = load_faiss_and_docs()

    def retrieve(query: str, k: int = 3) -> List[str]:
        q_emb = embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        # FAISS expects shape (n_queries, dim)
        _, indices = index.search(q_emb, k)

        results: List[str] = []
        for i in indices[0].tolist():
            if 0 <= i < len(docs):
                results.append(docs[i])
        return results

    def generate_answer(query: str) -> str:
        context_chunks = retrieve(query, k=3)
        context = "\n".join(context_chunks) if context_chunks else "(no relevant context found)"

        prompt = f"""
You are a car assistant. Answer based on the context.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer_tokens = outputs[0][input_len:]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # CLI loop
    while True:
        q = input("\nAsk about your car (type 'exit' to quit): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        answer = generate_answer(q)
        print("\n" + answer)


if __name__ == "__main__":
    main()

