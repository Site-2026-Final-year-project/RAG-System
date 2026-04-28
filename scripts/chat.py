from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

import requests

# Repo root on sys.path (so `database` / `rag_kb` resolve when running as `python scripts/chat.py`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Must run before importing torch/numpy/faiss: duplicate OpenMP runtimes often segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from database import RagUserProfileModel, SessionLocal
from rag_kb import count_user_chunks, kb_pgvector_enabled, search_kb_l2

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

INDEX_PATH = "models/faiss_index"
DOCS_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()
FALLBACK_LLM_MODEL = os.environ.get(
    "RAG_FALLBACK_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).strip()

REMOTE_LLM_URL = os.environ.get("RAG_REMOTE_LLM_URL", "").strip()
REMOTE_LLM_SECRET = os.environ.get("RAG_REMOTE_LLM_SECRET", "").strip()

# Increase Hub network timeouts to reduce transient download failures.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def use_pgvector_for_kb() -> bool:
    """auto: Postgres URL -> pgvector; faiss: local FAISS files; pgvector: force DB (must be Postgres)."""
    backend = os.environ.get("RAG_KB_BACKEND", "auto").lower().strip()
    if backend == "faiss":
        return False
    if backend == "pgvector":
        if not kb_pgvector_enabled():
            raise RuntimeError(
                "RAG_KB_BACKEND=pgvector requires DATABASE_URL to be a PostgreSQL URL "
                "(same DB as Express + pgvector extension)."
            )
        return True
    return kb_pgvector_enabled()


def load_faiss_and_docs(index_path: str, docs_path: str) -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing FAISS index: {index_path}. Run the index builder first."
        )
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Missing docs file: {docs_path}. Run the index builder first."
        )

    index = faiss.read_index(index_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return index, docs


def load_global_index() -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_PATH}. Run `python scripts/build_index.py` first."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Missing docs file: {DOCS_PATH}. Run `python scripts/build_index.py` first."
        )

    return load_faiss_and_docs(INDEX_PATH, DOCS_PATH)


def load_user_index(user_id: str) -> tuple[faiss.Index | None, List[str]]:
    user_index_path = os.path.join("models", user_id, "faiss_index")
    user_docs_path = os.path.join("models", user_id, "docs.txt")
    if not os.path.exists(user_index_path) or not os.path.exists(user_docs_path):
        return None, []
    return load_faiss_and_docs(user_index_path, user_docs_path)


def load_user_meta(user_id: str) -> str:
    if use_pgvector_for_kb():
        with SessionLocal() as db:
            row = db.get(RagUserProfileModel, user_id)
            if row and (row.vehicle_meta or "").strip():
                return row.vehicle_meta.strip()
        return ""
    meta_path = os.path.join("models", user_id, "meta.txt")
    if not os.path.exists(meta_path):
        return ""
    with open(meta_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_llm_device() -> str:
    """Pick device for the causal LM."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        if os.environ.get("RAG_USE_MPS", "").lower() in ("1", "true", "yes"):
            return "mps"
        return "cpu"
    return "cpu"


@dataclass
class RetrievalConfig:
    k_user: int = 2
    k_global: int = 1


class RAGAssistant:
    def __init__(
        self,
        user_id: str = "user1",
        car_context: str = "",
        *,
        use_user_manual: bool = True,
    ) -> None:
        self.user_id = user_id
        self.car_context = car_context.strip()
        self._use_user_manual = use_user_manual
        self._use_pgvector = use_pgvector_for_kb()

        if use_user_manual:
            self.user_model = load_user_meta(user_id)
        else:
            self.user_model = ""

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.remote_llm_url = REMOTE_LLM_URL
        self.remote_llm_secret = REMOTE_LLM_SECRET
        self.device = get_llm_device()

        if self.remote_llm_url:
            # Defer generation to a remote service (e.g. Colab). This avoids loading local
            # transformer weights on constrained hosts (Render free/CPU instances).
            self.active_llm_model = f"remote:{self.remote_llm_url}"
            self.llm = None
            self.tokenizer = None
        else:
            if self.device == "cuda":
                torch_dtype = torch.float16
            elif self.device == "mps":
                torch_dtype = torch.float32
            else:
                torch_dtype = None

            self.active_llm_model = ""
            load_errors: List[str] = []
            for candidate_model in [LLM_MODEL, FALLBACK_LLM_MODEL]:
                if not candidate_model or candidate_model in {self.active_llm_model}:
                    continue
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate_model,
                        clean_up_tokenization_spaces=False,
                    )
                    if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        candidate_model,
                        torch_dtype=torch_dtype,
                    ).to(self.device)
                    self.llm.eval()
                    self.active_llm_model = candidate_model
                    break
                except Exception as e:  # pragma: no cover - runtime/network dependent
                    load_errors.append(f"{candidate_model}: {e}")

            if not self.active_llm_model:
                msg = " | ".join(load_errors) if load_errors else "Unknown model loading failure"
                raise RuntimeError(f"Failed to load any LLM model. {msg}")

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        if self._use_pgvector:
            self.global_index = None
            self.global_docs = []
            self.user_index = None
            self.user_docs = []
        else:
            self.global_index, self.global_docs = load_global_index()
            if use_user_manual:
                self.user_index, self.user_docs = load_user_index(user_id)
            else:
                self.user_index, self.user_docs = None, []

    @property
    def has_user_index(self) -> bool:
        if self._use_pgvector:
            with SessionLocal() as db:
                return count_user_chunks(db, self.user_id) > 0
        return self.user_index is not None and bool(self.user_docs)

    def _search(self, index: faiss.Index, docs: List[str], q_emb: np.ndarray, k: int) -> List[str]:
        _, indices = index.search(q_emb, k)
        results: List[str] = []
        for i in indices[0].tolist():
            if 0 <= i < len(docs):
                results.append(docs[i])
        return results

    def _embed_query(self, text: str) -> np.ndarray:
        return self.embed_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

    def _retrieve_once(self, q_emb: np.ndarray, config: RetrievalConfig) -> tuple[List[str], List[str]]:
        user_results: List[str] = []
        global_results: List[str] = []
        if self._use_pgvector:
            with SessionLocal() as db:
                if self._use_user_manual:
                    user_results = search_kb_l2(
                        db,
                        scope="user",
                        owner_user_id=self.user_id,
                        query_embedding=q_emb,
                        k=config.k_user,
                    )
                global_results = search_kb_l2(
                    db,
                    scope="global",
                    owner_user_id=None,
                    query_embedding=q_emb,
                    k=config.k_global,
                )
        else:
            if self.user_index is not None and self.user_docs:
                user_results = self._search(self.user_index, self.user_docs, q_emb, config.k_user)
            global_results = self._search(self.global_index, self.global_docs, q_emb, config.k_global)
        return user_results, global_results

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        car_context: str = "",
        priority_context: str = "",
    ) -> List[str]:
        config = config or RetrievalConfig()
        ctx = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and not ctx:
            ctx = self.user_model

        # Priority order:
        # 1) Health + vehicle context + query (diagnostic first)
        # 2) Vehicle context + query (manual/model-targeted retrieval)
        # 3) Raw query only (general fallback)
        candidate_queries: List[str] = []
        if priority_context.strip():
            candidate_queries.append(f"{priority_context.strip()} {ctx} {query}".strip())
        if ctx:
            candidate_queries.append(f"{ctx} {query}".strip())
        candidate_queries.append(query.strip())

        merged: List[str] = []
        seen_queries: set[str] = set()
        for q in candidate_queries:
            if not q or q in seen_queries:
                continue
            seen_queries.add(q)
            q_emb = self._embed_query(q)
            user_results, global_results = self._retrieve_once(q_emb, config)
            for c in user_results + global_results:
                if c and c not in merged:
                    merged.append(c)
            if len(merged) >= (config.k_user + config.k_global):
                break

        return merged

    def _answer_mode(self, query: str) -> str:
        q = query.lower()
        simple_hints = ("explain", "what is", "what's", "meaning", "why", "how does")
        technical_hints = ("torque", "horsepower", "engine", "spec", "diagnostic", "dtc")
        if any(h in q for h in technical_hints):
            return "technical"
        if any(h in q for h in simple_hints):
            return "simple"
        return "simple"

    def _quick_smalltalk(self, query: str) -> str | None:
        q = query.strip().lower()
        q_alpha = re.sub(r"[^a-z\s]", "", q)
        greetings = {"hi", "hello", "hey", "yo", "hii", "helo"}
        thanks = {"thanks", "thank you", "thx", "ty"}
        bye = {"bye", "goodbye", "see you"}

        if q in greetings or q_alpha in greetings:
            return (
                "Hi! I can help with your vehicle health, symptoms, and manual-based guidance. "
                "Tell me what issue you are seeing."
            )
        if q in thanks or q_alpha in thanks:
            return "You are welcome. If you want, share your car issue and I will help step by step."
        if q in bye or q_alpha in bye:
            return "Goodbye. Drive safe!"
        return None

    def _build_prompt(self, context: str, query: str, mode: str) -> str:
        if mode == "technical":
            return f"""
You are an expert automotive assistant.

Provide a precise and technical answer based on the context.

Guidelines:
- Use correct automotive terminology
- Be concise but informative
- Do NOT repeat the question
- Do NOT generate extra questions
- If unsure, say you are not certain

Context:
{context}

Question:
{query}

Answer:
""".strip()

        return f"""
You are a friendly and knowledgeable car assistant helping drivers understand their vehicles.

Answer the question using ONLY the context provided.

Guidelines:
- Speak naturally like a human expert
- Be clear and concise
- Prioritize any vehicle health issues first, then explain model/manual-specific guidance
- Do NOT repeat the question
- Do NOT generate extra questions
- If unsure, say you are not certain

Context:
{context}

Question:
{query}

Answer:
""".strip()

    def _format_answer(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return "I am not certain based on the retrieved context."
        if cleaned.endswith(("?", "!", ".")):
            return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        return (cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()) + "."

    def _tokenize_for_generation(self, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
        if self.remote_llm_url:
            raise RuntimeError("Tokenization is not used when RAG_REMOTE_LLM_URL is set.")
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                if isinstance(tokenized, dict) and "input_ids" in tokenized:
                    tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                    return tokenized, tokenized["input_ids"].shape[1]
            except Exception:
                pass

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs, inputs["input_ids"].shape[1]

    def _remote_generate(self, prompt: str) -> str:
        if not self.remote_llm_url:
            raise RuntimeError("Remote LLM URL not configured.")
        headers = {"Content-Type": "application/json"}
        if self.remote_llm_secret:
            headers["X-LLM-Secret"] = self.remote_llm_secret
        payload = {
            "prompt": prompt,
            "max_new_tokens": 120,
        }
        try:
            resp = requests.post(self.remote_llm_url, json=payload, headers=headers, timeout=90)
        except requests.RequestException as e:
            raise RuntimeError(f"Remote LLM request failed: {e}") from e
        if resp.status_code != 200:
            body = resp.text[:500]
            raise RuntimeError(f"Remote LLM error {resp.status_code}: {body}")
        data = resp.json()
        answer = (data.get("answer") or "").strip()
        return answer

    def generate_answer(self, query: str, car_context: str = "", priority_context: str = "") -> str:
        quick = self._quick_smalltalk(query)
        if quick is not None:
            return quick

        context_chunks = self.retrieve(
            query,
            config=RetrievalConfig(k_user=2, k_global=2),
            car_context=car_context,
            priority_context=priority_context,
        )
        context = "\n".join(context_chunks) if context_chunks else "(no relevant context found)"
        # Keep prompt bounded for latency and to avoid max length warnings.
        context = context[:2500]
        active_context = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and self.has_user_index:
            active_context = f"{active_context} {self.user_model}".strip()
        if priority_context.strip():
            context = f"(Priority diagnostics)\n{priority_context.strip()[:800]}\n\n{context}"
        if active_context:
            context = f"(Vehicle focus: {active_context[:300]})\n\n{context}"
        mode = self._answer_mode(query)
        prompt = self._build_prompt(context=context, query=query, mode=mode)

        if self.remote_llm_url:
            answer = self._remote_generate(prompt)
        else:
            assert self.llm is not None
            assert self.tokenizer is not None
            inputs, input_len = self._tokenize_for_generation(prompt)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in full_output:
                answer = full_output.split("Answer:")[-1].strip()
            elif "Final Answer:" in full_output:
                answer = full_output.split("Final Answer:")[-1].strip()
            else:
                answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        answer = answer.split("Question:")[0].split("Context:")[0].strip()
        return self._format_answer(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG chat: pgvector (Postgres) or local FAISS — see RAG_KB_BACKEND."
    )
    parser.add_argument(
        "--car",
        default="",
        help="Optional car context (model, year, trim). Also set RAG_CAR_CONTEXT env var.",
    )
    parser.add_argument(
        "--user-id",
        default="user1",
        help="User id for user-scoped manual chunks (Postgres rag_kb_chunks or models/<id>/).",
    )
    args = parser.parse_args()

    car_context = (args.car or os.environ.get("RAG_CAR_CONTEXT", "") or "").strip()
    assistant = RAGAssistant(
        user_id=args.user_id,
        car_context=car_context,
        use_user_manual=True,
    )

    if torch.backends.mps.is_available() and assistant.device == "cpu":
        print("Using device: cpu (set RAG_USE_MPS=1 to try Apple GPU)")
    else:
        print(f"Using device: {assistant.device}")
    print(f"LLM model requested: {LLM_MODEL}")
    print(f"LLM fallback model: {FALLBACK_LLM_MODEL}")
    print(f"LLM model loaded: {assistant.active_llm_model}")
    if assistant._use_pgvector:
        print("Knowledge retrieval: PostgreSQL + pgvector (rag_kb_chunks).")
    else:
        print("Knowledge retrieval: local FAISS + models/docs.txt.")
    if assistant.has_user_index:
        model_note = f" ({assistant.user_model})" if assistant.user_model else ""
        if assistant._use_pgvector:
            print(f"User manual chunks in DB for user_id='{args.user_id}'{model_note}")
        else:
            print(f"Loaded user manual index: models/{args.user_id}/faiss_index{model_note}")
    else:
        print(f"No user manual chunks for '{args.user_id}', using global KB only.")
    if car_context:
        print(f"Car context: {car_context}")

    while True:
        q = input("\nAsk about your car (type 'exit' to quit): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        answer = assistant.generate_answer(q, car_context=car_context)
        print("\n" + answer)


if __name__ == "__main__":
    main()
