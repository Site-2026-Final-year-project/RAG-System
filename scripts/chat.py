from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
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
from gradio_client import Client
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

# One shared model/client for all RAGAssistant instances — critical on low-RAM hosts (e.g. Render Free),
# where caching assistants per user/context would load SentenceTransformer repeatedly and OOM → 502.
_shared_sentence_transformer: SentenceTransformer | None = None
_hf_space_client_lock_target: str | None = None
_hf_space_client_singleton: Client | None = None


def _get_shared_sentence_transformer() -> SentenceTransformer:
    global _shared_sentence_transformer
    if _shared_sentence_transformer is None:
        _shared_sentence_transformer = SentenceTransformer(EMBEDDING_MODEL)
    return _shared_sentence_transformer


def _get_hf_space_client(target: str) -> Client:
    global _hf_space_client_lock_target, _hf_space_client_singleton
    if _hf_space_client_singleton is None or _hf_space_client_lock_target != target:
        _hf_space_client_lock_target = target
        _hf_space_client_singleton = Client(target)
    return _hf_space_client_singleton


LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()
FALLBACK_LLM_MODEL = os.environ.get(
    "RAG_FALLBACK_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).strip()

REMOTE_LLM_URL = os.environ.get("RAG_REMOTE_LLM_URL", "").strip()
REMOTE_LLM_SECRET = os.environ.get("RAG_REMOTE_LLM_SECRET", "").strip()
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "").strip()
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "").strip()
HF_SPACE_API_NAME = os.environ.get("HF_SPACE_API_NAME", "/predict").strip() or "/predict"

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
    k_global: int = 3


CARCARE_PERSONA_PROMPT = """
You are CarCare AI, a practical and safety-first assistant for drivers.

ROLE
- Help users with vehicle maintenance, basic troubleshooting, safe driving guidance, and understanding car features/manuals.
- Use clear, simple language for non-experts.
- Be concise, structured, and actionable.

TONE
- Calm, professional, friendly.
- Never judgmental.
- Prefer short sections and bullet points over long paragraphs.

RESPONSE STYLE (ALWAYS)
1) Quick answer (1–2 lines)
2) Steps to follow (numbered)
3) What to check/prepare (bullets)
4) When to seek a mechanic (if relevant)
5) Safety warning (if relevant)

SAFETY RULES
- Prioritize human safety over convenience or cost.
- If there is risk (brake failure, fuel leak smell, overheating, smoke, warning lights with severe symptoms), advise the user to stop driving and seek professional help.
- Do not provide instructions that bypass safety systems or legal requirements.
- If unsure, say uncertainty clearly and suggest safe next steps.

BOUNDARIES
- Do not claim real-time sensor access unless explicitly provided in user context.
- Do not invent specs; if data is missing, ask a short clarifying question.
- Do not present guesses as facts.
- Avoid complex jargon unless user asks for technical detail.

VEHICLE-AWARE BEHAVIOR
- If vehicle details are provided, tailor answers to that vehicle.
- If key details are missing, ask for only what is necessary (model/year/engine/transmission/symptom).
- If manual context exists, prefer manual-consistent guidance.

FORMATTING RULES
- Keep answers compact and readable on mobile.
- Use markdown headings and numbered steps.
- Keep each step short and concrete.
- Avoid huge blocks of text.

ESCALATION TRIGGERS (URGENT)
Immediately include “Do not continue driving” when user mentions:
- brake not responding / severe steering issues
- engine overheating warning + steam/smell
- fuel leak smell
- smoke/fire signs
- battery/electrical burning smell
- sudden loss of power in traffic

OUTPUT QUALITY
- Give practical checks users can do safely.
- Include common causes ranked by likelihood when troubleshooting.
- End with one clear “next best action”.
""".strip()


def _normalize_chat_query(query: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", query.lower()).strip().split())


def _is_identity_or_meta_query(query: str) -> bool:
    """True for questions about the assistant itself — never run manual retrieval on these."""
    q = _normalize_chat_query(query)
    if len(q) > 140:
        return False
    patterns = (
        r"^who\s+are\s+you\b",
        r"^who\s+r\s+u\b",
        r"^what\s+are\s+you\s*(\?|$)",
        r"^what\s*(?:'s|s|is)\s+your\s+name\b",
        r"^what\s+should\s+i\s+call\s+you\b",
        r"^introduce\s+yourself\b",
        r"^tell\s+me\s+about\s+yourself\b",
        r"^what\s+(?:do\s+you\s+do|can\s+you\s+do|can\s+you\s+help(?:\s+with)?|are\s+you\s+for)\b",
        r"^how\s+can\s+you\s+help\b",
        r"^are\s+you\s+(?:a\s+)?(?:bot|ai|assistant|chatgpt)\b",
        r"^what\s+is\s+your\s+purpose\b",
        r"^do\s+you\s+work\s+for\b",
    )
    return any(re.search(p, q) for p in patterns)


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
        self.llm_provider = LLM_PROVIDER
        self.hf_space_id = HF_SPACE_ID
        self.hf_space_url = HF_SPACE_URL
        self.hf_space_api_name = HF_SPACE_API_NAME
        self.device = get_llm_device()

        if self.llm_provider == "hf_space":
            target = self.hf_space_id or self.hf_space_url or "(missing HF_SPACE_ID/HF_SPACE_URL)"
            self.active_llm_model = f"hf_space:{target}"
            self.llm = None
            self.tokenizer = None
        elif self.remote_llm_url:
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

        self.embed_model = _get_shared_sentence_transformer()

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

    def _retrieve_once(
        self,
        q_emb: np.ndarray,
        config: RetrievalConfig,
        *,
        manual_ids: Sequence[str] | None = None,
    ) -> tuple[List[str], List[str]]:
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
                    manual_ids=manual_ids,
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
        manual_ids: Sequence[str] | None = None,
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
        target_limit = config.k_user + config.k_global
        for q in candidate_queries:
            if not q or q in seen_queries:
                continue
            seen_queries.add(q)
            q_emb = self._embed_query(q)
            user_results: List[str] = []
            global_results: List[str] = []
            if manual_ids and self._use_pgvector:
                # Step 1: strict manual-targeted retrieval (exact/near vehicle manual candidates).
                user_results, global_results = self._retrieve_once(
                    q_emb,
                    RetrievalConfig(k_user=config.k_user, k_global=max(1, config.k_global - 1)),
                    manual_ids=manual_ids,
                )
                if len(user_results) + len(global_results) < max(2, target_limit // 2):
                    # Step 2: bounded fallback to full global set when targeted subset is sparse.
                    _, global_fallback = self._retrieve_once(q_emb, RetrievalConfig(k_user=0, k_global=2))
                    for c in global_fallback:
                        if c and c not in global_results:
                            global_results.append(c)
            else:
                user_results, global_results = self._retrieve_once(q_emb, config)
            for c in user_results + global_results:
                if c and c not in merged:
                    merged.append(c)
            if len(merged) >= target_limit:
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
                "**Quick answer**\n\n"
                "Hi — I’m **CarCare AI**, your vehicle helper.\n\n"
                "**Steps**\n\n"
                "1. Tell me your **symptom** or question.\n"
                "2. Add **year / make / model** if you know them.\n\n"
                "**What to have ready**\n\n"
                "- Any **warning lights**\n"
                "- **When** it happens\n\n"
                "**Next best action**\n\n"
                "Describe what you’re seeing on your car."
            )
        if q in thanks or q_alpha in thanks:
            return (
                "**Quick answer**\n\n"
                "Glad to help.\n\n"
                "**Next best action**\n\n"
                "If anything else comes up with your car, tell me the symptom and vehicle details."
            )
        if q in bye or q_alpha in bye:
            return (
                "**Quick answer**\n\n"
                "Take care — drive safe.\n\n"
                "**Safety**\n\n"
                "If a warning light or unusual smell/smoke appears, pull over when safe and get professional help."
            )
        return None

    def _identity_intro_reply(self) -> str:
        return (
            "**Quick answer**\n\n"
            "I’m **CarCare AI** — a practical, safety-first assistant for drivers.\n\n"
            "**What I help with**\n\n"
            "1. Maintenance basics and service intervals\n"
            "2. Safe troubleshooting when something feels wrong\n"
            "3. Understanding dashboard warnings and owner-manual-style guidance (when available)\n\n"
            "**What to tell me**\n\n"
            "- **Year / make / model**\n"
            "- **Symptom** (noise, smell, light, leak, vibration)\n"
            "- **When** it happens\n\n"
            "**Important**\n\n"
            "- I **don’t** have live sensor/OBD data unless your app sends it in context.\n"
            "- For **brake/steering failures**, **strong burning smells**, **smoke**, **overheating**, or **fuel odor**: "
            "**Do not continue driving** — get professional help.\n\n"
            "**Next best action**\n\n"
            "What’s going on with your car today?"
        )

    def _wants_vehicle_status(self, query: str) -> bool:
        q = " ".join(re.sub(r"[^\w\s]", " ", query.lower()).split())
        phrases = (
            "tell me about my car",
            "about my car",
            "my car status",
            "car status",
            "vehicle status",
            "how is my car",
            "health of my car",
            "overall health",
            "maintenance status",
            "condition of my car",
            "how my car is doing",
            "status of my car",
            "vehicle health",
        )
        return any(p in q for p in phrases)

    def _vehicle_status_missing_snapshot_message(self, vehicle_focus: str) -> str:
        label = vehicle_focus.strip() or "this vehicle"
        return (
            f"I don’t have a maintenance-health snapshot linked for **{label}** in this chat session.\n\n"
            "Make sure you **start the assistant from your vehicle screen** (or send **vehicle_id**) so the "
            "server can load **Vehicle** + **VehicleMaintenanceHealth** from the database.\n\n"
            "After that, ask again for status—I will report overall health and each component percentage."
        )

    def _vehicle_status_reply_from_priority(self, priority_context: str, vehicle_focus: str) -> str:
        lines_raw = [ln.strip() for ln in priority_context.splitlines() if ln.strip()]
        label = vehicle_focus.strip() or "your vehicle"
        for ln in lines_raw:
            if ln.lower().startswith("vehicle:"):
                label = ln.split(":", 1)[1].strip()
                break
        out: List[str] = [
            f"**Vehicle status — {label}**",
            "",
            "Summary from your **saved vehicle profile** (latest maintenance-health snapshot). "
            "This reflects calculated maintenance scores, not live OBD readings.",
            "",
        ]
        for ln in lines_raw:
            out.append(f"• {ln}")
        out.extend(
            [
                "",
                "**Disclaimer:** Values are planning aids only. For safety-critical faults or warning lamps, "
                "follow your owner manual and consult a qualified technician.",
            ]
        )
        return "\n".join(out)

    def _build_prompt(self, context: str, query: str, mode: str) -> str:
        if mode == "technical":
            return f"""
{CARCARE_PERSONA_PROMPT}

ADDITIONAL TECHNICAL MODE RULES
- If the user asks who you are, what you are, or your name: answer as **CarCare AI**; **do not** dump unrelated manual excerpts.
- Use correct automotive terminology and stay factual.
- If a Vehicle profile snapshot lists maintenance-health percentages, report them accurately.
- Do NOT invent DTCs, measurements, or specs.
- If information is missing, state what is missing and ask a brief clarifying question.
- Do not promote the manuals brand or name; focus on the content.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        return f"""
{CARCARE_PERSONA_PROMPT}

ADDITIONAL RESPONSE RULES
- If the user asks who you are, what you are, or your name: answer as **CarCare AI** in plain language; **do not**
  paste unrelated numbered lists from manuals.
- Answer using the context below.
- When context includes a Vehicle profile snapshot, treat those vehicle facts and percentages as authoritative.
- Use manual excerpts as supporting detail for procedures and warnings.
- If manual excerpts are missing, still answer from available vehicle/profile context.
- Do NOT repeat the user question verbatim.
- Do NOT invent sensors, DTC codes, or measurements not present in context.
- Do not promote the manuals brand or name; focus on the content.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    def _format_answer(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return "I am not certain based on the retrieved context."
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        cleaned = "\n".join(lines).strip()
        if not cleaned:
            return "I am not certain based on the retrieved context."
        capitalized = False
        formatted_lines: List[str] = []
        for line in lines:
            if line.strip() and not capitalized and line[0].isalpha():
                line = line[0].upper() + line[1:]
                capitalized = True
            formatted_lines.append(line)
        out = "\n".join(formatted_lines).strip()
        if "\n" not in out and not out.endswith(("?", "!", ".")):
            out += "."
        return out

    def _tokenize_for_generation(self, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
        if self.remote_llm_url or self.llm_provider == "hf_space":
            raise RuntimeError("Tokenization is not used when a remote LLM provider is configured.")
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

    def _hf_space_generate(self, prompt: str) -> str:
        target = self.hf_space_id or self.hf_space_url
        if not target:
            raise RuntimeError("LLM_PROVIDER=hf_space requires HF_SPACE_ID or HF_SPACE_URL.")
        client = _get_hf_space_client(target)
        try:
            result = client.predict(prompt=prompt, api_name=self.hf_space_api_name)
        except Exception as e:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(f"HF Space generation failed: {e}") from e

        if isinstance(result, str):
            text = result.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return str(parsed.get("answer") or text).strip()
                except json.JSONDecodeError:
                    pass
            return text
        if isinstance(result, dict):
            return str(result.get("answer") or "").strip()
        return str(result).strip()

    def generate_answer(
        self,
        query: str,
        car_context: str = "",
        priority_context: str = "",
        manual_ids: Sequence[str] | None = None,
    ) -> str:
        quick = self._quick_smalltalk(query)
        if quick is not None:
            return quick

        if _is_identity_or_meta_query(query):
            return self._identity_intro_reply()

        active_context = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and self.has_user_index:
            active_context = f"{active_context} {self.user_model}".strip()

        snap = priority_context.strip()
        wants_status = self._wants_vehicle_status(query)

        if wants_status:
            if snap:
                context_chunks = self.retrieve(
                    query,
                    config=RetrievalConfig(k_user=2, k_global=3),
                    car_context=car_context,
                    priority_context=priority_context,
                    manual_ids=manual_ids,
                )
                reply = self._vehicle_status_reply_from_priority(snap, active_context)
                if context_chunks:
                    reply += "\n\n—\n**Owner manual excerpts**\n"
                    for i, ch in enumerate(context_chunks[:3], 1):
                        excerpt = " ".join(ch.split())[:480]
                        reply += f"\n{i}. {excerpt}"
                return self._format_answer(reply)
            return self._format_answer(self._vehicle_status_missing_snapshot_message(active_context))

        context_chunks = self.retrieve(
            query,
            config=RetrievalConfig(k_user=2, k_global=3),
            car_context=car_context,
            priority_context=priority_context,
            manual_ids=manual_ids,
        )
        manual_block = (
            "No matching owner-manual excerpts were retrieved."
            if not context_chunks
            else "\n".join(context_chunks)
        )
        context = manual_block[:2500]
        if snap:
            context = (
                f"(Vehicle profile snapshot — factual vehicle state)\n{snap[:2400]}\n\n"
                f"(Owner manual excerpts)\n{context}"
            )
        if active_context:
            context = f"(Vehicle focus: {active_context[:300]})\n\n{context}"
        mode = self._answer_mode(query)
        prompt = self._build_prompt(context=context, query=query, mode=mode)

        if self.llm_provider == "hf_space":
            answer = self._hf_space_generate(prompt)
        elif self.remote_llm_url:
            answer = self._remote_generate(prompt)
        else:
            assert self.llm is not None
            assert self.tokenizer is not None
            inputs, input_len = self._tokenize_for_generation(prompt)
            max_tokens = 140 if snap else 96

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
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
        answer_lower = answer.lower()
        if (not answer or "no relevant context" in answer_lower) and snap:
            blended = self._vehicle_status_reply_from_priority(snap, active_context)
            if context_chunks:
                blended += "\n\n—\n**Owner manual excerpts**\n"
                for i, ch in enumerate(context_chunks[:2], 1):
                    blended += f"\n{i}. {' '.join(ch.split())[:400]}"
            return self._format_answer(blended)
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
