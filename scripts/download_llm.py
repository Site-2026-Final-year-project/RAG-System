#!/usr/bin/env python3
"""Pre-download the Hugging Face LLM into a local cache (avoids blocking first chat request).

Usage (from repo root, venv active):
  python scripts/download_llm.py
  python scripts/download_llm.py --model Qwen/Qwen2.5-3B-Instruct

Set in .env (same path when serving):
  HF_HOME=/absolute/path/to/RAG-System/.hf_cache

Or export before uvicorn:
  export HF_HOME="$(pwd)/.hf_cache"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env so RAG_LLM_MODEL / HF_HOME can come from there
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

DEFAULT_CACHE = _ROOT / ".hf_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LLM weights from Hugging Face Hub.")
    parser.add_argument(
        "--model",
        default=os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip(),
        help="Hub repo id (default: RAG_LLM_MODEL env or Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"HF_HOME directory (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args()

    cache = args.cache_dir or Path(os.environ.get("HF_HOME", str(DEFAULT_CACHE))).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    # Longer timeouts for slow networks
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    from huggingface_hub import snapshot_download

    print(f"Model: {args.model}")
    print(f"HF_HOME: {cache}")
    path = snapshot_download(args.model)
    print("Download complete:", path)


if __name__ == "__main__":
    main()
