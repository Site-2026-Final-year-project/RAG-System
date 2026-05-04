"""
Upload local car manual PDFs to driver-garage-backend (admin education / MANUALS).

API (see driver-garage-backend):
  POST {BASE}/admin/auth/login        JSON { "email", "password" } -> { "token" }
  POST {BASE}/admin/educational-content   multipart: title, description, category=MANUALS, pdf=<file>

Requires admin credentials (never commit these):
  export ADMIN_EMAIL="..."
  export ADMIN_PASSWORD="..."

If `/admin/auth/login` returns 429 (rate limit), skip login and pass a token from a fresh curl:
  export ADMIN_TOKEN="eyJ..."   # optional; refreshes on 401 via login retry

Optional:
  EXPRESS_BASE_URL / BASE_URL   default https://driver-garage-backend.onrender.com
  PDF_ROOT                       default <repo>/data/raw/car_manual_pdfs

Usage:
  cd RAG-System && source venv/bin/activate  # needs requests
  export ADMIN_EMAIL=... ADMIN_PASSWORD=...
  python scripts/upload_car_manuals_to_render.py

  python scripts/upload_car_manuals_to_render.py --dry-run
  python scripts/upload_car_manuals_to_render.py --limit 5 --delay 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    raise SystemExit(1)

# Backend rejects PDFs larger than this (multer limit).
MAX_BYTES = 25 * 1024 * 1024


def default_base_url() -> str:
    return (
        os.environ.get("EXPRESS_BASE_URL")
        or os.environ.get("BASE_URL")
        or "https://driver-garage-backend.onrender.com"
    ).rstrip("/")


def title_from_pdf_path(pdf: Path, root: Path) -> str:
    rel = pdf.relative_to(root)
    dir_parts = list(rel.parent.parts)
    make_folder = ""
    if dir_parts:
        make_folder = dir_parts[0].replace("_", " ").replace("-", " ").strip()
    stem = pdf.stem.replace("_", "-")
    tokens = []
    for p in stem.split("-"):
        if not p:
            continue
        if p.isdigit() and len(p) == 4:
            tokens.append(p)
        elif len(p) <= 3 and p.isalpha() and p.upper() == p:
            tokens.append(p.upper())
        else:
            tokens.append(p.capitalize())
    from_stem = " ".join(tokens) if tokens else stem
    if make_folder and make_folder.lower() not in ("unknown",):
        mf = make_folder.capitalize()
        if from_stem.lower().startswith(mf.lower()):
            return from_stem
        return f"{mf} {from_stem}"
    return from_stem or rel.as_posix().replace("/", " — ")


def description_from_pdf_path(pdf: Path, root: Path) -> str:
    rel = pdf.relative_to(root).as_posix()
    return f"Owner manual PDF ({rel}). Uploaded in bulk for education center MANUALS."


def login(session: requests.Session, base: str, email: str, password: str) -> tuple[int, str]:
    """Returns (status_code, token_or_error_body)."""
    r = session.post(
        f"{base}/admin/auth/login",
        json={"email": email, "password": password},
        timeout=120,
    )
    if r.status_code != 200:
        return r.status_code, r.text[:500]
    data = r.json()
    token = data.get("token")
    if not token:
        return 500, f"Login response missing token: {data!r}"
    return 200, token


def login_with_retry(session: requests.Session, base: str, email: str, password: str) -> str:
    """Login with backoff on HTTP 429 (express-rate-limit on admin login)."""
    delay = 65.0
    max_attempts = 12
    last_body = ""
    for attempt in range(1, max_attempts + 1):
        status, payload = login(session, base, email, password)
        if status == 200:
            return payload
        last_body = payload
        if status == 429 and attempt < max_attempts:
            print(
                f"Login rate limited (429), waiting {delay:.0f}s before retry {attempt}/{max_attempts}...",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay + 45.0, 920.0)
            continue
        raise RuntimeError(f"Login failed HTTP {status}: {payload}")
    raise RuntimeError(f"Login failed after retries: {last_body}")


def upload_one(
    session: requests.Session,
    base: str,
    token: str,
    pdf: Path,
    root: Path,
) -> tuple[int, str]:
    title = title_from_pdf_path(pdf, root)
    description = description_from_pdf_path(pdf, root)
    headers = {"Authorization": f"Bearer {token}"}
    with pdf.open("rb") as f:
        files = {"pdf": (pdf.name, f, "application/pdf")}
        data = {
            "title": title,
            "description": description,
            "category": "MANUALS",
        }
        r = session.post(
            f"{base}/admin/educational-content",
            headers=headers,
            data=data,
            files=files,
            timeout=300,
        )
    return r.status_code, r.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload car_manual_pdfs to Render Express admin API.")
    parser.add_argument("--base-url", default=default_base_url(), help="Express base URL (no trailing slash)")
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=_ROOT / "data" / "raw" / "car_manual_pdfs",
        help="Directory containing make/year/*.pdf",
    )
    parser.add_argument("--dry-run", action="store_true", help="List PDFs only, no network uploads")
    parser.add_argument("--limit", type=int, default=0, help="Max files (0 = all)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between uploads")
    parser.add_argument("--fail-log", type=Path, default=None, help="Append JSON lines of failures")
    args = parser.parse_args()

    root: Path = args.pdf_root.resolve()
    if not root.is_dir():
        print(f"PDF root not found: {root}", file=sys.stderr)
        raise SystemExit(2)

    pdfs = sorted(root.rglob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    oversized = [p for p in pdfs if p.stat().st_size > MAX_BYTES]
    if oversized:
        print(f"Warning: {len(oversized)} file(s) exceed {MAX_BYTES // (1024*1024)} MiB and will be skipped.", file=sys.stderr)

    email = os.environ.get("ADMIN_EMAIL", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    env_token = os.environ.get("ADMIN_TOKEN", "").strip()
    base = args.base_url.rstrip("/")

    if args.dry_run:
        print(f"base_url={base} pdf_root={root} count={len(pdfs)}")
        for p in pdfs[:20]:
            sz = p.stat().st_size
            skip = " SKIP>25MiB" if sz > MAX_BYTES else ""
            print(f"  {title_from_pdf_path(p, root)!r}{skip}")
        if len(pdfs) > 20:
            print(f"  ... and {len(pdfs) - 20} more")
        return

    has_password_login = bool(email and password)
    if not has_password_login and not env_token:
        print(
            "Set ADMIN_EMAIL + ADMIN_PASSWORD, or set ADMIN_TOKEN (from curl) if login is rate limited.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    session = requests.Session()
    if env_token:
        token = env_token
        print("Using ADMIN_TOKEN (skipping login).", file=sys.stderr, flush=True)
    else:
        token = login_with_retry(session, base, email, password)

    def refresh_token() -> None:
        nonlocal token
        if not has_password_login:
            raise RuntimeError(
                "JWT expired (401) and ADMIN_EMAIL/ADMIN_PASSWORD not set; renew ADMIN_TOKEN or set credentials."
            )
        token = login_with_retry(session, base, email, password)

    ok = 0
    failed: list[dict] = []
    for i, pdf in enumerate(pdfs):
        if pdf.stat().st_size > MAX_BYTES:
            failed.append({"path": str(pdf), "error": "file too large (>25 MiB)"})
            continue
        try:
            status, body = upload_one(session, base, token, pdf, root)
            if status == 401:
                refresh_token()
                status, body = upload_one(session, base, token, pdf, root)
            if status in (200, 201):
                ok += 1
                print(f"[{i+1}/{len(pdfs)}] OK {pdf.name}", flush=True)
            else:
                err = {"path": str(pdf), "status": status, "body": body[:2000]}
                failed.append(err)
                print(f"[{i+1}/{len(pdfs)}] FAIL {pdf.name} HTTP {status}", file=sys.stderr, flush=True)
        except requests.RequestException as e:
            failed.append({"path": str(pdf), "error": str(e)})
            print(f"[{i+1}/{len(pdfs)}] FAIL {pdf.name} {e}", file=sys.stderr, flush=True)

        if args.delay > 0 and i < len(pdfs) - 1:
            time.sleep(args.delay)

    print(f"Done. uploaded={ok} failed={len(failed)} total={len(pdfs)}", flush=True)
    if args.fail_log and failed:
        with args.fail_log.open("a", encoding="utf-8") as fl:
            for row in failed:
                fl.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote failures to {args.fail_log}")


if __name__ == "__main__":
    main()
