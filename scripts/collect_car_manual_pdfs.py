"""
Discover and download car manual PDFs from public websites.

Primary sources:
- Nissan TechInfo owner manual catalog pages (official, broad coverage)
- Carmans owner manual posts (multi-brand direct PDF links)

Fallback source:
- Search-based discovery (DuckDuckGo HTML), useful where not blocked.

Examples:
  python scripts/collect_car_manual_pdfs.py --source all --limit 300 --max-per-make 30
  python scripts/collect_car_manual_pdfs.py --source search --limit 100 --crawl-result-pages
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse, unquote

import requests

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

SEARCH_ENDPOINT = "https://duckduckgo.com/html/?q={query}"
NISSAN_ROOT = "https://www.nissan-techinfo.com"
NISSAN_INDEX_DEPT = 201
CARMANS_ROOT = "https://www.carmans.net"
CARMANS_CATEGORY = f"{CARMANS_ROOT}/category/all-owners-manuals/"
BYD_OWNERS_URL = "https://www.byd.com/om/service-maintenance/owners-manual"
SUZUKI_OWNERS_URL = "https://www.suzukimanuals.com.au/owners-manuals/"
DEFAULT_MAKES = [
    "toyota",
    "honda",
    "ford",
    "chevrolet",
    "nissan",
    "hyundai",
    "kia",
    "bmw",
    "mercedes",
    "audi",
    "mazda",
    "subaru",
    "lexus",
    "volkswagen",
    "volvo",
    "mitsubishi",
    "gmc",
    "jeep",
    "ram",
    "dodge",
    "chrysler",
    "cadillac",
    "buick",
    "lincoln",
    "acura",
    "infiniti",
    "porsche",
    "tesla",
]
DEFAULT_YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
ETHIOPIA_COMMON_MAKES = [
    "toyota",
    "suzuki",
    "hyundai",
    "kia",
    "nissan",
    "honda",
    "byd",
]
ETHIOPIA_COMMON_MODELS = [
    "corolla",
    "vitz",
    "rav4",
    "yaris",
    "hilux",
    "land cruiser",
    "prius",
    "suzuki swift",
    "suzuki alto",
    "hyundai i10",
    "kia picanto",
    "byd dolphin",
    "byd atto 3",
]


@dataclass
class CandidatePdf:
    pdf_url: str
    source_url: str
    query: str
    make: str
    year: str


def infer_make_year_from_url(url: str) -> tuple[str, str]:
    lowered = unquote(url.lower())
    years = re.findall(r"(19\d{2}|20\d{2})", lowered)
    year = years[0] if years else "unknown"
    makes = [
        "nissan",
        "honda",
        "toyota",
        "ford",
        "mazda",
        "kia",
        "hyundai",
        "subaru",
        "bmw",
        "mercedes",
        "audi",
        "volkswagen",
        "lexus",
    ]
    for make in makes:
        if make in lowered:
            return make, year
    return "unknown", year


def infer_make_year_from_text(text: str) -> tuple[str, str]:
    lowered = unquote(text.lower())
    years = re.findall(r"(19\d{2}|20\d{2})", lowered)
    year = years[0] if years else "unknown"
    for make in DEFAULT_MAKES:
        if make in lowered:
            return make, year
    return "unknown", year


def extract_urls_from_duckduckgo(html_text: str) -> list[str]:
    # DuckDuckGo HTML uses links like /l/?uddg=<encoded-target>.
    matches = re.findall(r'href="([^"]+)"', html_text)
    out: list[str] = []
    for href in matches:
        href = html.unescape(href)
        if href.startswith("/l/?"):
            parsed = urlparse(href)
            target = parse_qs(parsed.query).get("uddg", [])
            if target:
                out.append(target[0])
        elif href.startswith("http://") or href.startswith("https://"):
            out.append(href)
    # Keep order while deduplicating.
    deduped = list(dict.fromkeys(out))
    return deduped


def extract_pdf_links_from_page(base_url: str, page_html: str) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', page_html, flags=re.IGNORECASE)
    out: list[str] = []
    for href in hrefs:
        href = html.unescape(href).strip()
        if not href:
            continue
        full = urljoin(base_url, href)
        if ".pdf" in full.lower():
            out.append(full.split("#")[0])
    return list(dict.fromkeys(out))


def looks_like_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or ".pdf?" in lowered or "/pdf/" in lowered


def make_queries(makes: list[str], years: list[str], models: list[str]) -> list[tuple[str, str, str]]:
    queries: list[tuple[str, str, str]] = []
    for make in makes:
        for year in years:
            q = f"{make} {year} owner manual filetype:pdf"
            queries.append((q, make, year))
            for model in models:
                q_model = f"{make} {model} {year} owner manual filetype:pdf"
                queries.append((q_model, make, year))
    return queries


def discover_candidates(
    session: requests.Session,
    query: str,
    make: str,
    year: str,
    max_results_per_query: int,
    crawl_result_pages: bool,
    sleep_s: float,
) -> list[CandidatePdf]:
    endpoint = SEARCH_ENDPOINT.format(query=quote_plus(query))
    resp = session.get(endpoint, timeout=30)
    resp.raise_for_status()

    result_urls = extract_urls_from_duckduckgo(resp.text)[:max_results_per_query]
    candidates: list[CandidatePdf] = []

    for url in result_urls:
        if looks_like_pdf_url(url):
            candidates.append(CandidatePdf(url, url, query, make, year))
            continue

        if not crawl_result_pages:
            continue

        try:
            page = session.get(url, timeout=30)
            if page.status_code >= 400:
                continue
            for pdf_url in extract_pdf_links_from_page(url, page.text):
                candidates.append(CandidatePdf(pdf_url, url, query, make, year))
            time.sleep(sleep_s)
        except requests.RequestException:
            continue

    deduped: list[CandidatePdf] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.pdf_url
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def extract_nissan_dept_ids(index_html: str) -> list[int]:
    ids = re.findall(r"deptog\.aspx\?dept_id=(\d+)", index_html, flags=re.IGNORECASE)
    unique = list(dict.fromkeys(int(x) for x in ids))
    return unique


def discover_nissan_candidates(
    session: requests.Session,
    max_departments: int,
    sleep_s: float,
) -> list[CandidatePdf]:
    index_url = f"{NISSAN_ROOT}/deptog.aspx?dept_id={NISSAN_INDEX_DEPT}"
    index_resp = session.get(index_url, timeout=30)
    index_resp.raise_for_status()
    dept_ids = extract_nissan_dept_ids(index_resp.text)
    if not dept_ids:
        return []

    # Crawl the model departments first; include the index as well.
    crawl_ids = [NISSAN_INDEX_DEPT] + dept_ids[:max_departments]
    all_candidates: list[CandidatePdf] = []

    for i, dept_id in enumerate(crawl_ids, start=1):
        url = f"{NISSAN_ROOT}/deptog.aspx?dept_id={dept_id}"
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code >= 400:
                continue
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', resp.text, flags=re.IGNORECASE)
            for href in hrefs:
                full = urljoin(url, html.unescape(href))
                # Direct files are usually under /refgh0v/og/...pdf
                if full.lower().endswith(".pdf") and "/refgh0v/og/" in full.lower():
                    make, year = infer_make_year_from_url(full)
                    all_candidates.append(
                        CandidatePdf(
                            pdf_url=full,
                            source_url=url,
                            query=f"nissan dept_id={dept_id}",
                            make=make,
                            year=year,
                        )
                    )
            print(f"[nissan] crawled dept {dept_id} ({i}/{len(crawl_ids)})")
            time.sleep(sleep_s)
        except requests.RequestException:
            continue

    # Keep order while deduplicating by URL.
    deduped: list[CandidatePdf] = []
    seen: set[str] = set()
    for c in all_candidates:
        if c.pdf_url in seen:
            continue
        seen.add(c.pdf_url)
        deduped.append(c)
    return deduped


def discover_carmans_candidates(
    session: requests.Session,
    max_pages: int,
    max_posts_per_page: int,
    sleep_s: float,
) -> list[CandidatePdf]:
    post_urls: list[str] = []
    for page in range(1, max_pages + 1):
        page_url = CARMANS_CATEGORY if page == 1 else f"{CARMANS_CATEGORY}page/{page}/"
        try:
            resp = session.get(page_url, timeout=30)
            if resp.status_code >= 400:
                continue
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', resp.text, flags=re.IGNORECASE)
            discovered_posts: list[str] = []
            for href in hrefs:
                full = urljoin(page_url, html.unescape(href))
                if re.search(r"https://www\.carmans\.net/\d{4}-[^/]+/?$", full):
                    discovered_posts.append(full.rstrip("/") + "/")
            # Keep page-local order and cap.
            discovered_posts = list(dict.fromkeys(discovered_posts))[:max_posts_per_page]
            post_urls.extend(discovered_posts)
            print(f"[carmans] crawled category page {page} -> posts={len(discovered_posts)}")
            time.sleep(sleep_s)
        except requests.RequestException:
            continue

    post_urls = list(dict.fromkeys(post_urls))
    candidates: list[CandidatePdf] = []
    for i, post_url in enumerate(post_urls, start=1):
        try:
            post = session.get(post_url, timeout=30)
            if post.status_code >= 400:
                continue
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', post.text, flags=re.IGNORECASE)
            for href in hrefs:
                full = urljoin(post_url, html.unescape(href))
                if "/wp-content/uploads/pdf/" in full.lower() and full.lower().endswith(".pdf"):
                    make, year = infer_make_year_from_text(post_url)
                    if make == "unknown":
                        # fallback: sometimes slug omits make
                        make, year2 = infer_make_year_from_text(full)
                        if year == "unknown":
                            year = year2
                    candidates.append(
                        CandidatePdf(
                            pdf_url=full,
                            source_url=post_url,
                            query="carmans category crawl",
                            make=make,
                            year=year,
                        )
                    )
            if i % 25 == 0:
                print(f"[carmans] crawled post {i}/{len(post_urls)}")
            time.sleep(sleep_s)
        except requests.RequestException:
            continue

    deduped: list[CandidatePdf] = []
    seen: set[str] = set()
    for c in candidates:
        if c.pdf_url in seen:
            continue
        seen.add(c.pdf_url)
        deduped.append(c)
    return deduped


def discover_byd_candidates(session: requests.Session, sleep_s: float) -> list[CandidatePdf]:
    """
    Parse BYD Middle East/Africa owner-manual page.
    The page embeds a JS variable named `serviceData` with pdfPath entries.
    """
    try:
        resp = session.get(BYD_OWNERS_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    html_text = resp.text

    # Capture each manual-like object from the embedded `serviceData` script.
    # Example fields on page:
    # 'pdfTitle': `...`, 'pdfPath': `/material/...pdf`, 'vehicleType': `BYD ATTO 3`
    object_pattern = re.compile(
        r"\{.*?'pdfTitle'\s*:\s*`(?P<title>.*?)`.*?'pdfPath'\s*:\s*`(?P<path>.*?)`.*?'vehicleType'\s*:\s*`(?P<vehicle>.*?)`.*?\}",
        flags=re.S,
    )

    candidates: list[CandidatePdf] = []
    for m in object_pattern.finditer(html_text):
        title = (m.group("title") or "").strip()
        path = (m.group("path") or "").strip()
        vehicle = (m.group("vehicle") or "BYD").strip()
        if not path:
            continue
        pdf_url = urljoin(BYD_OWNERS_URL, path)
        make, year = infer_make_year_from_text(f"{title} {vehicle} {path}")
        if make == "unknown":
            make = "byd"
        candidates.append(
            CandidatePdf(
                pdf_url=pdf_url,
                source_url=BYD_OWNERS_URL,
                query=f"byd owners manual: {vehicle}",
                make=make,
                year=year,
            )
        )
        time.sleep(sleep_s)

    deduped: list[CandidatePdf] = []
    seen: set[str] = set()
    for c in candidates:
        if c.pdf_url in seen:
            continue
        seen.add(c.pdf_url)
        deduped.append(c)
    return deduped


def discover_suzuki_candidates(session: requests.Session, sleep_s: float) -> list[CandidatePdf]:
    """Parse Suzuki Australia owners-manuals page for direct PDF links."""
    try:
        resp = session.get(SUZUKI_OWNERS_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', resp.text, flags=re.IGNORECASE)
    candidates: list[CandidatePdf] = []
    for href in hrefs:
        full = urljoin(SUZUKI_OWNERS_URL, html.unescape(href).strip())
        if not full.lower().endswith(".pdf"):
            continue
        if "/assets/owners-manuals/" not in full.lower():
            continue
        make, year = infer_make_year_from_text(full)
        if make == "unknown":
            make = "suzuki"
        candidates.append(
            CandidatePdf(
                pdf_url=full,
                source_url=SUZUKI_OWNERS_URL,
                query="suzuki owners manuals australia",
                make=make,
                year=year,
            )
        )
        time.sleep(sleep_s)

    deduped: list[CandidatePdf] = []
    seen: set[str] = set()
    for c in candidates:
        if c.pdf_url in seen:
            continue
        seen.add(c.pdf_url)
        deduped.append(c)
    return deduped


def safe_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name or "manual.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    # Keep filenames portable.
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned or "manual.pdf"


def write_metadata_row(metadata_path: Path, row: dict[str, str]) -> None:
    exists = metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256",
                "file_path",
                "pdf_url",
                "source_url",
                "query",
                "make",
                "year",
                "size_bytes",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def download_pdf(
    session: requests.Session,
    candidate: CandidatePdf,
    output_dir: Path,
    metadata_path: Path,
    max_size_mb: int,
    seen_hashes: set[str],
) -> bool:
    try:
        # Separate connect/read timeout avoids hanging too long on slow hosts.
        resp = session.get(candidate.pdf_url, timeout=(20, 30), stream=True)
        if resp.status_code >= 400:
            return False

        content_type = resp.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not looks_like_pdf_url(candidate.pdf_url):
            return False

        max_bytes = max_size_mb * 1024 * 1024
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                return False
            chunks.append(chunk)
        blob = b"".join(chunks)
        if not blob.startswith(b"%PDF"):
            return False

        sha = hashlib.sha256(blob).hexdigest()
        if sha in seen_hashes:
            return False

        make_dir = output_dir / candidate.make / candidate.year
        make_dir.mkdir(parents=True, exist_ok=True)
        file_name = safe_filename_from_url(candidate.pdf_url)
        dst = make_dir / file_name
        if dst.exists():
            dst = make_dir / f"{dst.stem}_{sha[:8]}.pdf"
        dst.write_bytes(blob)
        seen_hashes.add(sha)

        write_metadata_row(
            metadata_path,
            {
                "sha256": sha,
                "file_path": str(dst),
                "pdf_url": candidate.pdf_url,
                "source_url": candidate.source_url,
                "query": candidate.query,
                "make": candidate.make,
                "year": candidate.year,
                "size_bytes": str(size),
            },
        )
        return True
    except requests.RequestException:
        return False


def load_existing_hashes(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["sha256"] for row in reader if row.get("sha256")}


def load_existing_make_counts(metadata_path: Path) -> dict[str, int]:
    if not metadata_path.exists():
        return {}
    counts: dict[str, int] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            make = (row.get("make") or "unknown").strip().lower() or "unknown"
            counts[make] = counts.get(make, 0) + 1
    return counts


def load_existing_pdf_urls(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    urls: set[str] = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = (row.get("pdf_url") or "").strip()
            if u:
                urls.add(u)
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect car owner manual PDFs from websites.")
    parser.add_argument(
        "--source",
        choices=["nissan", "carmans", "byd", "suzuki", "search", "all"],
        default="all",
        help="Discovery source to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/car_manual_pdfs"),
        help="Directory to store downloaded PDFs.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("data/raw/car_manual_pdfs/metadata.csv"),
        help="CSV file tracking downloaded documents.",
    )
    parser.add_argument(
        "--makes",
        type=str,
        default=",".join(DEFAULT_MAKES),
        help="Comma-separated car makes to search.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=",".join(DEFAULT_YEARS),
        help="Comma-separated years to search.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model keywords to prioritize (e.g. vitz,rav4,corolla).",
    )
    parser.add_argument("--limit", type=int, default=250, help="Maximum PDFs to download in this run.")
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=10,
        help="Max search results to inspect per query.",
    )
    parser.add_argument(
        "--crawl-result-pages",
        action="store_true",
        help="Also open non-PDF search results and extract linked PDFs.",
    )
    parser.add_argument("--max-size-mb", type=int, default=80, help="Skip PDFs larger than this size.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between web requests for polite crawling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used when shuffling queries.",
    )
    parser.add_argument(
        "--nissan-max-departments",
        type=int,
        default=60,
        help="Maximum Nissan department pages to crawl.",
    )
    parser.add_argument(
        "--carmans-max-pages",
        type=int,
        default=20,
        help="Maximum Carmans category pages to crawl.",
    )
    parser.add_argument(
        "--carmans-max-posts-per-page",
        type=int,
        default=50,
        help="Maximum Carmans post URLs to inspect per category page.",
    )
    parser.add_argument(
        "--max-per-make",
        type=int,
        default=0,
        help="If > 0, cap total manuals per make (including existing metadata).",
    )
    parser.add_argument(
        "--skip-makes",
        type=str,
        default="",
        help="Comma-separated makes to skip during download (e.g. nissan,gmc).",
    )
    parser.add_argument(
        "--ethiopia-common",
        action="store_true",
        help="Add common Ethiopia market makes/models to search queries.",
    )
    parser.add_argument(
        "--target-keywords",
        type=str,
        default="",
        help="Comma-separated keywords. If set, only keep candidates whose URL/source/query contains one keyword.",
    )
    return parser.parse_args()


def cleaned_csv_values(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen_hashes = load_existing_hashes(args.metadata_file)
    existing_make_counts = load_existing_make_counts(args.metadata_file)
    seen_pdf_urls: set[str] = load_existing_pdf_urls(args.metadata_file)
    downloaded = 0
    attempted = 0
    skip_makes = set(cleaned_csv_values(args.skip_makes))
    target_keywords = set(cleaned_csv_values(args.target_keywords))
    candidates: list[CandidatePdf] = []
    if args.source in {"nissan", "all"}:
        print(f"[discover] crawling nissan-techinfo departments up to {args.nissan_max_departments}")
        candidates.extend(
            discover_nissan_candidates(
                session=session,
                max_departments=args.nissan_max_departments,
                sleep_s=args.sleep,
            )
        )
    if args.source in {"carmans", "all"}:
        print(f"[discover] crawling carmans category pages up to {args.carmans_max_pages}")
        candidates.extend(
            discover_carmans_candidates(
                session=session,
                max_pages=args.carmans_max_pages,
                max_posts_per_page=args.carmans_max_posts_per_page,
                sleep_s=args.sleep,
            )
        )
    if args.source in {"byd", "all"}:
        print("[discover] crawling BYD owners-manual page")
        byd_candidates = discover_byd_candidates(session=session, sleep_s=args.sleep)
        print(f"[byd] discovered {len(byd_candidates)} candidates")
        candidates.extend(byd_candidates)
    if args.source in {"suzuki", "all"}:
        print("[discover] crawling Suzuki owners-manual page")
        suzuki_candidates = discover_suzuki_candidates(session=session, sleep_s=args.sleep)
        print(f"[suzuki] discovered {len(suzuki_candidates)} candidates")
        candidates.extend(suzuki_candidates)

    if args.source in {"search", "all"}:
        makes = cleaned_csv_values(args.makes)
        years = cleaned_csv_values(args.years)
        models = cleaned_csv_values(args.models)
        if args.ethiopia_common:
            makes = list(dict.fromkeys(makes + ETHIOPIA_COMMON_MAKES))
            models = list(dict.fromkeys(models + ETHIOPIA_COMMON_MODELS))
        queries = make_queries(makes, years, models)
        random.Random(args.random_seed).shuffle(queries)
        print(f"[discover] search queries={len(queries)}")
        for query, make, year in queries:
            try:
                discovered = discover_candidates(
                    session=session,
                    query=query,
                    make=make,
                    year=year,
                    max_results_per_query=args.results_per_query,
                    crawl_result_pages=args.crawl_result_pages,
                    sleep_s=args.sleep,
                )
                candidates.extend(discovered)
                print(f"[search] {query} -> {len(discovered)} candidates")
            except requests.RequestException:
                continue
            if len(candidates) >= args.limit * 3:
                break

    # Deduplicate candidates globally.
    deduped_candidates: list[CandidatePdf] = []
    seen_c: set[str] = set()
    for c in candidates:
        if c.pdf_url in seen_c:
            continue
        if target_keywords:
            hay = f"{c.pdf_url} {c.source_url} {c.query}".lower()
            if not any(k in hay for k in target_keywords):
                continue
        seen_c.add(c.pdf_url)
        deduped_candidates.append(c)

    print(
        json.dumps(
            {
                "candidates": len(deduped_candidates),
                "limit": args.limit,
                "existing_make_counts": existing_make_counts,
            },
            indent=2,
        )
    )
    per_make_downloaded: dict[str, int] = {}
    for candidate in deduped_candidates:
        if downloaded >= args.limit:
            break
        if candidate.pdf_url in seen_pdf_urls:
            continue
        make_key = (candidate.make or "unknown").lower()
        if make_key in skip_makes:
            continue
        total_for_make = existing_make_counts.get(make_key, 0) + per_make_downloaded.get(make_key, 0)
        if args.max_per_make > 0 and total_for_make >= args.max_per_make:
            continue
        seen_pdf_urls.add(candidate.pdf_url)
        attempted += 1
        if download_pdf(
            session=session,
            candidate=candidate,
            output_dir=args.output_dir,
            metadata_path=args.metadata_file,
            max_size_mb=args.max_size_mb,
            seen_hashes=seen_hashes,
        ):
            downloaded += 1
            per_make_downloaded[make_key] = per_make_downloaded.get(make_key, 0) + 1
            print(f"[{downloaded}] {candidate.make} {candidate.year} -> {candidate.pdf_url}")
        time.sleep(args.sleep)

    print(
        json.dumps(
            {
                "downloaded": downloaded,
                "attempted_candidates": attempted,
                "output_dir": str(args.output_dir),
                "metadata_file": str(args.metadata_file),
                "downloaded_by_make": per_make_downloaded,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    # Allow OpenMP-heavy libs in this repo to coexist if imported elsewhere.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
