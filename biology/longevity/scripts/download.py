"""Download all sources in sources.json to sources/{html,pdf}/.

URL priority per source (tried in order, first that yields content wins
for each label):
  1. ``pdf_url``   — single URL or list; PDFs preferred when available
  2. ``url``       — canonical landing page
  3. ``alt_url``   — single URL or list; fallbacks

Already-cached files (by source id + label) are skipped on re-run.
PDF responses are saved under ``sources/pdf/``; HTML under
``sources/html/``. Writes ``sources/download_log.json`` with per-attempt
status.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from collections.abc import Iterable

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources.json"
HTML_DIR = ROOT / "sources" / "html"
PDF_DIR = ROOT / "sources" / "pdf"
LOG = ROOT / "sources" / "download_log.json"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch(url: str, timeout: int = 40) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"  ! fetch failed: {url} -> {e}", file=sys.stderr)
        return None


def save(path: pathlib.Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def is_pdf(resp: requests.Response) -> bool:
    ct = resp.headers.get("Content-Type", "").lower()
    return "pdf" in ct or resp.url.lower().endswith(".pdf")


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def candidate_urls(src: dict) -> Iterable[tuple[str, str]]:
    """Yield (label, url) in priority order. First successful hit for a
    given label wins; labels with the same prefix (``pdf``) are numbered
    when multiple URLs are provided."""
    pdf_urls = _as_list(src.get("pdf_url"))
    for i, u in enumerate(pdf_urls):
        label = "pdf" if len(pdf_urls) == 1 else f"pdf{i + 1}"
        yield label, u
    if "url" in src:
        yield "primary", src["url"]
    alt_urls = _as_list(src.get("alt_url"))
    for i, u in enumerate(alt_urls):
        label = "alt" if len(alt_urls) == 1 else f"alt{i + 1}"
        yield label, u


def download_one(src: dict) -> dict:
    sid = src["id"]
    record = {"id": sid, "attempts": []}
    for label, url in candidate_urls(src):
        html_path = HTML_DIR / f"{sid}__{label}.html"
        pdf_path = PDF_DIR / f"{sid}__{label}.pdf"
        if html_path.exists() or pdf_path.exists():
            record["attempts"].append({"label": label, "url": url, "status": "cached"})
            continue
        print(f"  [{sid}][{label}] -> {url}")
        resp = fetch(url)
        if resp is None:
            record["attempts"].append({"label": label, "url": url, "status": "error"})
            continue
        if is_pdf(resp):
            save(pdf_path, resp.content)
            record["attempts"].append(
                {"label": label, "url": url, "status": "ok",
                 "path": str(pdf_path.relative_to(ROOT)), "bytes": len(resp.content)}
            )
        else:
            save(html_path, resp.content)
            record["attempts"].append(
                {"label": label, "url": url, "status": "ok",
                 "path": str(html_path.relative_to(ROOT)), "bytes": len(resp.content)}
            )
        time.sleep(1.0)
    return record


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    log = []
    for src in manifest["sources"]:
        print(f"[{src['id']}] {src['title']}")
        log.append(download_one(src))

    LOG.write_text(json.dumps(log, indent=2))
    print(f"\nDownload log: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
