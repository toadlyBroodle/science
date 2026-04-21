"""Lint the longevity wiki.

Checks:
  1. Broken wikilinks            (target page must exist)
  2. Paper pages without a URL   (must have `url:` in front matter)
  3. Paper pages without topics  (must have a non-empty `topics:` list)
  4. Topic pages with no inbound links (orphans)
  5. Paper pages not referenced by any topic page
  6. sources.json ids vs. wiki/papers/*.md ids (must match)
  7. Dangling URLs inside front-matter keys that claim to be URLs

Exit code == number of errors. CI-friendly.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _wiki import ROOT, PAPERS, TOPICS, load_pages, page_index  # noqa: E402

URL_RE = re.compile(r"^https?://")


def main() -> int:
    pages = load_pages()
    idx = page_index(pages)
    errors: list[str] = []
    warnings: list[str] = []

    # ---------- 1. broken wikilinks ----------
    for p in pages:
        for target in p.wikilinks:
            t = target.strip().removesuffix(".md")
            if t.startswith("../"):
                # Resolve relative to wiki/.
                resolved = (p.path.parent / target).resolve()
                if not resolved.exists():
                    errors.append(f"{p.rel}: broken wikilink -> {target}")
                continue
            if t not in idx:
                errors.append(f"{p.rel}: broken wikilink -> {target}")

    # ---------- 2 & 3. paper metadata ----------
    for p in pages:
        if p.kind != "paper":
            continue
        url = p.meta.get("url")
        if not (isinstance(url, str) and URL_RE.match(url)):
            errors.append(f"{p.rel}: missing or non-URL `url:` front-matter")
        topics = p.meta.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]
        if not topics:
            errors.append(f"{p.rel}: empty `topics:` front-matter")

    # ---------- 4. orphan topic pages ----------
    inbound: dict[str, set[str]] = defaultdict(set)
    for p in pages:
        for target in p.wikilinks:
            t = target.strip().removesuffix(".md")
            if t in idx:
                inbound[t].add(p.rel)
    for p in pages:
        if p.kind == "topic" and not inbound.get(p.rel):
            if p.rel != "index":
                warnings.append(f"{p.rel}: orphan topic page (no inbound links)")

    # ---------- 5. papers not referenced anywhere ----------
    for p in pages:
        if p.kind != "paper":
            continue
        if not inbound.get(p.rel):
            warnings.append(f"{p.rel}: paper not referenced by any other page")

    # ---------- 6. sources.json ↔ papers/ ----------
    manifest_path = ROOT / "sources.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_ids = {s["id"] for s in manifest.get("sources", [])}
        paper_ids = {p.rel.split("/", 1)[1] for p in pages if p.kind == "paper"}
        only_manifest = manifest_ids - paper_ids
        only_papers = paper_ids - manifest_ids
        for sid in sorted(only_manifest):
            errors.append(f"sources.json has `{sid}` but wiki/papers/{sid}.md is missing")
        for sid in sorted(only_papers):
            errors.append(f"wiki/papers/{sid}.md has no entry in sources.json")
        # Every source must have a `license` string.
        for src in manifest.get("sources", []):
            lic = src.get("license")
            if not isinstance(lic, str) or not lic:
                errors.append(f"sources.json[{src['id']}]: missing `license`; run scripts/licenses.py apply")

    # ---------- 7. front-matter URL sanity ----------
    url_keys = {"url", "pdf_url", "alt_url", "code", "pmc", "biorxiv", "medrxiv",
                "html", "pubmed", "sciencedirect", "preprint", "huggingface"}
    for p in pages:
        for k, v in p.meta.items():
            if k in url_keys and isinstance(v, str) and not URL_RE.match(v):
                errors.append(f"{p.rel}: front-matter `{k}:` is not a URL -> {v!r}")

    # ---------- report ----------
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n{len(warnings)} warning(s), {len(errors)} error(s) across {len(pages)} page(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
