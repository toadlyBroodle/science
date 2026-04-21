"""Convert downloaded HTML/PDF to cleaned markdown in sources/md/.

HTML -> markdown:
  trafilatura (extracts main content, strips nav/ads), fall back to
  BeautifulSoup + markdownify if trafilatura returns nothing.

PDF -> text:
  pdftotext -layout, then a light cleanup pass (collapse repeated blanks).

Each output file gets a YAML front-matter block with source metadata so it
can round-trip through the indexer/linter.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import trafilatura
from bs4 import BeautifulSoup
from markdownify import markdownify as md_html

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources.json"
HTML_DIR = ROOT / "sources" / "html"
PDF_DIR = ROOT / "sources" / "pdf"
MD_DIR = ROOT / "sources" / "md"


def yaml_front_matter(src: dict[str, Any], label: str, origin_path: pathlib.Path) -> str:
    lines = [
        "---",
        f"id: {src['id']}",
        f"title: \"{src['title'].replace(chr(34), chr(39))}\"",
        f"url: {src['url']}",
        f"access: {src.get('access', 'unknown')}",
        f"kind: {src.get('kind', 'paper')}",
        f"topics: [{', '.join(src.get('topics', []))}]",
        f"source_label: {label}",
        f"source_file: {origin_path.relative_to(ROOT)}",
        "---",
        "",
    ]
    return "\n".join(lines)


def clean_markdown(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip() + "\n"


def html_to_md(html_path: pathlib.Path) -> str:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    extracted = trafilatura.extract(raw, include_links=True, include_tables=True, favor_recall=True)
    if extracted and len(extracted) > 500:
        return extracted
    # Fallback: strip boilerplate and convert.
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return md_html(str(main), heading_style="ATX")


def pdf_to_md(pdf_path: pathlib.Path) -> str:
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return out.stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"  ! pdftotext failed on {pdf_path}: {e}", file=sys.stderr)
        return ""


def convert_source(src: dict[str, Any]) -> list[pathlib.Path]:
    sid = src["id"]
    outputs: list[pathlib.Path] = []
    # Find all downloaded files for this source.
    html_files = sorted(HTML_DIR.glob(f"{sid}__*.html"))
    pdf_files = sorted(PDF_DIR.glob(f"{sid}__*.pdf"))

    for f in pdf_files:
        label = f.stem.split("__", 1)[1]
        text = pdf_to_md(f)
        if not text.strip():
            continue
        out = MD_DIR / f"{sid}__{label}.md"
        out.write_text(yaml_front_matter(src, label, f) + clean_markdown(text), encoding="utf-8")
        outputs.append(out)

    for f in html_files:
        label = f.stem.split("__", 1)[1]
        text = html_to_md(f)
        if not text.strip():
            continue
        out = MD_DIR / f"{sid}__{label}.md"
        out.write_text(yaml_front_matter(src, label, f) + clean_markdown(text), encoding="utf-8")
        outputs.append(out)

    return outputs


def main() -> int:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    total = 0
    for src in manifest["sources"]:
        outs = convert_source(src)
        total += len(outs)
        print(f"[{src['id']}] wrote {len(outs)} markdown file(s)")
    print(f"\nTotal markdown files: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
