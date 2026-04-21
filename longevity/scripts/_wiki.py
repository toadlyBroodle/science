"""Shared utilities for the longevity wiki pipeline.

Keeps YAML-front-matter parsing, wikilink extraction, and path helpers
in one place so index.py / lint.py / query.py stay small.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
PAPERS = WIKI / "papers"
TOPICS = WIKI / "topics"
BUILD = WIKI / "build"

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
# Matches [[target]] and [[target|alias]].
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
# Matches [text](url).
MD_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
# Matches bare URLs in front-matter values.
URL_RE = re.compile(r"https?://\S+")


@dataclass
class Page:
    path: pathlib.Path
    rel: str            # e.g. "papers/epinflammage"
    kind: str           # "paper" | "topic" | "index"
    meta: dict
    body: str
    wikilinks: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.meta.get("title") or self.meta.get("topic") or self.rel


def _parse_yaml_scalar(value: str) -> str | list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [s.strip().strip('"').strip("'") for s in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Tiny YAML-subset front-matter parser: key: value / key: [a, b] / key: "str"."""
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    yaml_block = m.group(1)
    meta: dict = {}
    for line in yaml_block.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _parse_yaml_scalar(value)
    return meta, text[m.end():]


def extract_wikilinks(body: str) -> list[str]:
    return [m.strip() for m in WIKILINK_RE.findall(body)]


def extract_urls(body: str, meta: dict) -> list[str]:
    urls = set(MD_LINK_RE.findall(body))
    for v in meta.values():
        if isinstance(v, str):
            urls.update(URL_RE.findall(v))
    return sorted(urls)


def load_page(path: pathlib.Path) -> Page:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    rel = path.relative_to(WIKI).with_suffix("").as_posix()
    if rel.startswith("papers/"):
        kind = "paper"
    elif rel.startswith("topics/"):
        kind = "topic"
    elif rel.startswith("analysis/"):
        kind = "analysis"
    else:
        kind = "index"
    return Page(
        path=path,
        rel=rel,
        kind=kind,
        meta=meta,
        body=body,
        wikilinks=extract_wikilinks(body),
        urls=extract_urls(body, meta),
    )


def iter_pages() -> Iterable[Page]:
    for path in sorted(WIKI.rglob("*.md")):
        # Skip the build/ output directory.
        if BUILD in path.parents:
            continue
        yield load_page(path)


def load_pages() -> list[Page]:
    return list(iter_pages())


def page_index(pages: list[Page]) -> dict[str, Page]:
    return {p.rel: p for p in pages}
