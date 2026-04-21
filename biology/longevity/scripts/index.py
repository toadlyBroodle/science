"""Build a searchable index over wiki/ into wiki/build/.

Outputs:
  wiki/build/index.json   — per-page metadata + wikilinks + urls
  wiki/build/tfidf.npz    — scipy sparse TF-IDF matrix
  wiki/build/tfidf_vocab.json
  wiki/build/pages.json   — page id -> row index
  wiki/build/graph.json   — adjacency list (wikilink graph)
  wiki/build/keywords.json — keyword -> list of page ids

Karpathy-style: plain files, no server, greppable + loadable in one
line of numpy.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _wiki import BUILD, load_pages, page_index  # noqa: E402


def build_graph(pages) -> dict[str, list[str]]:
    idx = page_index(pages)
    graph: dict[str, list[str]] = {}
    for p in pages:
        neighbors = []
        for target in p.wikilinks:
            # Normalize: wikilinks may be "papers/foo", "topics/bar", or
            # a relative "../README.md". We only resolve in-wiki targets.
            tgt = target.strip()
            if tgt.startswith("../"):
                continue
            tgt = tgt.removesuffix(".md")
            if tgt in idx:
                neighbors.append(tgt)
        graph[p.rel] = sorted(set(neighbors))
    return graph


def build_tfidf(pages):
    docs = [p.body for p in pages]
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs)
    return X, vec


def build_keywords(pages):
    kw: dict[str, list[str]] = {}
    for p in pages:
        topics = p.meta.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]
        for t in topics:
            t = t.strip()
            if not t:
                continue
            kw.setdefault(t, []).append(p.rel)
        # Also index explicit "topic:" field on topic pages.
        single = p.meta.get("topic")
        if isinstance(single, str) and single.strip():
            kw.setdefault(single.strip(), []).append(p.rel)
    return {k: sorted(set(v)) for k, v in sorted(kw.items())}


def page_record(p):
    return {
        "rel": p.rel,
        "kind": p.kind,
        "title": p.title,
        "year": p.meta.get("year"),
        "venue": p.meta.get("venue"),
        "url": p.meta.get("url"),
        "access": p.meta.get("access"),
        "topics": p.meta.get("topics") or [],
        "wikilinks": p.wikilinks,
        "urls": p.urls,
    }


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    pages = load_pages()
    if not pages:
        print("No wiki pages found.", file=sys.stderr)
        return 1

    records = [page_record(p) for p in pages]
    (BUILD / "index.json").write_text(json.dumps(records, indent=2))

    graph = build_graph(pages)
    (BUILD / "graph.json").write_text(json.dumps(graph, indent=2))

    keywords = build_keywords(pages)
    (BUILD / "keywords.json").write_text(json.dumps(keywords, indent=2))

    X, vec = build_tfidf(pages)
    sparse.save_npz(BUILD / "tfidf.npz", X)
    (BUILD / "tfidf_vocab.json").write_text(
        json.dumps({w: int(i) for w, i in vec.vocabulary_.items()})
    )
    (BUILD / "pages.json").write_text(
        json.dumps({p.rel: i for i, p in enumerate(pages)}, indent=2)
    )

    print(f"indexed {len(pages)} pages")
    print(f"  tf-idf:   {X.shape[0]} docs × {X.shape[1]} terms, nnz={X.nnz}")
    print(f"  graph:    {sum(len(v) for v in graph.values())} wikilink edges")
    print(f"  keywords: {len(keywords)} topics")
    print(f"output -> {BUILD.relative_to(pathlib.Path.cwd()) if BUILD.is_relative_to(pathlib.Path.cwd()) else BUILD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
