#!/usr/bin/env python3
"""Refresh live benchmark data for the edge-LLM dashboard.

Scrapes three sources (stdlib only, each fails independently):
  1. benchlm.ai SWE-bench Verified leaderboard  (JSON in __NEXT_DATA__)
  2. gorilla.cs.berkeley.edu BFCL data_overall.csv
  3. swe-rebench.com leaderboard                (RSC flight payload)

Writes live.js (window.DASH_LIVE) next to this script and appends one
snapshot line per run to history.jsonl. index.html works without live.js;
it just falls back to the curated seed in data.js.
"""
import csv
import io
import json
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) edge-llm-dashboard/1.0"}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def scrape_benchlm(slug="sweVerified"):
    html = fetch(f"https://benchlm.ai/benchmarks/{slug}")
    m = re.search(r'__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S)
    rows = json.loads(m.group(1))["props"]["pageProps"]["leaderboard"]
    return [
        {
            "name": r["model"],
            "creator": r.get("creator", ""),
            "sourceType": r.get("sourceType", ""),
            "score": r.get("score", r.get("displayScore")),
        }
        for r in rows
        if r.get("score") or r.get("displayScore")
    ]


def scrape_bfcl():
    text = fetch("https://gorilla.cs.berkeley.edu/data_overall.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        name = re.sub(r"<[^>]+>", "", r.get("Model", "")).strip()
        try:
            overall = float(str(r.get("Overall Acc", "")).rstrip("%"))
        except ValueError:
            continue
        entry = {"name": name, "overall": overall}
        mt = str(r.get("Multi Turn Acc", "")).rstrip("%")
        try:
            entry["multiTurn"] = float(mt)
        except ValueError:
            pass
        out.append(entry)
    return out


def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def scrape_rebench():
    """Parse the swe-rebench leaderboard into window-aware records.

    SWE-rebench draws fresh GitHub tasks every month and never backfills older
    models, so each model carries its own `taskRangeTimestamp` evaluation
    window. Its `rangeStats` buckets silently CLAMP to the intersection of the
    requested window and the tasks that model actually ran: asking a
    Feb-2026-only model for its 2025-01-01..2026-05-15 rate returns its Feb
    number unchanged. Reading the widest bucket therefore yields a headline
    score that looks global but is really scoped to whatever batch existed at
    submission time, so cross-model comparison is meaningless whenever the
    windows differ (Qwen3.5-27B ran on Feb 2026, Qwen3.6-27B on Mar-May 2026;
    the 22-point gap between them is mostly batch difficulty, not capability).

    So: emit each model's native window plus its rate on every candidate window
    FULLY CONTAINED in that native window. Two models are comparable on a
    window only if both report a score for it.
    """
    html = fetch("https://swe-rebench.com/")
    s = html.replace('\\"', '"')
    dec = json.JSONDecoder()

    recs, seen_ids = [], set()
    for m in re.finditer(r'\{"modelId":"', s):
        try:
            o, _ = dec.raw_decode(s, m.start())
        except (ValueError, json.JSONDecodeError):
            continue
        if not isinstance(o, dict) or "rangeStats" not in o:
            continue
        if o.get("meta", {}).get("instance_type") != "model":
            continue  # skip harness entries (Claude Code, Codex, Cursor, Junie)
        if not o.get("taskRangeTimestamp") or not o.get("release"):
            continue
        if o["modelId"] in seen_ids:
            continue
        seen_ids.add(o["modelId"])
        recs.append(o)

    # One record per model: the `tools` agent scaffold beats `text` where both exist.
    by_name = {}
    for o in recs:
        prev = by_name.get(o["modelName"])
        if prev is None or (o.get("agentVersion") == "tools" and prev.get("agentVersion") != "tools"):
            by_name[o["modelName"]] = o

    candidates = sorted({(o["taskRangeTimestamp"]["from"], o["taskRangeTimestamp"]["to"])
                         for o in by_name.values()})

    models, win_counts = [], {}
    for o in by_name.values():
        trt = o["taskRangeTimestamp"]
        stats = o["rangeStats"]
        own = stats.get(f"{trt['from']}:{trt['to']}")
        if not isinstance(own, dict) or own.get("resolvedRate") is None:
            continue

        scores = {}
        for lo, hi in candidates:
            if not (trt["from"] <= lo and trt["to"] >= hi):
                continue  # window not fully covered by this model's own tasks
            v = stats.get(f"{lo}:{hi}")
            if not isinstance(v, dict) or not v.get("totalTokenUsage"):
                continue  # model never actually ran on this window
            key = f"{_iso(lo)}:{_iso(hi)}"
            scores[key] = round(v["resolvedRate"], 1)
            win_counts[key] = win_counts.get(key, 0) + 1

        models.append({
            "name": o["modelName"],
            "agent": o.get("agentVersion", ""),
            "developer": o.get("meta", {}).get("developer", "").lower(),
            "date": o["release"]["date"],
            "from": _iso(trt["from"]),
            "to": _iso(trt["to"]),
            "score": round(own["resolvedRate"], 1),
            "sem": round(own.get("sem") or 0, 2),
            "scores": scores,
        })

    # Newest batch first, then widest coverage: the default view should show the
    # current frontier, not whichever stale window happens to have the most models.
    windows = sorted(
        ({"key": k, "from": k.split(":")[0], "to": k.split(":")[1], "n": n}
         for k, n in win_counts.items() if n >= 2),
        key=lambda w: (w["to"], w["n"]),
        reverse=True,
    )
    return {"windows": windows, "models": sorted(models, key=lambda m: m["date"])}


def main():
    live = {
        "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "sources": {},
    }
    for key, fn, url in [
        ("benchlm", lambda: scrape_benchlm("sweVerified"), "https://benchlm.ai/benchmarks/sweVerified"),
        ("lcb", lambda: scrape_benchlm("liveCodeBench"), "https://benchlm.ai/benchmarks/liveCodeBench"),
        ("bfclv4", lambda: scrape_benchlm("bfclV4"), "https://benchlm.ai/benchmarks/bfclV4"),
        ("bfcl", scrape_bfcl, "https://gorilla.cs.berkeley.edu/leaderboard.html"),
        ("rebench", scrape_rebench, "https://swe-rebench.com/"),
    ]:
        t0 = time.time()
        try:
            data = fn()
            live[key] = data
            n = len(data["models"]) if isinstance(data, dict) else len(data)
            live["sources"][key] = {"ok": True, "n": n, "secs": round(time.time() - t0, 1), "url": url}
            print(f"{key}: {n} rows in {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - each source fails independently
            live["sources"][key] = {"ok": False, "error": f"{type(e).__name__}: {e}", "url": url}
            print(f"{key}: FAILED {type(e).__name__}: {e}", file=sys.stderr)

    (HERE / "live.js").write_text(
        "window.DASH_LIVE = " + json.dumps(live, indent=1) + ";\n"
    )
    with open(HERE / "history.jsonl", "a") as f:
        f.write(json.dumps({"date": str(date.today()), "sources": live["sources"]}) + "\n")
    ok = sum(1 for s in live["sources"].values() if s["ok"])
    print(f"live.js written ({ok}/5 sources ok)")


if __name__ == "__main__":
    main()
