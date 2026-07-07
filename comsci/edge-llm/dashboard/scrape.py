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


def scrape_rebench():
    html = fetch("https://swe-rebench.com/")
    s = html.replace('\\"', '"')
    dec = json.JSONDecoder()
    out, seen = [], set()
    for m in re.finditer(r'"instance_type":"model"', s):
        window = s[max(0, m.start() - 2000): m.start() + 20000]
        name_m = None
        for nm in re.finditer(r'"modelName":"([^"]{2,80})"', window[: window.find('"instance_type"')]):
            name_m = nm  # last modelName before the meta block is this record
        rel = re.search(r'"release":\{"timestamp":\d+,"date":"(\d{4}-\d{2}-\d{2})"', window)
        dev = re.search(r'"developer":"([^"]+)"', window)
        rs_rel = window.find('"rangeStats":')
        if not (name_m and rel and rs_rel != -1):
            continue
        rs_abs = max(0, m.start() - 2000) + rs_rel + len('"rangeStats":')
        try:
            stats, _ = dec.raw_decode(s, rs_abs)
        except (ValueError, json.JSONDecodeError):
            continue
        # Full-range bucket spans the widest from:to window; fall back to max.
        best_key, best_span, rate = None, -1, None
        for k in stats:
            try:
                lo, hi = (int(x) for x in k.split(":"))
            except ValueError:
                continue
            if hi - lo > best_span:
                best_span, best_key = hi - lo, k
        if best_key and isinstance(stats[best_key], dict):
            rate = stats[best_key].get("resolvedRate")
        if rate is None:
            rates = [v.get("resolvedRate", 0) for v in stats.values() if isinstance(v, dict)]
            rate = max(rates) if rates else None
        if rate is None:
            continue
        score = round(rate * 100, 1) if rate <= 1 else round(rate, 1)
        name = name_m.group(1)
        if name in seen:
            continue
        seen.add(name)
        out.append(
            {
                "name": name,
                "date": rel.group(1),
                "developer": (dev.group(1) if dev else "").lower(),
                "score": score,
            }
        )
    return out


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
            live["sources"][key] = {"ok": True, "n": len(data), "secs": round(time.time() - t0, 1), "url": url}
            print(f"{key}: {len(data)} rows in {time.time() - t0:.1f}s")
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
