#!/usr/bin/env bash
# Refresh live benchmark data, then open the dashboard.
cd "$(dirname "$0")"
python3 scrape.py || echo "scrape failed; opening with last data" >&2
f="$(pwd)/index.html"
if command -v wslview >/dev/null 2>&1; then wslview "$f"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$f"
else echo "open $f manually"; fi
