#!/usr/bin/env python3
"""Save current pipeline run to a named runs/ directory.

Usage:
  python save_run.py <run_name>
  python save_run.py 02_relaxed

Saves a flat directory with key summary files matching the format of
existing runs (e.g. runs/01_blue_cv/).
"""

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, 'runs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIGS_DIR = os.path.join(BASE_DIR, '..', 'figs', 'pipeline')

if len(sys.argv) < 2:
    print("Usage: python save_run.py <run_name>")
    print("Example: python save_run.py 02_relaxed")
    sys.exit(1)

run_name = sys.argv[1]
run_dir = os.path.join(RUNS_DIR, run_name)

if os.path.exists(run_dir):
    print(f"WARNING: {run_dir} already exists, overwriting")

os.makedirs(run_dir, exist_ok=True)
print(f"Saving pipeline run to: {run_dir}")

# Files to save (flat, matching 01_blue_cv format)
data_files = [
    '04_high_priority.csv',
    '07_deep_investigated.csv',
]
fig_files = [
    'novel_lightcurves.png',
    'novel_periods.png',
    'strong_candidates.png',
]

# Copy data files
for fname in data_files:
    src = os.path.join(DATA_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(run_dir, fname))
        print(f"  Copied {fname}")
    else:
        print(f"  WARNING: {fname} not found in {DATA_DIR}")

# Copy figures
for fname in fig_files:
    src = os.path.join(FIGS_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(run_dir, fname))
        print(f"  Copied {fname}")
    else:
        print(f"  WARNING: {fname} not found in {FIGS_DIR}")

# Copy config
config_src = os.path.join(BASE_DIR, 'config.py')
if os.path.exists(config_src):
    shutil.copy2(config_src, os.path.join(run_dir, 'config.py'))
    print(f"  Copied config.py")

print(f"\nDone! Run saved to {run_dir}")
