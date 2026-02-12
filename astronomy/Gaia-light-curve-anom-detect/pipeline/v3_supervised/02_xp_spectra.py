#!/usr/bin/env python3
"""Step 2: Download and calibrate Gaia XP (BP/RP) spectra.

Uses GaiaXPy to convert internally-calibrated spectral coefficients
to absolute-flux-calibrated spectra sampled at ~2nm intervals.
Results are cached to avoid re-downloading.
"""

import os
import numpy as np
import pandas as pd
import pickle
from config import CONFIG, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, '02_xp_spectra.pkl')
BATCH_SIZE = CONFIG['xp_batch_size']

print("=" * 70)
print("STEP 2: DOWNLOAD & CALIBRATE GAIA XP SPECTRA")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '01_sample.csv'))
all_ids = df['source_id'].astype(int).tolist()
print(f"  {len(all_ids)} sources to process")

# Load cache
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    print(f"  Cache: {len(cache)} spectra already downloaded")

remaining = [sid for sid in all_ids if sid not in cache]
print(f"  Remaining: {len(remaining)} to download\n")

if remaining:
    try:
        from gaiaxpy import calibrate
    except ImportError:
        print("  ERROR: GaiaXPy not installed. Run: pip install GaiaXPy")
        print("  Skipping download, using cached data only.")
        remaining = []

# Always save cache file (even if empty) so downstream steps don't crash
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

# Process in batches
n_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(remaining))
    batch_ids = remaining[start:end]

    print(f"  Batch {batch_idx + 1}/{n_batches}: {len(batch_ids)} sources "
          f"({start + 1}–{end}/{len(remaining)})...", flush=True)

    try:
        cal_spectra, sampling = calibrate(batch_ids)

        # Store each spectrum as {wavelength: array, flux: array, flux_error: array}
        for _, row in cal_spectra.iterrows():
            sid = int(row['source_id'])
            flux = np.array(row['flux'], dtype=float)
            # sampling is the wavelength array in nm
            cache[sid] = {
                'wavelength': np.array(sampling, dtype=float),
                'flux': flux,
            }

        # Incremental save
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)

        print(f"    OK — {len(batch_ids)} spectra calibrated, "
              f"cache now {len(cache)} total")

    except Exception as e:
        print(f"    ERROR: {e}")
        print(f"    Skipping batch, continuing...")
        continue

# Summary
n_with_spectra = sum(1 for sid in all_ids if sid in cache)
n_missing = len(all_ids) - n_with_spectra

print(f"\n{'='*70}")
print(f"XP SPECTRA SUMMARY")
print(f"{'='*70}")
print(f"  Total sources: {len(all_ids)}")
print(f"  With XP spectra: {n_with_spectra}")
print(f"  Missing: {n_missing}")

if cache:
    # Show wavelength range from first spectrum
    sample = next(iter(cache.values()))
    wl = sample['wavelength']
    print(f"  Wavelength range: {wl.min():.0f}–{wl.max():.0f} nm "
          f"({len(wl)} samples)")

print(f"\n  Cache saved to {CACHE_FILE}")
