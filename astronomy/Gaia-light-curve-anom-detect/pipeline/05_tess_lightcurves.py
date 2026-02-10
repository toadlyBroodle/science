#!/usr/bin/env python3
"""Step 5: Extract TESS light curves for high-priority candidates."""

import os, sys, time, signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Tesscut
from config import CONFIG, DATA_DIR, PLOT_DIR
from lc_utils import collapse_gaps, plot_lc
import pickle

REFETCH = '--refetch' in sys.argv

os.makedirs(PLOT_DIR, exist_ok=True)


def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


print("=" * 70)
print("STEP 5: TESS LIGHT CURVE EXTRACTION")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '04_high_priority.csv'))
candidates = df[df['tic_id'].notna()].copy()
print(f"  {len(candidates)} high-priority candidates with TIC matches")

# --- Deduplicate by sky proximity (TESS 21"/px, 3x3 aperture ~ 63") ---
# Group candidates within 63" so we only fetch one cutout per TESS patch
coords = SkyCoord(ra=candidates['ra'].values*u.deg, dec=candidates['dec'].values*u.deg)
DEDUP_RADIUS = 63  # arcsec — matches 3x3 TESS aperture

keep_mask = np.ones(len(candidates), dtype=bool)
indices = list(candidates.index)
groups = {}  # representative index -> list of grouped indices

for j in range(len(indices)):
    if not keep_mask[j]:
        continue
    rep = indices[j]
    groups[rep] = [rep]
    for k in range(j + 1, len(indices)):
        if not keep_mask[k]:
            continue
        sep = coords[j].separation(coords[k]).arcsec
        if sep < DEDUP_RADIUS:
            keep_mask[k] = False
            groups[rep].append(indices[k])

candidates_dedup = candidates.iloc[keep_mask].copy()
n_removed = len(candidates) - len(candidates_dedup)
if n_removed > 0:
    print(f"  Deduplicated: {n_removed} sources within {DEDUP_RADIUS}\" of another (TESS aperture overlap)")
    for rep, grp in groups.items():
        if len(grp) > 1:
            tics = [int(candidates.loc[g, 'tic_id']) for g in grp]
            print(f"    Group: TIC {', '.join(str(t) for t in tics)} -> keeping TIC {tics[0]}")
candidates = candidates_dedup
n = len(candidates)
print(f"  Extracting {n} unique light curves\n")

# --- Load cached light curves from previous runs ---
lc_path = os.path.join(DATA_DIR, '05_lc_data.pkl')
if REFETCH:
    lc_data = {}
    print(f"  --refetch: ignoring cached data, re-downloading all")
elif os.path.exists(lc_path):
    with open(lc_path, 'rb') as f:
        lc_data = pickle.load(f)
    print(f"  Loaded {len(lc_data)} cached light curves from previous run")
else:
    lc_data = {}

n_cols = 2
n_rows = max((n + 1) // 2, 1)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

for i, (_, row) in enumerate(candidates.iterrows()):
    tic_id = int(row['tic_id'])
    gaia_id = int(row['source_id'])
    ax = axes_flat[i] if i < len(axes_flat) else None

    print(f"  [{i+1}/{n}] TIC {tic_id} (Gaia DR3 {gaia_id})", flush=True)

    # Check cache
    if tic_id in lc_data:
        cached = lc_data[tic_id]
        t, flux_norm = cached['time'], cached['flux']
        sector_str = ','.join(str(s) for s in cached.get('sectors', [cached.get('sector', '?')]))
        print(f"    Cached: {len(t)} pts from S{sector_str}")
        if ax:
            vsx_type = row.get('vsx_type', '')
            plot_lc(ax, t, flux_norm, f"TIC {tic_id} | {vsx_type} | S{sector_str}", sector_str)
        continue

    coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')

    # Discover sectors
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        avail = Tesscut.get_sectors(coordinates=coord)
        signal.alarm(0)
        sectors = list(avail['sector']) if avail and len(avail) > 0 else []
    except:
        signal.alarm(0)
        sectors = []

    if not sectors:
        print(f"    No TESS sectors found")
        if ax: ax.text(0.5, 0.5, f'TIC {tic_id}\nNo sectors', ha='center', va='center', transform=ax.transAxes)
        continue

    print(f"    {len(sectors)} sector(s): {sectors}")

    # Download and stitch all sectors
    all_t, all_flux = [], []
    sectors_used = []

    for sector in sectors:
        cutout = None
        for attempt in range(3):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                cutout = Tesscut.get_cutouts(coordinates=coord, size=5, sector=sector)
                signal.alarm(0)
                break
            except TimeoutError:
                signal.alarm(0)
                print(f"    Timeout on S{sector}, attempt {attempt+1}")
                if attempt < 2: time.sleep(5)
            except Exception as e:
                signal.alarm(0)
                err = str(e).lower()
                if any(k in err for k in ['disconnect', 'connection', 'timeout', 'reset']):
                    if attempt < 2:
                        print(f"    Network error S{sector}, retrying...")
                        time.sleep(5 * (attempt + 1))
                        continue
                print(f"    S{sector} error: {str(e)[:60]}")
                break

        if not cutout or len(cutout) == 0:
            continue

        hdu = cutout[0]
        time_arr = hdu[1].data['TIME']
        flux_arr = hdu[1].data['FLUX']

        # 3x3 aperture photometry
        ny, nx = flux_arr.shape[1], flux_arr.shape[2]
        cy, cx = ny // 2, nx // 2
        ap = flux_arr[:, max(0, cy-1):cy+2, max(0, cx-1):cx+2]
        lc_flux = np.nansum(ap, axis=(1, 2))

        valid = np.isfinite(time_arr) & np.isfinite(lc_flux) & (lc_flux > 0)
        t_sec, flux_sec = time_arr[valid], lc_flux[valid]

        if len(t_sec) < 5:
            continue

        # Normalize each sector independently to median=1
        med = np.nanmedian(flux_sec)
        flux_sec_norm = flux_sec / med

        all_t.append(t_sec)
        all_flux.append(flux_sec_norm)
        sectors_used.append(sector)
        print(f"    S{sector}: {len(t_sec)} pts")

    if not all_t:
        print(f"    No usable cutout data")
        if ax: ax.text(0.5, 0.5, f'TIC {tic_id}\nNo cutout', ha='center', va='center', transform=ax.transAxes)
        continue

    t = np.concatenate(all_t)
    flux_norm = np.concatenate(all_flux)

    # Sort by time
    sort_idx = np.argsort(t)
    t, flux_norm = t[sort_idx], flux_norm[sort_idx]

    if len(t) < 10:
        print(f"    Too few valid points ({len(t)})")
        continue

    # Stats
    residual = flux_norm - np.nanmedian(flux_norm)
    std = np.nanstd(residual)
    n_outliers = np.sum(np.abs(residual) > 3 * std) if std > 0 else 0
    max_exc = np.nanmax(np.abs(residual)) / std if std > 0 else 0

    sector_str = ','.join(str(s) for s in sectors_used)
    lc_data[tic_id] = {
        'time': t, 'flux': flux_norm, 'gaia_id': gaia_id,
        'sectors': sectors_used, 'n_sectors': len(sectors)
    }

    # Save incrementally after each new download
    with open(lc_path, 'wb') as f:
        pickle.dump(lc_data, f)

    # Plot
    if ax:
        vsx_type = row.get('vsx_type', '')
        plot_lc(ax, t, flux_norm, f"TIC {tic_id} | {vsx_type} | S{sector_str}", sector_str)

    print(f"    Total: {len(t)} pts from {len(sectors_used)}/{len(sectors)} sectors, {n_outliers} outliers, max {max_exc:.1f}sig")

# Hide unused axes
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'tess_lightcurves.png'), dpi=150)
plt.close()
print(f"\n  Plot: {PLOT_DIR}/tess_lightcurves.png")

# Final save
with open(lc_path, 'wb') as f:
    pickle.dump(lc_data, f)
print(f"  Saved {len(lc_data)} light curves to {lc_path}")
