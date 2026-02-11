#!/usr/bin/env python3
"""Step 5: Extract TESS light curves for high-priority candidates.

Uses lightkurve for proper background-subtracted photometry:
  1. Pre-made products (SPOC, QLP, ELEANOR-LITE, TESS-SPOC) where available
  2. TESScut cutouts with lightkurve background estimation as fallback
"""

import os, sys, time, signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from config import CONFIG, DATA_DIR, PLOT_DIR
from lc_utils import collapse_gaps, plot_lc
import pickle
import warnings
warnings.filterwarnings('ignore')
import lightkurve as lk

REFETCH = '--refetch' in sys.argv

os.makedirs(PLOT_DIR, exist_ok=True)


def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


MAX_SECTORS = 12  # cap to avoid multi-hour downloads for CVS-rich targets


def extract_lc_lightkurve(tic_id, ra, dec):
    """Extract background-subtracted TESS light curve using lightkurve.

    Priority order:
      1. Pre-made light curves (SPOC > QLP > ELEANOR-LITE > TESS-SPOC)
      2. TESScut cutouts with background subtraction

    Returns (time, flux_norm, sectors_used, source) or (None, None, [], 'none').
    """
    target = f'TIC {tic_id}'

    # --- Strategy 1: Pre-made light curve products ---
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        sr = lk.search_lightcurve(target, mission='TESS', author='any')
        signal.alarm(0)
    except:
        signal.alarm(0)
        sr = None

    if sr is not None and len(sr) > 0:
        # Prefer highest-quality authors
        author_priority = ['SPOC', 'QLP', 'TESS-SPOC', 'GSFC-ELEANOR-LITE']
        best_author = None
        for pref in author_priority:
            mask = [pref.lower() in str(a).lower() for a in sr.table['author']]
            if any(mask):
                best_author = pref
                break

        # Deduplicate: one product per sector, prefer 120s cadence over 20s
        sector_to_idx = {}
        for i, row in enumerate(sr.table):
            author = str(row['author'])
            if best_author and best_author.lower() not in author.lower():
                continue
            sector = int(str(row['mission']).split()[-1])
            exptime = float(row['exptime'])
            if sector not in sector_to_idx:
                sector_to_idx[sector] = (i, exptime)
            else:
                _, prev_exp = sector_to_idx[sector]
                if abs(exptime - 120) < abs(prev_exp - 120):
                    sector_to_idx[sector] = (i, exptime)

        # Cap sectors: keep evenly-spaced subset for temporal coverage
        sectors_sorted = sorted(sector_to_idx.keys())
        if len(sectors_sorted) > MAX_SECTORS:
            step = len(sectors_sorted) / MAX_SECTORS
            sectors_sorted = [sectors_sorted[int(i * step)] for i in range(MAX_SECTORS)]
            print(f"    Capped to {MAX_SECTORS}/{len(sector_to_idx)} sectors")

        all_t, all_flux, sectors_used = [], [], []
        source = best_author or str(sr.table['author'][0])

        for sector in sectors_sorted:
            idx, exptime = sector_to_idx[sector]
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                lc = sr[idx].download()
                signal.alarm(0)
                if lc is None:
                    continue
                lc = lc.remove_nans().remove_outliers(sigma=10)
                t = np.array(lc.time.value, dtype=float)
                f = np.array(lc.flux.value, dtype=float)
                good = np.isfinite(t) & np.isfinite(f) & (f > 0)
                t, f = t[good], f[good]
                if len(t) < 10:
                    continue
                med = np.nanmedian(f)
                f = f / med
                all_t.append(t)
                all_flux.append(f)
                sectors_used.append(sector)
                print(f"    S{sector}: {len(t)} pts ({source}, {exptime:.0f}s)")
            except:
                signal.alarm(0)
                print(f"    S{sector}: download failed ({source})")
                continue

        if all_t:
            t = np.concatenate(all_t)
            flux = np.concatenate(all_flux)
            sort_idx = np.argsort(t)
            return t[sort_idx], flux[sort_idx], sectors_used, source

    # --- Strategy 2: TESScut with background subtraction ---
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        sr_cut = lk.search_tesscut(target)
        signal.alarm(0)
    except:
        signal.alarm(0)
        sr_cut = None

    if sr_cut is None or len(sr_cut) == 0:
        return None, None, [], 'none'

    all_t, all_flux, sectors_used = [], [], []
    source = 'TESScut+bkg_sub'

    # Cap TESScut sectors too
    indices = list(range(len(sr_cut)))
    if len(indices) > MAX_SECTORS:
        step = len(indices) / MAX_SECTORS
        indices = [int(i * step) for i in range(MAX_SECTORS)]
        print(f"    Capped TESScut to {MAX_SECTORS}/{len(sr_cut)} sectors")

    for i in indices:
        sector = int(str(sr_cut.table['mission'][i]).split()[-1])
        for attempt in range(3):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)
                tpf = sr_cut[i].download(cutout_size=11)
                signal.alarm(0)
                break
            except TimeoutError:
                signal.alarm(0)
                print(f"    S{sector}: timeout attempt {attempt+1}")
                if attempt < 2:
                    time.sleep(5)
                tpf = None
            except Exception as e:
                signal.alarm(0)
                err = str(e).lower()
                if any(k in err for k in ['disconnect', 'connection', 'timeout', 'reset']):
                    if attempt < 2:
                        print(f"    S{sector}: network error, retrying...")
                        time.sleep(5 * (attempt + 1))
                        continue
                print(f"    S{sector}: error: {str(e)[:60]}")
                tpf = None
                break

        if tpf is None:
            continue

        try:
            # Create aperture from brightest pixels (threshold mask)
            ap_mask = tpf.create_threshold_mask(threshold=5, reference_pixel='center')
            if ap_mask is None or ap_mask.sum() == 0:
                ap_mask = tpf.create_threshold_mask(threshold=3, reference_pixel='center')
            if ap_mask is None or ap_mask.sum() == 0:
                # Fallback: 3x3 center
                ny, nx = tpf.shape[1], tpf.shape[2]
                cy, cx = ny // 2, nx // 2
                ap_mask = np.zeros((ny, nx), dtype=bool)
                ap_mask[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = True

            # Background: median of pixels outside aperture
            bkg_mask = ~ap_mask
            if bkg_mask.sum() > 5:
                bkg_flux = np.nanmedian(tpf.flux.value[:, bkg_mask], axis=1)
                n_ap = ap_mask.sum()
                # Subtract scaled background from aperture sum
                raw_lc = tpf.flux.value[:, ap_mask].sum(axis=1)
                bkg_subtracted = raw_lc - bkg_flux * n_ap
            else:
                bkg_subtracted = tpf.flux.value[:, ap_mask].sum(axis=1)

            t = tpf.time.value
            f = bkg_subtracted
            good = np.isfinite(t) & np.isfinite(f) & (f > 0)
            t, f = t[good], f[good]

            if len(t) < 10:
                continue

            # Remove quality-flagged cadences
            if hasattr(tpf, 'quality') and tpf.quality is not None:
                q = tpf.quality[good] if len(tpf.quality) == len(good) else None
                if q is not None:
                    good_q = q == 0
                    t, f = t[good_q], f[good_q]

            if len(t) < 10:
                continue

            # Normalize
            med = np.nanmedian(f)
            f = f / med

            all_t.append(t)
            all_flux.append(f)
            sectors_used.append(sector)
            print(f"    S{sector}: {len(t)} pts (TESScut bkg-sub, {ap_mask.sum()}px aperture)")

        except Exception as e:
            print(f"    S{sector}: extraction error: {str(e)[:60]}")
            continue

    if not all_t:
        return None, None, [], 'none'

    t = np.concatenate(all_t)
    flux = np.concatenate(all_flux)
    sort_idx = np.argsort(t)
    return t[sort_idx], flux[sort_idx], sectors_used, source


# ── Main ────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 5: TESS LIGHT CURVE EXTRACTION (lightkurve)")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '04_high_priority.csv'))
candidates = df[df['tic_id'].notna()].copy()
print(f"  {len(candidates)} high-priority candidates with TIC matches")

# --- Deduplicate by sky proximity (TESS 21"/px, aperture overlap) ---
coords = SkyCoord(ra=candidates['ra'].values*u.deg, dec=candidates['dec'].values*u.deg)
DEDUP_RADIUS = 63  # arcsec

keep_mask = np.ones(len(candidates), dtype=bool)
indices = list(candidates.index)
groups = {}

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
    # Invalidate old-format caches (no 'source' key = old raw aperture)
    old_keys = [k for k, v in lc_data.items() if 'source' not in v]
    for k in old_keys:
        del lc_data[k]
    if old_keys:
        print(f"  Invalidated {len(old_keys)} old-format (no bkg-sub) cached entries")
    # Ensure all cached arrays are plain numpy (not astropy MaskedNDArray)
    for k, v in lc_data.items():
        v['time'] = np.array(v['time'], dtype=float)
        v['flux'] = np.array(v['flux'], dtype=float)
    print(f"  Loaded {len(lc_data)} cached light curves from previous run")
else:
    lc_data = {}

n_cols = 4
n_rows = max((n + n_cols - 1) // n_cols, 1)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
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
        sector_str = ','.join(str(s) for s in cached.get('sectors', []))
        src = cached.get('source', '?')
        print(f"    Cached: {len(t)} pts from S{sector_str} ({src})")
        if ax:
            vsx_type = row.get('vsx_type', '')
            plot_lc(ax, t, flux_norm, f"TIC {tic_id} | {vsx_type} | S{sector_str}", sector_str)
        continue

    # Extract with lightkurve
    t, flux_norm, sectors_used, source = extract_lc_lightkurve(tic_id, row['ra'], row['dec'])

    if t is None or len(t) < 10:
        print(f"    No usable data")
        if ax:
            ax.text(0.5, 0.5, f'TIC {tic_id}\nNo data', ha='center', va='center', transform=ax.transAxes)
        continue

    # Stats
    residual = flux_norm - np.nanmedian(flux_norm)
    std = np.nanstd(residual)
    n_outliers = np.sum(np.abs(residual) > 3 * std) if std > 0 else 0
    max_exc = np.nanmax(np.abs(residual)) / std if std > 0 else 0

    sector_str = ','.join(str(s) for s in sectors_used)
    lc_data[tic_id] = {
        'time': t, 'flux': flux_norm, 'gaia_id': gaia_id,
        'sectors': sectors_used, 'n_sectors': len(sectors_used),
        'source': source
    }

    # Save incrementally
    with open(lc_path, 'wb') as f:
        pickle.dump(lc_data, f)

    # Plot
    if ax:
        vsx_type = row.get('vsx_type', '')
        plot_lc(ax, t, flux_norm, f"TIC {tic_id} | {vsx_type} | S{sector_str}", sector_str)

    print(f"    Total: {len(t)} pts from {len(sectors_used)} sectors, "
          f"{n_outliers} outliers, max {max_exc:.1f}sig ({source})")

# Hide unused axes
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'tess_lightcurves.png'), dpi=120)
plt.close()
print(f"\n  Plot: {PLOT_DIR}/tess_lightcurves.png")

# Final save
with open(lc_path, 'wb') as f:
    pickle.dump(lc_data, f)
print(f"  Saved {len(lc_data)} light curves to {lc_path}")
