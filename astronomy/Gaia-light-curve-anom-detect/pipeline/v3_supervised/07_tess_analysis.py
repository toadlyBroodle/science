#!/usr/bin/env python3
"""Step 7: TESS light curve fetch + period analysis for top CV candidates.

Combined LC extraction and Lomb-Scargle period search for the top N candidates
from step 6. Searches TESS by Gaia coordinates (no TIC ID needed).

Produces a 3-column figure per candidate:
  (a) Raw light curve (gap-collapsed, log scale)
  (b) Lomb-Scargle periodogram (whitened, log period axis)
  (c) Phase-folded light curve (binned median + scatter) if FAP < 0.01
"""

import os, sys, time, signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
import astropy.units as u
import pickle
import warnings
warnings.filterwarnings('ignore')
import lightkurve as lk

# Import config from this directory, lc_utils from parent pipeline dir
from config import CONFIG, DATA_DIR, PLOT_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lc_utils import (collapse_gaps, plot_lc, isolate_quiescent,
                       detrend, mask_tess_systematics)

REFETCH = '--refetch' in sys.argv
N_CANDIDATES = CONFIG.get('n_tess_candidates', 20)
MAX_SECTORS = 12

os.makedirs(PLOT_DIR, exist_ok=True)


# ── Timeout helper ───────────────────────────────────────────────────

def timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")


# ── LC extraction (adapted from v2 05_tess_lightcurves.py) ──────────

def extract_lc_by_coords(ra, dec):
    """Extract TESS light curve by sky coordinates using lightkurve.

    Two-tier strategy:
      1. Pre-made products (SPOC > QLP > TESS-SPOC > ELEANOR-LITE)
      2. TESScut cutouts with background subtraction

    Returns (time, flux_norm, sectors_used, source) or (None, None, [], 'none').
    """
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    # --- Strategy 1: Pre-made light curve products ---
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        sr = lk.search_lightcurve(coord, mission='TESS', radius=21*u.arcsec)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        sr = None

    if sr is not None and len(sr) > 0:
        author_priority = ['SPOC', 'QLP', 'TESS-SPOC', 'GSFC-ELEANOR-LITE']
        best_author = None
        for pref in author_priority:
            mask = [pref.lower() in str(a).lower() for a in sr.table['author']]
            if any(mask):
                best_author = pref
                break

        # Deduplicate: one product per sector, prefer 120s cadence
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

        # Cap sectors for temporal coverage
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
            except Exception:
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
        sr_cut = lk.search_tesscut(coord)
        signal.alarm(0)
    except Exception:
        signal.alarm(0)
        sr_cut = None

    if sr_cut is None or len(sr_cut) == 0:
        return None, None, [], 'none'

    all_t, all_flux, sectors_used = [], [], []
    source = 'TESScut+bkg_sub'

    indices = list(range(len(sr_cut)))
    if len(indices) > MAX_SECTORS:
        step = len(indices) / MAX_SECTORS
        indices = [int(i * step) for i in range(MAX_SECTORS)]
        print(f"    Capped TESScut to {MAX_SECTORS}/{len(sr_cut)} sectors")

    for i in indices:
        sector = int(str(sr_cut.table['mission'][i]).split()[-1])
        tpf = None
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
            ap_mask = tpf.create_threshold_mask(threshold=5, reference_pixel='center')
            if ap_mask is None or ap_mask.sum() == 0:
                ap_mask = tpf.create_threshold_mask(threshold=3, reference_pixel='center')
            if ap_mask is None or ap_mask.sum() == 0:
                ny, nx = tpf.shape[1], tpf.shape[2]
                cy, cx = ny // 2, nx // 2
                ap_mask = np.zeros((ny, nx), dtype=bool)
                ap_mask[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = True

            bkg_mask = ~ap_mask
            if bkg_mask.sum() > 5:
                bkg_flux = np.nanmedian(tpf.flux.value[:, bkg_mask], axis=1)
                n_ap = ap_mask.sum()
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

            if hasattr(tpf, 'quality') and tpf.quality is not None:
                q = tpf.quality[good] if len(tpf.quality) == len(good) else None
                if q is not None:
                    good_q = q == 0
                    t, f = t[good_q], f[good_q]
            if len(t) < 10:
                continue

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


# ── Period analysis (adapted from v2 06_period_analysis.py) ─────────

def run_period_analysis(t, flux):
    """Run Lomb-Scargle period search on a light curve.

    Returns dict with best_period, fap, top_periods, n_quiescent, n_outburst,
    or None if insufficient data.
    """
    # Isolate quiescent data
    t_q, flux_q, n_outburst, n_quiescent = isolate_quiescent(t, flux)
    if len(t_q) < 20:
        return None, None, None

    # Detrend
    t_q, flux_detrend = detrend(t_q, flux_q)

    # Cadence and Nyquist
    dt = np.diff(np.sort(t_q))
    dt_clean = dt[dt > 0]
    median_cadence = np.median(dt_clean) if len(dt_clean) > 0 else 30.0 / (24 * 60)
    nyquist_period = 2 * median_cadence

    min_period = nyquist_period
    max_period = min(0.5, (t_q[-1] - t_q[0]) / 3)  # 12 hours max

    if min_period >= max_period:
        return None, None, None

    # Downsample dense LCs
    t_ls, f_ls = t_q, flux_detrend
    if len(t_q) > 30000:
        step = len(t_q) // 30000
        t_ls, f_ls = t_q[::step], flux_detrend[::step]

    freq = np.geomspace(1/max_period, 1/min_period, 5000)
    ls = LombScargle(t_ls, f_ls)
    power_raw = ls.power(freq)
    periods = 1 / freq

    power = mask_tess_systematics(periods, power_raw)

    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power_raw[best_idx]
    fap = ls.false_alarm_probability(best_power)

    top_indices = np.argsort(power)[-5:][::-1]
    top_periods = periods[top_indices]

    # Amplitude estimate (peak-to-trough of binned phase curve)
    phase = ((t_q - t_q[0]) / best_period) % 1.0
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_meds = []
    for b in range(n_bins):
        bmask = (phase >= bin_edges[b]) & (phase < bin_edges[b+1])
        if bmask.sum() >= 3:
            bin_meds.append(np.nanmedian(flux_detrend[bmask]))
    amplitude = (max(bin_meds) - min(bin_meds)) if len(bin_meds) >= 5 else np.nan

    result = {
        'best_period': best_period, 'fap': fap,
        'top_periods': top_periods.tolist(),
        'amplitude': amplitude,
        'n_quiescent': len(t_q), 'n_outburst': n_outburst,
    }

    return result, (t_q, flux_detrend, periods, power)


# ── Main ─────────────────────────────────────────────────────────────

print("=" * 70)
print("STEP 7: TESS LIGHT CURVE & PERIOD ANALYSIS")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '06_top_candidates.csv'))
candidates = df.head(N_CANDIDATES).copy()
n = len(candidates)
print(f"  Top {n} candidates selected for TESS follow-up\n")

# --- Load cached LC data ---
lc_path = os.path.join(DATA_DIR, '07_lc_data.pkl')
if REFETCH:
    lc_data = {}
    print(f"  --refetch: ignoring cached data, re-downloading all")
elif os.path.exists(lc_path):
    with open(lc_path, 'rb') as f:
        lc_data = pickle.load(f)
    for k, v in lc_data.items():
        v['time'] = np.array(v['time'], dtype=float)
        v['flux'] = np.array(v['flux'], dtype=float)
    print(f"  Loaded {len(lc_data)} cached light curves from previous run")
else:
    lc_data = {}

# --- Fetch TESS light curves ---
for i, (_, row) in enumerate(candidates.iterrows()):
    gaia_id = int(row['source_id'])
    ra, dec = row['ra'], row['dec']
    g_mag = row['phot_g_mean_mag']
    p_oof = row.get('cv_probability_oof', np.nan)

    print(f"  [{i+1}/{n}] Gaia DR3 {gaia_id} (G={g_mag:.1f}, P_oof={p_oof:.3f})", flush=True)

    if gaia_id in lc_data:
        cached = lc_data[gaia_id]
        sector_str = ','.join(str(s) for s in cached.get('sectors', []))
        src = cached.get('source', '?')
        print(f"    Cached: {len(cached['time'])} pts from S{sector_str} ({src})")
        continue

    t, flux_norm, sectors_used, source = extract_lc_by_coords(ra, dec)

    if t is None or len(t) < 10:
        print(f"    No usable TESS data")
        lc_data[gaia_id] = {
            'time': np.array([]), 'flux': np.array([]),
            'sectors': [], 'source': 'none'
        }
    else:
        sector_str = ','.join(str(s) for s in sectors_used)
        lc_data[gaia_id] = {
            'time': t, 'flux': flux_norm,
            'sectors': sectors_used, 'n_sectors': len(sectors_used),
            'source': source
        }
        print(f"    Total: {len(t)} pts from {len(sectors_used)} sectors ({source})")

    # Save incrementally
    with open(lc_path, 'wb') as f:
        pickle.dump(lc_data, f)

print(f"\n  Saved {len(lc_data)} light curves to {lc_path}")

# --- Period analysis ---
print(f"\n{'='*70}")
print("PERIOD ANALYSIS")
print(f"{'='*70}\n")

period_results = {}

for i, (_, row) in enumerate(candidates.iterrows()):
    gaia_id = int(row['source_id'])
    g_mag = row['phot_g_mean_mag']
    p_oof = row.get('cv_probability_oof', np.nan)

    cached = lc_data.get(gaia_id)
    if cached is None or len(cached['time']) < 20:
        print(f"  [{i+1}/{n}] Gaia DR3 {gaia_id}: skipped (no/insufficient data)")
        continue

    t, flux = cached['time'], cached['flux']
    print(f"  [{i+1}/{n}] Gaia DR3 {gaia_id} ({len(t)} pts)")

    result, analysis_data = run_period_analysis(t, flux)

    if result is None:
        print(f"    No significant period (too few quiescent pts or narrow range)")
        period_results[gaia_id] = {
            'best_period': np.nan, 'fap': np.nan, 'amplitude': np.nan,
            'top_periods': [], 'n_quiescent': 0, 'n_outburst': 0,
        }
        continue

    period_results[gaia_id] = result
    bp = result['best_period']
    fap = result['fap']
    amp = result['amplitude']
    sig = "***" if fap < 0.01 else ""
    print(f"    Best: {bp*24*60:.1f} min ({bp:.6f} d), FAP={fap:.2e} {sig}")
    print(f"    Amplitude: {amp:.4f}" if np.isfinite(amp) else "    Amplitude: N/A")

# --- Figure: N rows x 3 columns ---
print(f"\n{'='*70}")
print("GENERATING FIGURE")
print(f"{'='*70}\n")

fig, axes = plt.subplots(n, 3, figsize=(18, 2.8 * n), squeeze=False)

for i, (_, row) in enumerate(candidates.iterrows()):
    gaia_id = int(row['source_id'])
    g_mag = row['phot_g_mean_mag']
    p_oof = row.get('cv_probability_oof', np.nan)
    ax_raw, ax_ls, ax_fold = axes[i]

    cached = lc_data.get(gaia_id)
    has_data = cached is not None and len(cached['time']) >= 20
    pr = period_results.get(gaia_id)

    label = f"Gaia DR3 {gaia_id} (G={g_mag:.1f}, P_oof={p_oof:.2f})"

    if not has_data:
        for ax in (ax_raw, ax_ls, ax_fold):
            ax.text(0.5, 0.5, 'No TESS data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(label, fontsize=7)
        continue

    t, flux = cached['time'], cached['flux']
    sector_str = ','.join(str(s) for s in cached.get('sectors', []))

    # (a) Raw light curve
    plot_lc(ax_raw, t, flux, f"{label}\nS{sector_str}", sector_str)

    # Period analysis data for panels (b) and (c)
    if pr is None or np.isnan(pr.get('fap', np.nan)):
        ax_ls.text(0.5, 0.5, 'Insufficient data\nfor period search',
                   ha='center', va='center', transform=ax_ls.transAxes,
                   fontsize=8, color='gray')
        ax_fold.text(0.5, 0.5, 'No period analysis',
                     ha='center', va='center', transform=ax_fold.transAxes,
                     fontsize=8, color='gray')
        continue

    # Re-run analysis to get plot data (periods, power, detrended LC)
    result, analysis_data = run_period_analysis(t, flux)
    if analysis_data is None:
        ax_ls.text(0.5, 0.5, 'Analysis failed', ha='center', va='center',
                   transform=ax_ls.transAxes, fontsize=8, color='gray')
        ax_fold.text(0.5, 0.5, 'No period', ha='center', va='center',
                     transform=ax_fold.transAxes, fontsize=8, color='gray')
        continue

    t_q, flux_detrend, periods, power = analysis_data
    best_period = pr['best_period']
    fap = pr['fap']

    # (b) Periodogram
    ax_ls.semilogx(periods * 24 * 60, power, 'k-', lw=0.4)
    ax_ls.axvline(best_period * 24 * 60, color='red', ls='--', alpha=0.7, lw=0.8,
                  label=f'{best_period*24*60:.1f}m')
    ax_ls.set_xlabel('Period (min)', fontsize=7)
    ax_ls.set_ylabel('Power (whitened)', fontsize=7)
    ax_ls.set_title(f'FAP={fap:.1e}', fontsize=7)
    ax_ls.legend(fontsize=6, loc='upper right')
    ax_ls.tick_params(labelsize=6)

    # (c) Phase-folded light curve
    if fap < 0.01:
        # Clip outliers for cleaner fold
        med_d = np.nanmedian(flux_detrend)
        std_d = np.nanstd(flux_detrend)
        clip = np.abs(flux_detrend - med_d) < 3 * std_d
        t_clip, flux_clip = t_q[clip], flux_detrend[clip]

        phase = ((t_clip - t_clip[0]) / best_period) % 1.0

        # Scatter (subsample if dense)
        if len(phase) > 3000:
            rng = np.random.default_rng(42)
            sub = rng.choice(len(phase), 3000, replace=False)
            ax_fold.scatter(phase[sub], flux_clip[sub], s=0.3, alpha=0.06, c='steelblue')
            ax_fold.scatter(phase[sub] + 1, flux_clip[sub], s=0.3, alpha=0.06, c='steelblue')
        else:
            ax_fold.scatter(phase, flux_clip, s=0.5, alpha=0.1, c='steelblue')
            ax_fold.scatter(phase + 1, flux_clip, s=0.5, alpha=0.1, c='steelblue')

        # Binned median
        n_bins = 30
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers, bin_medians, bin_errs = [], [], []
        for b in range(n_bins):
            bmask = (phase >= bin_edges[b]) & (phase < bin_edges[b+1])
            if bmask.sum() >= 3:
                bin_centers.append((bin_edges[b] + bin_edges[b+1]) / 2)
                bin_medians.append(np.nanmedian(flux_clip[bmask]))
                bin_errs.append(np.nanstd(flux_clip[bmask]) / np.sqrt(bmask.sum()))
        if bin_centers:
            bc = np.array(bin_centers)
            bm = np.array(bin_medians)
            be = np.array(bin_errs)
            ax_fold.fill_between(bc, bm - be, bm + be, color='red', alpha=0.15)
            ax_fold.fill_between(bc + 1, bm - be, bm + be, color='red', alpha=0.15)
            ax_fold.plot(bc, bm, 'r-', lw=1.5)
            ax_fold.plot(bc + 1, bm, 'r-', lw=1.5)

        ax_fold.axhline(1, color='gray', ls='--', alpha=0.3)
        ax_fold.set_xlim(0, 2)
        y_lo, y_hi = np.percentile(flux_clip, [1, 99])
        y_pad = (y_hi - y_lo) * 0.2
        ax_fold.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax_fold.set_xlabel('Phase', fontsize=7)
        ax_fold.set_ylabel('Flux', fontsize=7)
        ax_fold.set_title(f'P={best_period*24*60:.1f} min', fontsize=7)
        ax_fold.tick_params(labelsize=6)
    else:
        ax_fold.text(0.5, 0.5, 'No significant period\n(FAP > 0.01)',
                     ha='center', va='center', transform=ax_fold.transAxes,
                     fontsize=8, color='gray')
        ax_fold.set_title(f'P={best_period*24*60:.1f} min (not sig.)', fontsize=7)

plt.tight_layout()
fig_path = os.path.join(PLOT_DIR, 'tess_lc_analysis.png')
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"  Figure saved: {fig_path}")

# --- Save results ---
with open(os.path.join(DATA_DIR, '07_period_results.pkl'), 'wb') as f:
    pickle.dump(period_results, f)

# Build output CSV
rows = []
for _, row in candidates.iterrows():
    gaia_id = int(row['source_id'])
    pr = period_results.get(gaia_id, {})
    cached = lc_data.get(gaia_id, {})
    rows.append({
        'source_id': gaia_id,
        'ra': row['ra'],
        'dec': row['dec'],
        'phot_g_mean_mag': row['phot_g_mean_mag'],
        'cv_probability_oof': row.get('cv_probability_oof', np.nan),
        'tess_sectors': ','.join(str(s) for s in cached.get('sectors', [])),
        'tess_source': cached.get('source', 'none'),
        'n_pts': len(cached.get('time', [])),
        'best_period_d': pr.get('best_period', np.nan),
        'best_period_min': pr.get('best_period', np.nan) * 24 * 60
            if np.isfinite(pr.get('best_period', np.nan)) else np.nan,
        'fap': pr.get('fap', np.nan),
        'amplitude': pr.get('amplitude', np.nan),
        'n_quiescent': pr.get('n_quiescent', 0),
        'n_outburst': pr.get('n_outburst', 0),
    })
out_df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, '07_tess_candidates.csv')
out_df.to_csv(out_path, index=False)
print(f"  Saved {out_path}")

# --- Summary table ---
print(f"\n{'='*70}")
print("TESS PERIOD ANALYSIS SUMMARY")
print(f"{'='*70}")
print(f"{'Gaia DR3 ID':>22s}  {'G':>5s}  {'P_oof':>5s}  {'Period':>8s}  {'FAP':>9s}  {'Amp':>6s}  {'Pts':>5s}")
print("-" * 70)
for _, r in out_df.iterrows():
    sig = "***" if r['fap'] < 0.01 else "   " if np.isfinite(r['fap']) else "   "
    p_str = f"{r['best_period_min']:.1f}m" if np.isfinite(r['best_period_min']) else "---"
    fap_str = f"{r['fap']:.1e}" if np.isfinite(r['fap']) else "---"
    amp_str = f"{r['amplitude']:.4f}" if np.isfinite(r['amplitude']) else "---"
    print(f"  {int(r['source_id']):>20d}  {r['phot_g_mean_mag']:5.1f}  "
          f"{r['cv_probability_oof']:5.3f}  {p_str:>8s}  {fap_str:>9s} {sig} {amp_str:>6s}  {r['n_pts']:5.0f}")

n_sig = out_df[out_df['fap'] < 0.01].shape[0]
n_data = out_df[out_df['n_pts'] > 0].shape[0]
print(f"\n  {n_data}/{n} candidates with TESS data")
print(f"  {n_sig}/{n_data} with significant periods (FAP < 0.01)")
