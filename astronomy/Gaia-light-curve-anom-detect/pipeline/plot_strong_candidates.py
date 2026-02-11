#!/usr/bin/env python3
"""Generate publication-quality figures for STRONG novel CV candidates.

Produces a multi-panel figure for each STRONG candidate showing:
  - Raw TESS light curve (gap-collapsed)
  - Quiescent detrended light curve
  - Lomb-Scargle periodogram
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pickle
from config import DATA_DIR, PLOT_DIR
from lc_utils import collapse_gaps, isolate_quiescent, detrend, mask_tess_systematics

os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("STRONG CANDIDATE FIGURES")
print("=" * 70)

# Load data
with open(os.path.join(DATA_DIR, '05_lc_data.pkl'), 'rb') as f:
    lc_data = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, '07_deep_investigated.csv'))
strong = df[df['assessment'].str.contains('STRONG', na=False)].copy()

print(f"  {len(strong)} STRONG candidates\n")

# Match TIC IDs to lc_data
candidates = []
for _, row in strong.iterrows():
    tic_id = int(row['tic_id']) if pd.notna(row['tic_id']) else None
    if tic_id and tic_id in lc_data:
        candidates.append((tic_id, row))
    else:
        print(f"  WARNING: TIC {tic_id} (Gaia {row['source_id']}) not in lc_data, skipping")

n = len(candidates)
print(f"  {n} candidates with TESS data\n")

if n == 0:
    print("  No candidates to plot")
    exit(0)

# Create figure: 3 columns (raw LC, detrended LC, periodogram) x n rows
fig, axes = plt.subplots(n, 3, figsize=(18, 3.2 * n), squeeze=False)

for i, (tic_id, row) in enumerate(candidates):
    lc = lc_data[tic_id]
    t = np.array(lc['time'], dtype=float)
    flux = np.array(lc['flux'], dtype=float)
    gaia_id = int(row['source_id'])
    sectors = lc.get('sectors', [])
    source = lc.get('source', '?')
    sector_str = ','.join(str(s) for s in sectors)
    notes = row.get('assessment_notes', '')
    g_mag = row.get('phot_g_mean_mag', 0)
    teff = row.get('gaia_teff', None)
    teff_str = f"Teff={int(teff)}K" if pd.notna(teff) else "Teff=?"

    print(f"  [{i+1}/{n}] TIC {tic_id} (Gaia DR3 {gaia_id})")
    print(f"    G={g_mag:.1f}, {teff_str}, sectors={sector_str}, source={source}")
    print(f"    Indicators: {notes}")

    ax_raw = axes[i][0]
    ax_det = axes[i][1]
    ax_ls = axes[i][2]

    # --- Panel 1: Raw light curve (gap-collapsed) ---
    t_plot, flux_plot, breaks, segments = collapse_gaps(t, flux)
    ax_raw.scatter(t_plot, flux_plot, s=0.5, alpha=0.5, c='steelblue')
    ax_raw.axhline(1, color='gray', ls='--', alpha=0.3)
    for bp in breaks:
        ax_raw.axvline(bp, color='orange', ls=':', alpha=0.4, lw=0.8)
    ax_raw.set_yscale('log')
    ax_raw.set_xlabel('Days (collapsed)', fontsize=8)
    ax_raw.set_ylabel('Normalized Flux', fontsize=8)
    ax_raw.set_title(f'TIC {tic_id} | G={g_mag:.1f} | S{sector_str}\n'
                     f'Gaia DR3 {gaia_id} | {teff_str}',
                     fontsize=8, fontweight='bold')
    ax_raw.tick_params(labelsize=7)

    # --- Panel 2: Quiescent detrended ---
    t_q, flux_q, n_outburst, n_quiescent = isolate_quiescent(t, flux)

    if len(t_q) < 20:
        ax_det.text(0.5, 0.5, f'Too few quiescent\npoints ({len(t_q)})',
                    ha='center', va='center', transform=ax_det.transAxes, fontsize=9)
        ax_ls.text(0.5, 0.5, 'No period analysis', ha='center', va='center',
                   transform=ax_ls.transAxes, fontsize=9)
        print(f"    Too few quiescent points ({len(t_q)})")
        continue

    t_q, flux_detrend = detrend(t_q, flux_q)

    ax_det.scatter(t_q, flux_detrend, s=0.5, alpha=0.5, c='steelblue')
    ax_det.axhline(1, color='gray', ls='--', alpha=0.3)
    y_lo, y_hi = np.percentile(flux_detrend, [0.5, 99.5])
    y_pad = (y_hi - y_lo) * 0.15
    ax_det.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax_det.set_xlabel('BTJD', fontsize=8)
    ax_det.set_ylabel('Detrended Flux', fontsize=8)
    ax_det.set_title(f'Quiescent: {n_quiescent}/{len(t)} pts '
                     f'({n_outburst} outburst removed)', fontsize=8)
    ax_det.tick_params(labelsize=7)

    # --- Panel 3: Lomb-Scargle periodogram ---
    dt = np.diff(np.sort(t_q))
    dt_clean = dt[dt > 0]
    median_cadence = np.median(dt_clean) if len(dt_clean) > 0 else 30.0 / (24 * 60)
    nyquist_period = 2 * median_cadence
    min_period = nyquist_period
    max_period = min(0.5, (t_q[-1] - t_q[0]) / 3)

    if min_period >= max_period:
        ax_ls.text(0.5, 0.5, 'Period range\ntoo narrow', ha='center', va='center',
                   transform=ax_ls.transAxes, fontsize=9)
        print(f"    Period range too narrow")
        continue

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

    ax_ls.semilogx(periods * 24 * 60, power, 'k-', lw=0.6)
    ax_ls.axvline(best_period * 24 * 60, color='red', ls='--', alpha=0.7, lw=1,
                  label=f'Best: {best_period*24*60:.1f}m')
    ax_ls.set_xlabel('Period (min)', fontsize=8)
    ax_ls.set_ylabel('Whitened Power', fontsize=8)
    sig_str = f'FAP={fap:.1e}' if fap > 0 else 'FAP<1e-300'
    ax_ls.set_title(f'L-S: {best_period*24*60:.1f} min ({sig_str})', fontsize=8)
    ax_ls.legend(fontsize=7, loc='upper right')
    ax_ls.tick_params(labelsize=7)

    print(f"    Best period: {best_period*24*60:.1f} min, FAP={fap:.1e}")

fig.suptitle('STRONG Novel CV Candidates — Gaia CV Hunter Pipeline',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
outpath = os.path.join(PLOT_DIR, 'strong_candidates.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {outpath}")
