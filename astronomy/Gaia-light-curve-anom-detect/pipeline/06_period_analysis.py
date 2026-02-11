#!/usr/bin/env python3
"""Step 6: Period analysis (Lomb-Scargle + phase folding) for extracted TESS LCs."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pickle
from config import CONFIG, DATA_DIR, PLOT_DIR
from lc_utils import isolate_quiescent, detrend, mask_tess_systematics

os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 6: PERIOD ANALYSIS")
print("=" * 70)


with open(os.path.join(DATA_DIR, '05_lc_data.pkl'), 'rb') as f:
    lc_data = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_DIR, '04_high_priority.csv'))
n = len(lc_data)
print(f"  Analyzing {n} light curves\n")

period_results = {}
detrended_cache = {}  # Cache quiescent detrended data for phase folding

# Compact grid: 2 cols per candidate (LC + periodogram), 3 candidates per row = 6 columns
n_per_row = 3
n_grid_rows = max((n + n_per_row - 1) // n_per_row, 1)
fig, axes = plt.subplots(n_grid_rows, n_per_row * 2, figsize=(20, 2.8 * n_grid_rows), squeeze=False)

for plot_idx, (tic_id, lc) in enumerate(lc_data.items()):
    grid_row = plot_idx // n_per_row
    grid_col = (plot_idx % n_per_row) * 2
    ax_lc = axes[grid_row][grid_col]
    ax_ls = axes[grid_row][grid_col + 1]
    t = lc['time']
    flux = lc['flux']
    gaia_id = lc['gaia_id']

    print(f"  [{plot_idx+1}/{n}] TIC {tic_id} (Gaia DR3 {gaia_id})")

    # VSX period lookup
    cand = df[df['source_id'] == gaia_id]
    vsx_period = None
    if len(cand) > 0 and pd.notna(cand.iloc[0].get('vsx_period')):
        vsx_period = cand.iloc[0]['vsx_period']

    # Cadence and Nyquist
    dt = np.diff(np.sort(t))
    dt_clean = dt[dt > 0]
    median_cadence = np.median(dt_clean) if len(dt_clean) > 0 else 30.0 / (24 * 60)
    nyquist_period = 2 * median_cadence
    print(f"    Cadence: {median_cadence*24*60:.1f} min, Nyquist: {nyquist_period*24*60:.1f} min")

    # Step 1: Isolate quiescent data (excise full outburst episodes)
    t_q, flux_q, n_outburst, n_quiescent = isolate_quiescent(t, flux)
    print(f"    Quiescent: {n_quiescent}/{len(t)}, outburst episodes removed: {n_outburst}")

    if len(t_q) < 20:
        print(f"    Too few quiescent points, skipping")
        ax_lc.text(0.5, 0.5, 'Too few points', ha='center', va='center', transform=ax_lc.transAxes)
        continue

    # Step 2: Median filter detrend
    t_q, flux_detrend = detrend(t_q, flux_q)
    detrended_cache[tic_id] = (t_q, flux_detrend)

    # Step 3: Lomb-Scargle (capped at 12 hours to focus on orbital periods)
    min_period = nyquist_period
    max_period = min(0.5, (t_q[-1] - t_q[0]) / 3)  # 0.5 days = 12 hours max

    if min_period >= max_period:
        print(f"    Period range too narrow")
        continue

    # Downsample dense light curves for speed (keep Nyquist safe)
    t_ls, f_ls = t_q, flux_detrend
    if len(t_q) > 30000:
        step = len(t_q) // 30000
        t_ls, f_ls = t_q[::step], flux_detrend[::step]

    freq = np.geomspace(1/max_period, 1/min_period, 5000)
    ls = LombScargle(t_ls, f_ls)
    power_raw = ls.power(freq)
    periods = 1 / freq

    # Mask TESS systematic frequencies (10, 15, 20, 30 min + boundary)
    power = mask_tess_systematics(periods, power_raw)

    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power_raw[best_idx]  # FAP from raw power at that frequency
    fap = ls.false_alarm_probability(best_power)

    # Top 5 peaks (from masked power)
    top_indices = np.argsort(power)[-5:][::-1]
    top_periods = periods[top_indices]

    print(f"    Best: {best_period*24*60:.1f} min ({best_period:.6f} d), FAP={fap:.2e}")
    print(f"    Top 5: {', '.join(f'{p*24*60:.1f}' for p in top_periods)} min")

    # VSX match check
    vsx_match = None
    if vsx_period and vsx_period > 0:
        if vsx_period * 24 * 60 < nyquist_period * 24 * 60:
            print(f"    VSX period ({vsx_period*24*60:.0f} min) below Nyquist ({nyquist_period*24*60:.0f} min)")
        else:
            for tp in top_periods:
                for harmonic in [1, 2, 3, 0.5]:
                    if abs(tp - vsx_period * harmonic) / (vsx_period * harmonic) < 0.05:
                        vsx_match = (tp, harmonic)
                        print(f"    VSX MATCH: 1:{harmonic:.0f} harmonic at {tp*24*60:.1f} min")
                        break
                if vsx_match: break
            if not vsx_match:
                print(f"    VSX period: {vsx_period*24*60:.1f} min - no match in top peaks")

    period_results[tic_id] = {
        'best_period': best_period, 'fap': fap, 'top_periods': top_periods.tolist(),
        'vsx_period': vsx_period, 'vsx_match': vsx_match is not None,
        'n_quiescent': len(t_q), 'n_outburst': n_outburst
    }

    # Plot: detrended LC (clip y-axis to quiescent range)
    ax_lc.scatter(t_q, flux_detrend, s=0.3, alpha=0.4, c='steelblue')
    ax_lc.axhline(1, color='gray', ls='--', alpha=0.3)
    y_lo, y_hi = np.percentile(flux_detrend, [0.5, 99.5])
    y_pad = (y_hi - y_lo) * 0.15
    ax_lc.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax_lc.set_xlabel('BTJD', fontsize=7); ax_lc.set_ylabel('Flux', fontsize=7)
    ax_lc.set_title(f'TIC {tic_id}', fontsize=8)
    ax_lc.tick_params(labelsize=6)

    # Plot: periodogram (whitened, log period axis)
    ax_ls.semilogx(periods * 24 * 60, power, 'k-', lw=0.4)
    ax_ls.axvline(best_period * 24 * 60, color='red', ls='--', alpha=0.7, lw=0.8, label=f'{best_period*24*60:.1f}m')
    if vsx_period:
        ax_ls.axvline(vsx_period * 24 * 60, color='blue', ls=':', alpha=0.7, lw=0.8, label=f'V:{vsx_period*24*60:.0f}m')
    ax_ls.set_xlabel('P (min)', fontsize=7); ax_ls.set_ylabel('Power', fontsize=7)
    ax_ls.set_title(f'FAP={fap:.0e}', fontsize=8)
    ax_ls.legend(fontsize=6, loc='upper right')
    ax_ls.tick_params(labelsize=6)

# Hide unused axes
for j in range(plot_idx + 1, n_grid_rows * n_per_row):
    r, c = j // n_per_row, (j % n_per_row) * 2
    if r < n_grid_rows:
        axes[r][c].set_visible(False)
        axes[r][c + 1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'period_analysis.png'), dpi=120)
plt.close()
print(f"\n  Plot: {PLOT_DIR}/period_analysis.png")


# --- Phase folding ---
print(f"\n--- Phase folding ---")

# Only phase fold significant L-S detections (FAP < 0.01) or known VSX periods
foldable = {tid: pr for tid, pr in period_results.items()
            if pr['fap'] < 0.01 or pr['vsx_period'] is not None}
n_fold = len(foldable)

if n_fold > 0:
    # Compact grid: 3 candidates per row, each gets 2 columns (L-S fold + VSX fold)
    pf_per_row = 3
    pf_rows = max((n_fold + pf_per_row - 1) // pf_per_row, 1)
    fig2, axes2 = plt.subplots(pf_rows, pf_per_row * 2, figsize=(20, 2.8 * pf_rows), squeeze=False)

    fold_idx = 0
    for tic_id, pr in foldable.items():
        t_q, flux_detrend = detrended_cache[tic_id]

        med_d = np.nanmedian(flux_detrend)
        std_d = np.nanstd(flux_detrend)
        clip = np.abs(flux_detrend - med_d) < 3 * std_d
        t_clip, flux_clip = t_q[clip], flux_detrend[clip]

        vsx_p = pr['vsx_period']
        ls_p = pr['best_period']
        ls_sig = pr['fap'] < 0.01

        fold_list = []
        if ls_sig:
            fold_list.append((ls_p, f"L-S={ls_p*24*60:.1f}m"))
        if vsx_p and vsx_p > 0:
            fold_list.append((vsx_p, f"V={vsx_p*24*60:.1f}m"))

        pf_row = fold_idx // pf_per_row
        pf_col_base = (fold_idx % pf_per_row) * 2

        for col_idx in range(2):
            ax = axes2[pf_row][pf_col_base + col_idx]
            if col_idx >= len(fold_list):
                ax.set_visible(False)
                continue

            fold_period, label = fold_list[col_idx]
            phase = ((t_clip - t_clip[0]) / fold_period) % 1.0

            if len(phase) > 3000:
                rng = np.random.default_rng(42)
                sub = rng.choice(len(phase), 3000, replace=False)
                ax.scatter(phase[sub], flux_clip[sub], s=0.3, alpha=0.06, c='steelblue')
                ax.scatter(phase[sub] + 1, flux_clip[sub], s=0.3, alpha=0.06, c='steelblue')
            else:
                ax.scatter(phase, flux_clip, s=0.5, alpha=0.1, c='steelblue')
                ax.scatter(phase + 1, flux_clip, s=0.5, alpha=0.1, c='steelblue')

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
                ax.fill_between(bc, bm - be, bm + be, color='red', alpha=0.15)
                ax.fill_between(bc + 1, bm - be, bm + be, color='red', alpha=0.15)
                ax.plot(bc, bm, 'r-', lw=1.5)
                ax.plot(bc + 1, bm, 'r-', lw=1.5)

            ax.axhline(1, color='gray', ls='--', alpha=0.3)
            ax.set_xlim(0, 2)
            y_lo, y_hi = np.percentile(flux_clip, [1, 99])
            y_pad = (y_hi - y_lo) * 0.2
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.set_xlabel('Phase', fontsize=7)
            ax.set_ylabel('Flux', fontsize=7)
            ax.set_title(f'{tic_id} {label}', fontsize=7)
            ax.tick_params(labelsize=6)

        labels = []
        if ls_sig:
            labels.append(f"L-S={ls_p*24*60:.1f}m")
        if vsx_p:
            labels.append(f"VSX={vsx_p*24*60:.1f}m")
        print(f"  TIC {tic_id}: folded on {', '.join(labels)}")
        fold_idx += 1

    # Hide unused axes
    for j in range(fold_idx, pf_rows * pf_per_row):
        r, c = j // pf_per_row, (j % pf_per_row) * 2
        if r < pf_rows:
            axes2[r][c].set_visible(False)
            axes2[r][c + 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'phase_folded.png'), dpi=120)
    plt.close()
    print(f"  Plot: {PLOT_DIR}/phase_folded.png")
else:
    print("  No candidates with significant periods or VSX periods to fold")


# Save
with open(os.path.join(DATA_DIR, '06_period_results.pkl'), 'wb') as f:
    pickle.dump(period_results, f)
print(f"\n  Saved period results to {DATA_DIR}/06_period_results.pkl")

# Summary
print(f"\n{'='*70}")
print("PERIOD ANALYSIS SUMMARY")
print(f"{'='*70}")
for tic_id, pr in period_results.items():
    sig = "SIGNIFICANT" if pr['fap'] < 0.01 else "not significant"
    vsx = f"VSX match" if pr['vsx_match'] else f"no VSX match"
    print(f"  TIC {tic_id}: {pr['best_period']*24*60:.1f} min (FAP={pr['fap']:.1e}, {sig}), {vsx}")
