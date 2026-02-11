#!/usr/bin/env python3
"""Dedicated quiescent-only period search for TIC 22888126.

Uses lightkurve for proper background-subtracted TESS FFI photometry.
Excises outbursts, detrends, and runs a high-resolution Lomb-Scargle
search focused on the 20-120 min range to test the VSX-reported 57.3-min
period.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import sys, os
import warnings
warnings.filterwarnings('ignore')
import lightkurve as lk

sys.path.insert(0, os.path.dirname(__file__))
from lc_utils import isolate_quiescent, detrend

# ── Target ──────────────────────────────────────────────────────────
TIC = 22888126
VSX_PERIOD_MIN = 57.3
OUTDIR = '../figs/pipeline'
os.makedirs(OUTDIR, exist_ok=True)

# ── 1. Download TESS data with lightkurve ───────────────────────────
print("=" * 70)
print(f"DEDICATED PERIOD SEARCH — TIC {TIC} (lightkurve bkg-sub)")
print("=" * 70)

target = f'TIC {TIC}'

# Try pre-made products first
sr = lk.search_lightcurve(target, mission='TESS', author='any')
print(f"  Pre-made products: {len(sr) if sr else 0}")

# TESScut for full coverage
sr_cut = lk.search_tesscut(target)
print(f"  TESScut sectors: {len(sr_cut) if sr_cut else 0}")

# ── 2. Extract light curves per sector (TESScut with bkg subtraction) ──
all_t, all_flux = [], []
sector_data = {}

for i in range(len(sr_cut)):
    sector = int(str(sr_cut.table['mission'][i]).split()[-1])
    print(f"  Downloading S{sector}...", flush=True)

    try:
        tpf = sr_cut[i].download(cutout_size=11)
    except Exception as e:
        print(f"    S{sector} error: {str(e)[:60]}")
        continue

    if tpf is None:
        continue

    # Threshold aperture mask
    ap_mask = tpf.create_threshold_mask(threshold=5, reference_pixel='center')
    if ap_mask is None or ap_mask.sum() == 0:
        ap_mask = tpf.create_threshold_mask(threshold=3, reference_pixel='center')
    if ap_mask is None or ap_mask.sum() == 0:
        ny, nx = tpf.shape[1], tpf.shape[2]
        cy, cx = ny // 2, nx // 2
        ap_mask = np.zeros((ny, nx), dtype=bool)
        ap_mask[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = True

    # Background subtraction
    bkg_mask = ~ap_mask
    if bkg_mask.sum() > 5:
        bkg_flux = np.nanmedian(tpf.flux.value[:, bkg_mask], axis=1)
        n_ap = ap_mask.sum()
        raw_lc = tpf.flux.value[:, ap_mask].sum(axis=1)
        f_sec = raw_lc - bkg_flux * n_ap
    else:
        f_sec = tpf.flux.value[:, ap_mask].sum(axis=1)

    t_sec = tpf.time.value
    good = np.isfinite(t_sec) & np.isfinite(f_sec) & (f_sec > 0)
    t_sec, f_sec = t_sec[good], f_sec[good]

    if len(t_sec) < 50:
        print(f"    S{sector}: only {len(t_sec)} pts, skipping")
        continue

    # Normalize to median
    med = np.nanmedian(f_sec)
    f_sec = f_sec / med

    print(f"    S{sector}: {len(t_sec)} pts, {ap_mask.sum()}px aperture, "
          f"flux range {f_sec.min():.2f}-{f_sec.max():.2f}")

    sector_data[sector] = (t_sec, f_sec)
    all_t.append(t_sec)
    all_flux.append(f_sec)

t_all = np.concatenate(all_t)
flux_all = np.concatenate(all_flux)
print(f"\n  Total: {len(t_all)} points across {len(sector_data)} sectors")

# ── 3. Isolate quiescent data ───────────────────────────────────────
print("\n  Isolating quiescent data...")
t_q, flux_q, n_out, n_q = isolate_quiescent(t_all, flux_all, sigma_thresh=2.0, buffer_pts=10)
print(f"  Outburst points removed: {n_out}")
print(f"  Quiescent points: {n_q}")

# Also per-sector quiescent
sector_q = {}
for sec, (ts, fs) in sector_data.items():
    tq, fq, no, nq = isolate_quiescent(ts, fs, sigma_thresh=2.0, buffer_pts=10)
    if len(tq) > 50:
        sector_q[sec] = (tq, fq)
        print(f"  Sector {sec}: {nq} quiescent pts (removed {no} outburst)")

# ── 4. Detrend quiescent data ───────────────────────────────────────
print("\n  Detrending...")
t_det, flux_det = detrend(t_q, flux_q)
print(f"  Detrended: {len(t_det)} pts, "
      f"std={np.std(flux_det):.4f}")

# Per-sector detrend
sector_det = {}
for sec, (tq, fq) in sector_q.items():
    td, fd = detrend(tq, fq)
    sector_det[sec] = (td, fd)
    print(f"  Sector {sec}: detrended std={np.std(fd):.4f}")

# ── 5. High-resolution Lomb-Scargle in 20-120 min range ─────────────
print("\n  Running Lomb-Scargle (20-120 min, 20000 frequencies)...")

min_period_days = 20.0 / (24 * 60)   # 20 min
max_period_days = 120.0 / (24 * 60)  # 120 min
n_freq = 20000

freq_grid = np.linspace(1/max_period_days, 1/min_period_days, n_freq)
period_grid = 1.0 / freq_grid
period_min = period_grid * 24 * 60  # to minutes

# Combined quiescent
ls = LombScargle(t_det, flux_det)
power = ls.power(freq_grid)
best_idx = np.argmax(power)
best_period = period_min[best_idx]
best_power = power[best_idx]

# FAP at best period
fap = ls.false_alarm_probability(best_power)
print(f"  Best period: {best_period:.2f} min (power={best_power:.4f}, FAP={fap:.2e})")

# Power at 57.3 min
idx_57 = np.argmin(np.abs(period_min - VSX_PERIOD_MIN))
power_57 = power[idx_57]
fap_57 = ls.false_alarm_probability(power_57)
print(f"  Power at 57.3 min: {power_57:.4f} (FAP={fap_57:.2e})")

# Per-sector analysis
print("\n  Per-sector Lomb-Scargle:")
sector_results = {}
for sec in sorted(sector_det.keys()):
    td, fd = sector_det[sec]
    ls_s = LombScargle(td, fd)
    pow_s = ls_s.power(freq_grid)
    bi = np.argmax(pow_s)
    bp = period_min[bi]
    bpow = pow_s[bi]
    bfap = ls_s.false_alarm_probability(bpow)
    p57 = pow_s[np.argmin(np.abs(period_min - VSX_PERIOD_MIN))]
    fap57_s = ls_s.false_alarm_probability(p57)
    sector_results[sec] = (pow_s, bp, bpow, bfap, p57, fap57_s)
    print(f"    Sector {sec}: best={bp:.1f} min (FAP={bfap:.2e}), "
          f"power@57.3={p57:.4f} (FAP={fap57_s:.2e}), {len(td)} pts")

# ── 6. Phase fold at 57.3 min and at best period ────────────────────
def phase_fold_stats(t, flux, period_days):
    """Phase fold and compute binned statistics."""
    phase = ((t - t[0]) % period_days) / period_days
    # 3-sigma clip
    med, std = np.median(flux), np.std(flux)
    clip = np.abs(flux - med) < 3 * std
    phase_c, flux_c = phase[clip], flux[clip]
    # Bin
    n_bins = 40
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_med = np.zeros(n_bins)
    bin_std = np.zeros(n_bins)
    bin_n = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (phase_c >= bin_edges[i]) & (phase_c < bin_edges[i+1])
        if np.sum(in_bin) > 2:
            bin_med[i] = np.median(flux_c[in_bin])
            bin_std[i] = np.std(flux_c[in_bin]) / np.sqrt(np.sum(in_bin))
            bin_n[i] = np.sum(in_bin)
        else:
            bin_med[i] = np.nan
            bin_std[i] = np.nan
    amplitude = np.nanmax(bin_med) - np.nanmin(bin_med)
    return phase_c, flux_c, bin_centers, bin_med, bin_std, amplitude

vsx_period_days = VSX_PERIOD_MIN / (24 * 60)
best_period_days = best_period / (24 * 60)

ph_57, fl_57, bc_57, bm_57, bs_57, amp_57 = phase_fold_stats(
    t_det, flux_det, vsx_period_days)
ph_best, fl_best, bc_best, bm_best, bs_best, amp_best = phase_fold_stats(
    t_det, flux_det, best_period_days)

print(f"\n  Phase fold at 57.3 min: amplitude = {amp_57:.5f} (normalized flux)")
print(f"  Phase fold at {best_period:.1f} min: amplitude = {amp_best:.5f}")
print(f"  Quiescent scatter (std): {np.std(flux_det):.5f}")
print(f"  Signal-to-noise (57.3 min): {amp_57/np.std(flux_det):.2f}")

# ── 7. Plot ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))

# Row 1: Combined periodogram
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(period_min, power, 'k-', lw=0.5, alpha=0.7)
ax1.axvline(VSX_PERIOD_MIN, color='red', ls='--', lw=1.5, label=f'VSX: {VSX_PERIOD_MIN} min')
ax1.axvline(best_period, color='blue', ls='--', lw=1.5, label=f'L-S best: {best_period:.1f} min')
# Mark FAP levels
fap_levels = ls.false_alarm_level([0.1, 0.01, 0.001])
for fl, lab, ls_style in zip(fap_levels, ['10%', '1%', '0.1%'], [':', '--', '-.']):
    ax1.axhline(fl, color='gray', ls=ls_style, alpha=0.5, lw=0.8, label=f'FAP {lab}')
ax1.set_xlabel('Period (min)')
ax1.set_ylabel('L-S Power')
ax1.set_title(f'TIC {TIC} — Quiescent-Only Periodogram (bkg-subtracted, {len(t_det)} pts)')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_xlim(20, 120)

# Row 2: Phase folds
ax2 = fig.add_subplot(3, 2, 3)
ax2.scatter(ph_57, fl_57, s=1, alpha=0.15, c='steelblue')
ax2.errorbar(bc_57, bm_57, yerr=bs_57, fmt='o-', color='red', ms=4, lw=1.5,
             capsize=2, label='Binned median')
ax2.set_xlabel('Phase')
ax2.set_ylabel('Detrended Flux')
ax2.set_title(f'Phase Fold at VSX Period: {VSX_PERIOD_MIN} min')
p5, p95 = np.nanpercentile(fl_57, [1, 99])
ax2.set_ylim(p5, p95)
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(3, 2, 4)
ax3.scatter(ph_best, fl_best, s=1, alpha=0.15, c='steelblue')
ax3.errorbar(bc_best, bm_best, yerr=bs_best, fmt='o-', color='red', ms=4, lw=1.5,
             capsize=2, label='Binned median')
ax3.set_xlabel('Phase')
ax3.set_ylabel('Detrended Flux')
ax3.set_title(f'Phase Fold at L-S Best: {best_period:.1f} min')
p5, p95 = np.nanpercentile(fl_best, [1, 99])
ax3.set_ylim(p5, p95)
ax3.legend(fontsize=8)

# Row 3: Per-sector periodograms
n_sec = len(sector_results)
colors = plt.cm.tab10(np.linspace(0, 1, max(n_sec, 1)))
ax4 = fig.add_subplot(3, 2, (5, 6))
for i, sec in enumerate(sorted(sector_results.keys())):
    pow_s, bp, bpow, bfap, p57, fap57_s = sector_results[sec]
    ax4.plot(period_min, pow_s, lw=0.6, alpha=0.7, color=colors[i],
             label=f'S{sec} (best={bp:.0f}m, @57={p57:.3f})')
ax4.axvline(VSX_PERIOD_MIN, color='red', ls='--', lw=1.5, alpha=0.7)
ax4.set_xlabel('Period (min)')
ax4.set_ylabel('L-S Power')
ax4.set_title('Per-Sector Periodograms (quiescent, bkg-subtracted)')
ax4.legend(loc='upper right', fontsize=7)
ax4.set_xlim(20, 120)

fig.suptitle(f'TIC {TIC} — Dedicated Quiescent Period Search (lightkurve bkg-sub)', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

outfile = os.path.join(OUTDIR, 'tic22888126_dedicated_period_search.png')
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\n  Saved: {outfile}")

# ── Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  VSX period: {VSX_PERIOD_MIN} min")
print(f"  L-S best (quiescent): {best_period:.2f} min (FAP={fap:.2e})")
print(f"  Power at 57.3 min: {power_57:.4f} (FAP={fap_57:.2e})")
print(f"  Phase fold amplitude at 57.3 min: {amp_57:.5f}")
print(f"  Quiescent scatter: {np.std(flux_det):.5f}")
print(f"  SNR at 57.3 min: {amp_57/np.std(flux_det):.2f}")
if fap_57 < 0.01:
    print("  --> 57.3 min period IS significant (FAP < 1%)")
elif fap_57 < 0.1:
    print("  --> 57.3 min period is MARGINAL (1% < FAP < 10%)")
else:
    print("  --> 57.3 min period is NOT significant (FAP > 10%)")
