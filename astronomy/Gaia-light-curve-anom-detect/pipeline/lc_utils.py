"""Shared light curve utility functions for the CV Hunter pipeline."""

import numpy as np
from scipy.ndimage import median_filter


def collapse_gaps(t, flux, gap_days=1.0, gap_insert=0.5):
    """Remove large time gaps between sectors. Returns collapsed time, flux,
    sector break positions (collapsed coords), and segment BTJD ranges."""
    sort_idx = np.argsort(t)
    t, flux = t[sort_idx], flux[sort_idx]

    dt = np.diff(t)
    gap_indices = np.where(dt > gap_days)[0]

    # Identify segment boundaries (real BTJD ranges)
    seg_starts = [0] + [idx + 1 for idx in gap_indices]
    seg_ends = [idx for idx in gap_indices] + [len(t) - 1]
    segments = [(t[s], t[e]) for s, e in zip(seg_starts, seg_ends)]

    # Collapse gaps
    t_plot = t - t[0]
    dt_plot = np.diff(t_plot)
    for idx in gap_indices:
        shift = dt_plot[idx] - gap_insert
        t_plot[idx + 1:] -= shift

    break_positions = [t_plot[idx] + gap_insert / 2 for idx in gap_indices]
    return t_plot, flux, break_positions, segments


def plot_lc(ax, t, flux, title, sector_str):
    """Plot light curve with log scale, collapsed gaps, and BTJD annotations."""
    t_plot, flux_plot, breaks, segments = collapse_gaps(t, flux)
    ax.scatter(t_plot, flux_plot, s=1, alpha=0.5, c='steelblue')
    ax.axhline(1, color='gray', ls='--', alpha=0.3)
    for bp in breaks:
        ax.axvline(bp, color='orange', ls=':', alpha=0.4, lw=0.8)
    # Annotate each sector segment with BTJD range at top
    y_top = 0.97
    for j, (btjd_start, btjd_end) in enumerate(segments):
        seg_start_plot = t_plot[0] if j == 0 else breaks[j - 1]
        seg_end_plot = breaks[j] if j < len(breaks) else t_plot[-1]
        x_mid = (seg_start_plot + seg_end_plot) / 2
        ax.text(x_mid, y_top, f'{btjd_start:.0f}-{btjd_end:.0f}',
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=6, color='dimgray', alpha=0.8)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Days (gaps collapsed)')
    ax.set_ylabel('Norm Flux (log)')


def isolate_quiescent(t, flux, sigma_thresh=2.0, buffer_pts=5):
    """Excise entire outburst episodes (not just individual bright points).

    1. Iterative sigma clip to find quiescent baseline
    2. Mark contiguous outburst regions
    3. Expand each region by buffer_pts on each side
    4. Return only quiescent segments with >= 20 points
    """
    sort_idx = np.argsort(t)
    t, flux = t[sort_idx], flux[sort_idx]

    # Find quiescent baseline
    mask = np.ones(len(flux), dtype=bool)
    for _ in range(5):
        med = np.nanmedian(flux[mask])
        std = np.nanstd(flux[mask])
        if std <= 0:
            break
        mask = flux < (med + sigma_thresh * std)

    # Mark outburst points
    outburst = ~mask

    # Expand outburst regions by buffer on each side
    expanded = outburst.copy()
    for i in np.where(outburst)[0]:
        lo = max(0, i - buffer_pts)
        hi = min(len(expanded), i + buffer_pts + 1)
        expanded[lo:hi] = True

    # Also mark any points in small isolated quiescent gaps within outbursts
    # (< 10 points of quiescence between two outburst regions)
    quiescent_runs = []
    in_q = False
    start = 0
    for i in range(len(expanded)):
        if not expanded[i] and not in_q:
            start = i
            in_q = True
        elif expanded[i] and in_q:
            quiescent_runs.append((start, i))
            in_q = False
    if in_q:
        quiescent_runs.append((start, len(expanded)))

    # Remove short quiescent gaps (< 10 pts) - they're mid-outburst
    for start, end in quiescent_runs:
        if end - start < 10:
            expanded[start:end] = True

    quiescent = ~expanded
    return t[quiescent], flux[quiescent], int(np.sum(outburst)), int(np.sum(quiescent))


def mask_tess_systematics(periods, power, tol=0.15):
    """Whiten red noise and mask known TESS systematic periods.

    1. Spectral whitening: divide by running median to flatten red noise slope
    2. Zero out power at known TESS systematics (10, 15, 20, 30 min)
    3. Zero out boundary artifact at max_period

    Whitening is done BEFORE masking to avoid edge artifacts at mask boundaries.
    Returns cleaned, whitened power array (copy).
    """
    # Step 1: Spectral whitening on raw power to remove red noise slope
    win = max(51, len(power) // 20)
    if win % 2 == 0:
        win += 1
    noise_floor = median_filter(power, size=win)
    fallback = np.nanmedian(power[power > 0]) if np.any(power > 0) else 1.0
    noise_floor[noise_floor <= 0] = fallback
    whitened = power / noise_floor

    # Step 2: Zero out known TESS systematics
    systematic_min = [7.5, 10.0, 15.0, 20.0, 30.0]
    systematic_days = [m / (24 * 60) for m in systematic_min]
    for sp in systematic_days:
        bad = np.abs(periods - sp) / sp < tol
        whitened[bad] = 0.0

    # Step 3: Mask the top 2% of the period range (boundary artifact)
    p_max = periods.max()
    whitened[periods > 0.98 * p_max] = 0.0

    return whitened


def detrend(t_q, flux_q):
    """Median-filter detrend quiescent data."""
    sort_idx = np.argsort(t_q)
    t_q, flux_q = t_q[sort_idx], flux_q[sort_idx]
    fw = max(5, len(t_q) // 20)
    if fw % 2 == 0:
        fw += 1
    trend = median_filter(flux_q, size=fw)
    # Avoid division by zero
    trend[trend <= 0] = np.nanmedian(flux_q)
    return t_q, flux_q / trend
