#!/usr/bin/env python3
"""Step 6: Rank novel CV candidates and generate publication figures.

Ranks non-CV sources by classifier probability, cross-matches top
candidates with X-ray/UV catalogs, and produces figures:
  - XP spectra of top candidates vs known CVs
  - CMD colored by CV probability
  - Feature radar comparing top candidates to known CV median
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from config import CONFIG, DATA_DIR, PLOT_DIR

TIMEOUT = CONFIG['query_timeout']
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 6: RANK CANDIDATES & PUBLICATION FIGURES")
print("=" * 70)

# --- Load data ---
df = pd.read_csv(os.path.join(DATA_DIR, '05_classification_results.csv'))

# Load feature importances for radar plot
feat_imp = pd.read_csv(os.path.join(DATA_DIR, '05_feature_importances.csv'))

# Load feature groups
with open(os.path.join(DATA_DIR, '04_feature_groups.json')) as f:
    groups = json.load(f)

# Load XP spectra cache
xp_cache = {}
xp_path = os.path.join(DATA_DIR, '02_xp_spectra.pkl')
if os.path.exists(xp_path):
    with open(xp_path, 'rb') as f:
        xp_cache = pickle.load(f)

# Known CVs and novel candidates
known = df[df['is_known_cv'] == True].copy()
novel = df[df['is_known_cv'] == False].copy()
novel = novel.sort_values('cv_probability_oof', ascending=False)

print(f"  {len(df)} total sources")
print(f"  {len(known)} known CVs")
print(f"  {len(novel)} non-CV sources ranked by CV probability (OOF)")

# High-confidence candidates
n_above_50 = (novel['cv_probability_oof'] > 0.5).sum()
n_above_80 = (novel['cv_probability_oof'] > 0.8).sum()
print(f"  Candidates with p_oof > 0.5: {n_above_50}")
print(f"  Candidates with p_oof > 0.8: {n_above_80}")


# --- Cross-match top candidates with ROSAT + GALEX ---
print("\n--- Cross-matching top candidates ---")
top_n = min(50, len(novel))
top = novel.head(top_n).copy()

rosat_flags, galex_fuv, galex_nuv = [], [], []

for i, (_, row) in enumerate(top.iterrows()):
    coord = SkyCoord(ra=row['ra'] * u.deg, dec=row['dec'] * u.deg,
                     frame='icrs')

    # ROSAT X-ray
    xray = False
    for cat in ['IX/10A/1rxs', 'IX/29/rass_fsc']:
        try:
            result = Vizier(timeout=TIMEOUT).query_region(
                coord, radius=CONFIG['rosat_search_radius'] * u.arcsec,
                catalog=cat)
            if result and len(result) > 0 and len(result[0]) > 0:
                xray = True
                break
        except Exception:
            pass
    rosat_flags.append(xray)

    # GALEX UV
    fuv, nuv = None, None
    try:
        result = Vizier(timeout=TIMEOUT).query_region(
            coord, radius=CONFIG['galex_search_radius'] * u.arcsec,
            catalog='II/335/galex_ais')
        if result and len(result) > 0 and len(result[0]) > 0:
            g = result[0][0]
            fuv = float(g['FUVmag']) if not np.ma.is_masked(
                g.get('FUVmag')) else None
            nuv = float(g['NUVmag']) if not np.ma.is_masked(
                g.get('NUVmag')) else None
    except Exception:
        pass
    galex_fuv.append(fuv)
    galex_nuv.append(nuv)

    if (i + 1) % 10 == 0 or i + 1 == top_n:
        print(f"  {i + 1}/{top_n}", flush=True)

top['has_xray'] = rosat_flags
top['galex_fuv'] = galex_fuv
top['galex_nuv'] = galex_nuv

n_xray = sum(rosat_flags)
n_galex = sum(1 for f in galex_fuv if f is not None)
print(f"\n  X-ray detections: {n_xray}/{top_n}")
print(f"  GALEX detections: {n_galex}/{top_n}")


# --- Print top candidates ---
print(f"\n{'=' * 70}")
print(f"TOP NOVEL CANDIDATES (by OOF CV probability)")
print(f"{'=' * 70}")
print(f"{'Rank':<5} {'Gaia DR3 ID':<22} {'G':>5} {'BP-RP':>6} {'P_oof':>6} "
      f"{'Blue/Red':>8} {'Halpha':>7} {'Xray':>5} {'FUV':>5}")
print("-" * 85)

for rank, (_, row) in enumerate(top.iterrows(), 1):
    sid = int(row['source_id'])
    br = (f"{row['blue_red_ratio']:.3f}"
          if pd.notna(row.get('blue_red_ratio')) else '---')
    ha = (f"{row['halpha_excess']:.3f}"
          if pd.notna(row.get('halpha_excess')) else '---')
    xr = 'Y' if row.get('has_xray', False) else 'N'
    fv = (f"{row['galex_fuv']:.1f}"
          if pd.notna(row.get('galex_fuv')) else '---')
    print(f"{rank:<5} {sid:<22} {row['phot_g_mean_mag']:>5.1f} "
          f"{row['bp_rp']:>6.2f} {row['cv_probability_oof']:>6.3f} "
          f"{br:>8} {ha:>7} {xr:>5} {fv:>5}")

    if rank <= 10:
        print(f"      Aladin: https://aladin.cds.unistra.fr/AladinLite/"
              f"?target={row['ra']:.6f}+{row['dec']:.6f}&fov=0.05")


# --- Figure 1: XP Spectral Atlas ---
print("\n--- Generating figures ---")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
cmap = plt.cm.tab10

# Top panel: known CVs
ax = axes[0]
n_plot = min(10, len(known))
for j, (_, row) in enumerate(known.head(n_plot).iterrows()):
    sid = int(row['source_id'])
    if sid in xp_cache:
        spec = xp_cache[sid]
        wl, flux = spec['wavelength'], spec['flux']
        flux_norm = flux / np.nanmedian(flux[flux > 0])
        ax.plot(wl, flux_norm, alpha=0.6, lw=0.8, color=cmap(j % 10),
                label=f'Gaia {sid} (G={row["phot_g_mean_mag"]:.1f})')
ax.axvline(656.3, color='red', ls=':', alpha=0.5, lw=1, label='H-alpha')
ax.axvline(365, color='blue', ls=':', alpha=0.5, lw=1, label='Balmer jump')
ax.set_xlabel('Wavelength (nm)', fontsize=10)
ax.set_ylabel('Normalized Flux', fontsize=10)
ax.set_title(f'Known CVs -- Calibrated XP Spectra ({n_plot} shown)',
             fontsize=11)
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(330, 1050)
ax.tick_params(labelsize=9)

# Bottom panel: top candidates
ax = axes[1]
n_plot = min(10, len(top))
for j, (_, row) in enumerate(top.head(n_plot).iterrows()):
    sid = int(row['source_id'])
    if sid in xp_cache:
        spec = xp_cache[sid]
        wl, flux = spec['wavelength'], spec['flux']
        flux_norm = flux / np.nanmedian(flux[flux > 0])
        markers = []
        if row.get('has_xray', False):
            markers.append('Xray')
        if pd.notna(row.get('galex_fuv')):
            markers.append('FUV')
        label = (f'Gaia {sid} (G={row["phot_g_mean_mag"]:.1f}, '
                 f'p={row["cv_probability_oof"]:.2f}')
        if markers:
            label += f', {"+".join(markers)}'
        label += ')'
        ax.plot(wl, flux_norm, alpha=0.6, lw=0.8, color=cmap(j % 10),
                label=label)
ax.axvline(656.3, color='red', ls=':', alpha=0.5, lw=1, label='H-alpha')
ax.axvline(365, color='blue', ls=':', alpha=0.5, lw=1, label='Balmer jump')
ax.set_xlabel('Wavelength (nm)', fontsize=10)
ax.set_ylabel('Normalized Flux', fontsize=10)
ax.set_title(f'Top Novel Candidates -- Calibrated XP Spectra ({n_plot} shown)',
             fontsize=11)
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.set_xlim(330, 1050)
ax.tick_params(labelsize=9)

plt.suptitle('CV Hunter v3: XP Spectral Atlas',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xp_spectral_atlas.png'), dpi=150)
plt.close()
print(f"  Saved: {PLOT_DIR}/xp_spectral_atlas.png")


# --- Figure 2: CMD + Feature Comparison ---
fig = plt.figure(figsize=(16, 7))

# (a) CMD colored by CV probability
ax1 = fig.add_subplot(121)
# Background sources
bg = df[~df['is_known_cv']]
sc = ax1.scatter(bg['bp_rp'], bg['abs_mag_g'],
                 c=bg['cv_probability_oof'], cmap='YlOrRd',
                 s=3, alpha=0.5, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax1, label='P(CV) OOF', shrink=0.8)

# Known CVs
if len(known) > 0:
    ax1.scatter(known['bp_rp'], known['abs_mag_g'],
                s=30, edgecolors='blue', facecolors='none', lw=1,
                label=f'Known CVs ({len(known)})', zorder=10)
# Top candidates
ax1.scatter(top['bp_rp'], top['abs_mag_g'],
            s=40, marker='D', edgecolors='lime', facecolors='none', lw=1.2,
            label=f'Top {top_n} candidates', zorder=11)
ax1.invert_yaxis()
ax1.set_xlabel('BP-RP', fontsize=11)
ax1.set_ylabel('Abs G mag', fontsize=11)
ax1.set_title('(a) CMD Colored by CV Probability', fontsize=12,
              fontweight='bold')
ax1.legend(fontsize=9, loc='lower left')

# (b) Feature radar plot: top candidates vs known CV median
ax2 = fig.add_subplot(122, polar=True)

# Select top 8 features by importance
radar_features = feat_imp.head(8)['feature'].tolist()

# Compute percentile-normalized values for each group
feat_mins = df[radar_features].min()
feat_maxs = df[radar_features].max()
feat_range = feat_maxs - feat_mins
feat_range[feat_range == 0] = 1

cv_medians = df.loc[df['is_known_cv'], radar_features].median()
cand_medians = top[radar_features].median()
pop_medians = df[radar_features].median()

cv_norm = ((cv_medians - feat_mins) / feat_range).values
cand_norm = ((cand_medians - feat_mins) / feat_range).values
pop_norm = ((pop_medians - feat_mins) / feat_range).values

# Set up angles
angles = np.linspace(0, 2 * np.pi, len(radar_features),
                     endpoint=False).tolist()
# Close polygons
cv_norm = np.concatenate([cv_norm, [cv_norm[0]]])
cand_norm = np.concatenate([cand_norm, [cand_norm[0]]])
pop_norm = np.concatenate([pop_norm, [pop_norm[0]]])
angles += [angles[0]]

ax2.plot(angles, cv_norm, 'o-', label='Known CVs (median)',
         lw=2, color='blue', markersize=4)
ax2.fill(angles, cv_norm, alpha=0.1, color='blue')
ax2.plot(angles, cand_norm, 's-', label='Top candidates (median)',
         lw=2, color='red', markersize=4)
ax2.fill(angles, cand_norm, alpha=0.1, color='red')
ax2.plot(angles, pop_norm, '--', label='Population (median)',
         lw=1, color='gray', alpha=0.5)

# Shorten feature labels for readability
short_labels = []
for f in radar_features:
    f = f.replace('_', ' ').replace('mag g fov', '').replace('spec pca ', 'PC')
    if len(f) > 15:
        f = f[:14] + '.'
    short_labels.append(f)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(short_labels, fontsize=8)
ax2.set_ylim(0, 1)
ax2.set_title('(b) Feature Comparison (top 8)', fontsize=12,
              fontweight='bold', pad=20)
ax2.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.suptitle('CV Hunter v3: Candidate Analysis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'candidate_analysis.png'), dpi=150)
plt.close()
print(f"  Saved: {PLOT_DIR}/candidate_analysis.png")


# --- Save final candidate table ---
out = os.path.join(DATA_DIR, '06_top_candidates.csv')
top.to_csv(out, index=False)
print(f"\n  Saved {len(top)} top candidates to {out}")

# Save known CVs for reference
known_out = os.path.join(DATA_DIR, '06_known_cv_spectra.csv')
known.to_csv(known_out, index=False)
print(f"  Saved {len(known)} known CVs to {known_out}")

# --- Final summary ---
print(f"\n{'=' * 70}")
print("CANDIDATE SUMMARY")
print(f"{'=' * 70}")
print(f"  Top {top_n} candidates by OOF CV probability")
print(f"  X-ray detections: {n_xray}/{top_n} "
      f"({100 * n_xray / top_n:.0f}%)")
print(f"  GALEX detections: {n_galex}/{top_n} "
      f"({100 * n_galex / top_n:.0f}%)")
print(f"  Median P_oof of top {top_n}: "
      f"{top['cv_probability_oof'].median():.3f}")
print(f"  Median BP-RP of top {top_n}: {top['bp_rp'].median():.2f}")
print(f"  Median G mag of top {top_n}: "
      f"{top['phot_g_mean_mag'].median():.1f}")
