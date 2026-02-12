#!/usr/bin/env python3
"""Step 3: Extract CV-specific features from calibrated XP spectra.

Computes physical features that distinguish CVs from normal stars:
  - Blue/red flux ratio (accretion disc = blue SED)
  - Balmer jump (disc discontinuity at 365nm)
  - H-alpha excess (emission from accretion)
  - Spectral slope
  - UV excess relative to Teff
  - Composite spectrum residual vs single-star model
  - PCA components for unsupervised structure discovery
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR

print("=" * 70)
print("STEP 3: EXTRACT SPECTRAL FEATURES")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '01_sample.csv'))

xp_path = os.path.join(DATA_DIR, '02_xp_spectra.pkl')
if not os.path.exists(xp_path):
    print("  ERROR: 02_xp_spectra.pkl not found. Run step 2 first.")
    raise SystemExit(1)

with open(xp_path, 'rb') as f:
    xp_cache = pickle.load(f)

if len(xp_cache) == 0:
    print("  ERROR: No XP spectra in cache. Install GaiaXPy and rerun step 2:")
    print("    pip install GaiaXPy")
    print("    ./run_all.sh v3 2")
    raise SystemExit(1)

print(f"  {len(df)} sources, {len(xp_cache)} with XP spectra")


def band_flux(wavelength, flux, wl_min, wl_max):
    """Mean flux in a wavelength band."""
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)
    if mask.sum() == 0:
        return np.nan
    return np.nanmean(flux[mask])


def interpolated_continuum(wavelength, flux, target_min, target_max,
                           blue_min, blue_max, red_min, red_max):
    """Interpolate continuum at target wavelength from blue+red sidebands."""
    blue_flux = band_flux(wavelength, flux, blue_min, blue_max)
    red_flux = band_flux(wavelength, flux, red_min, red_max)
    blue_center = (blue_min + blue_max) / 2
    red_center = (red_min + red_max) / 2
    target_center = (target_min + target_max) / 2

    if np.isnan(blue_flux) or np.isnan(red_flux) or blue_center == red_center:
        return np.nan

    # Linear interpolation
    frac = (target_center - blue_center) / (red_center - blue_center)
    return blue_flux + frac * (red_flux - blue_flux)


def blackbody_flux(wavelength_nm, teff):
    """Planck function normalized to peak=1, wavelength in nm."""
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23
    wl_m = wavelength_nm * 1e-9
    with np.errstate(over='ignore', divide='ignore'):
        exponent = h * c / (wl_m * k * teff)
        bb = 1.0 / (wl_m**5 * (np.exp(np.clip(exponent, -500, 500)) - 1))
    bb = bb / np.nanmax(bb)  # normalize to peak=1
    return bb


def extract_features(source_id, teff=None):
    """Extract spectral features for one source."""
    if source_id not in xp_cache:
        return None

    spec = xp_cache[source_id]
    wl = spec['wavelength']
    flux = spec['flux'].copy()

    # Mask non-positive flux
    flux[flux <= 0] = np.nan
    if np.isnan(flux).all():
        return None

    # Normalize flux to median
    med = np.nanmedian(flux)
    if med <= 0:
        return None
    flux_norm = flux / med

    features = {}

    # 1. Blue/red ratio: accretion disc = blue, normal star = red
    blue_band = band_flux(wl, flux, 400, 500)
    red_band = band_flux(wl, flux, 700, 800)
    features['blue_red_ratio'] = blue_band / red_band if red_band > 0 else np.nan

    # 2. Balmer jump: ratio across the Balmer discontinuity at ~365nm
    pre_balmer = band_flux(wl, flux, 345, 365)
    post_balmer = band_flux(wl, flux, 375, 410)
    features['balmer_jump'] = pre_balmer / post_balmer if post_balmer > 0 else np.nan

    # 3. H-alpha excess: flux at H-alpha vs interpolated continuum
    halpha_flux = band_flux(wl, flux, 648, 668)
    halpha_cont = interpolated_continuum(
        wl, flux,
        target_min=648, target_max=668,
        blue_min=600, blue_max=640,
        red_min=680, red_max=720
    )
    if halpha_cont > 0 and not np.isnan(halpha_cont):
        features['halpha_excess'] = halpha_flux / halpha_cont
    else:
        features['halpha_excess'] = np.nan

    # 4. Spectral slope: linear fit to log(flux) vs wavelength
    valid = np.isfinite(flux_norm) & (flux_norm > 0)
    if valid.sum() > 10:
        coeffs = np.polyfit(wl[valid], np.log10(flux_norm[valid]), 1)
        features['spectral_slope'] = coeffs[0]  # negative = blue
    else:
        features['spectral_slope'] = np.nan

    # 5. UV excess: observed blue flux vs expected from Teff
    if teff and teff > 0 and not np.isnan(teff):
        bb = blackbody_flux(wl, teff)
        bb_norm = bb / np.nanmedian(bb[np.isfinite(bb)]) if np.nanmedian(bb[np.isfinite(bb)]) > 0 else bb
        uv_mask = (wl >= 340) & (wl <= 450) & np.isfinite(flux_norm) & np.isfinite(bb_norm)
        if uv_mask.sum() > 3:
            features['uv_excess'] = np.nanmean(flux_norm[uv_mask]) / np.nanmean(bb_norm[uv_mask])
        else:
            features['uv_excess'] = np.nan

        # 6. Teff residual: RMS deviation from single-star blackbody
        all_valid = np.isfinite(flux_norm) & np.isfinite(bb_norm) & (bb_norm > 0)
        if all_valid.sum() > 10:
            # Scale blackbody to match observed in red band (where donor dominates)
            red_mask = all_valid & (wl >= 700) & (wl <= 900)
            if red_mask.sum() > 3:
                scale = np.nanmean(flux_norm[red_mask]) / np.nanmean(bb_norm[red_mask])
                bb_scaled = bb_norm * scale
                residual = flux_norm[all_valid] - bb_scaled[all_valid]
                features['teff_residual_rms'] = np.sqrt(np.nanmean(residual**2))
            else:
                features['teff_residual_rms'] = np.nan
        else:
            features['teff_residual_rms'] = np.nan
    else:
        features['uv_excess'] = np.nan
        features['teff_residual_rms'] = np.nan

    return features


# --- Extract features for all sources ---
print("\n  Extracting spectral features...")
rows = []
for i, (_, row) in enumerate(df.iterrows()):
    sid = int(row['source_id'])
    teff = row.get('teff_gspphot')
    teff = float(teff) if pd.notna(teff) else None

    feats = extract_features(sid, teff)
    if feats:
        feats['source_id'] = sid
        rows.append(feats)

    if (i + 1) % 500 == 0 or i + 1 == len(df):
        print(f"    {i+1}/{len(df)} ({len(rows)} with features)", flush=True)

feat_df = pd.DataFrame(rows)
print(f"\n  Extracted features for {len(feat_df)} / {len(df)} sources")

# --- PCA on normalized spectral shapes ---
print("\n  Computing PCA on spectral shapes...")

# Build spectral matrix: each row = normalized spectrum for one source
spec_matrix = []
spec_ids = []
# Use a common wavelength grid from the first spectrum
sample_wl = next(iter(xp_cache.values()))['wavelength']

for sid in feat_df['source_id']:
    if sid in xp_cache:
        spec = xp_cache[sid]
        flux = spec['flux'].copy()
        flux[flux <= 0] = np.nan
        med = np.nanmedian(flux)
        if med > 0:
            flux_norm = flux / med
            # Replace NaN with 1.0 (median) for PCA
            flux_norm = np.where(np.isfinite(flux_norm), flux_norm, 1.0)
            spec_matrix.append(flux_norm)
            spec_ids.append(sid)

spec_matrix = np.array(spec_matrix)
print(f"    Spectral matrix: {spec_matrix.shape}")

n_components = min(10, spec_matrix.shape[0] - 1, spec_matrix.shape[1])
scaler = StandardScaler()
spec_scaled = scaler.fit_transform(spec_matrix)
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(spec_scaled)

print(f"    Explained variance (first 5): "
      f"{pca.explained_variance_ratio_[:5].round(3)}")

# Add PCA components to feature dataframe
pca_df = pd.DataFrame(
    {f'spec_pca_{j}': pca_result[:, j] for j in range(n_components)},
    index=range(len(spec_ids))
)
pca_df['source_id'] = spec_ids
feat_df = feat_df.merge(pca_df, on='source_id', how='left')

# --- Save ---
out = os.path.join(DATA_DIR, '03_spectral_features.csv')
feat_df.to_csv(out, index=False)
print(f"\n  Saved {len(feat_df)} rows to {out}")

# Summary statistics
print(f"\n  Feature summary:")
for col in ['blue_red_ratio', 'balmer_jump', 'halpha_excess',
            'spectral_slope', 'uv_excess', 'teff_residual_rms']:
    vals = feat_df[col].dropna()
    if len(vals) > 0:
        print(f"    {col:>20}: median={vals.median():.4f}, "
              f"std={vals.std():.4f}, N={len(vals)}")
