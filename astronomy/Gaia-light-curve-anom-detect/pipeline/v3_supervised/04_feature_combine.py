#!/usr/bin/env python3
"""Step 4: Combine variability + spectral + CMD features into unified feature matrix.

Merges three feature sets:
  - Gaia variability statistics (from vari_summary)
  - XP spectral features (from step 3)
  - CMD position (absolute magnitude, color)
"""

import os
import numpy as np
import pandas as pd
from config import DATA_DIR

print("=" * 70)
print("STEP 4: COMBINE MULTI-MODAL FEATURES")
print("=" * 70)

# Load data
sample = pd.read_csv(os.path.join(DATA_DIR, '01_sample.csv'))
spec_feats = pd.read_csv(os.path.join(DATA_DIR, '03_spectral_features.csv'))

print(f"  Sample: {len(sample)} sources")
print(f"  Spectral features: {len(spec_feats)} sources")

# Merge on source_id
df = sample.merge(spec_feats, on='source_id', how='inner')
print(f"  After merge: {len(df)} sources with all data")

# --- Define feature groups ---

# Variability features (from Gaia vari_summary)
vari_features = [
    'range_mag_g_fov',           # photometric amplitude
    'std_dev_mag_g_fov',         # scatter
    'skewness_mag_g_fov',        # asymmetry (negative = outbursts)
    'kurtosis_mag_g_fov',        # peakedness (high = impulsive events)
]

# Derived variability features
df['rel_std'] = df['std_dev_mag_g_fov'] / df['mean_mag_g_fov']
df['proper_motion'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)
vari_features += ['rel_std', 'proper_motion']

# CMD features
cmd_features = ['abs_mag_g', 'bp_rp']

# Spectral features (physics-motivated)
physics_spec = [
    'blue_red_ratio',           # accretion disc = blue
    'balmer_jump',              # disc Balmer discontinuity
    'halpha_excess',            # H-alpha emission
    'spectral_slope',           # overall SED slope
    'uv_excess',                # UV excess vs Teff model
    'teff_residual_rms',        # composite spectrum deviation
]

# Spectral PCA components
pca_cols = [c for c in df.columns if c.startswith('spec_pca_')]
spec_features = physics_spec + pca_cols[:5]  # first 5 PCA components

# All combined
all_features = vari_features + cmd_features + spec_features

print(f"\n  Feature groups:")
print(f"    Variability:  {len(vari_features)} features")
print(f"    CMD position: {len(cmd_features)} features")
print(f"    Spectral:     {len(spec_features)} features "
      f"({len(physics_spec)} physics + {min(5, len(pca_cols))} PCA)")
print(f"    Total:        {len(all_features)} features")

# --- Handle NaNs ---
# Report missingness
for feat in all_features:
    n_missing = df[feat].isna().sum()
    if n_missing > 0:
        pct = 100 * n_missing / len(df)
        print(f"    {feat}: {n_missing} missing ({pct:.1f}%)")

# Drop rows with too many missing features (>30%)
max_missing = int(0.3 * len(all_features))
n_missing_per_row = df[all_features].isna().sum(axis=1)
df_clean = df[n_missing_per_row <= max_missing].copy()

# Fill remaining NaNs with column median
for feat in all_features:
    median_val = df_clean[feat].median()
    df_clean[feat] = df_clean[feat].fillna(median_val)

print(f"\n  After cleaning: {len(df_clean)} sources "
      f"(dropped {len(df) - len(df_clean)} with >30% missing)")

# --- Save feature matrix and metadata ---
# Save the feature group definitions for step 5
import json
feature_groups = {
    'variability': vari_features,
    'cmd': cmd_features,
    'spectral': spec_features,
    'all': all_features,
}
with open(os.path.join(DATA_DIR, '04_feature_groups.json'), 'w') as f:
    json.dump(feature_groups, f, indent=2)

df_clean.to_csv(os.path.join(DATA_DIR, '04_combined_features.csv'), index=False)
print(f"\n  Saved {len(df_clean)} sources to 04_combined_features.csv")
print(f"  Saved feature groups to 04_feature_groups.json")
