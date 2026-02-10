#!/usr/bin/env python3
"""Step 2: Feature engineering + dual ML anomaly detection."""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from config import CONFIG, DATA_DIR, PLOT_DIR

os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 2: ANOMALY DETECTION")
print("=" * 70)

df = pd.read_csv(os.path.join(DATA_DIR, '01_gaia_sources.csv'))
print(f"  Loaded {len(df)} sources")

# --- Feature engineering ---
df['amplitude'] = df['range_mag_g_fov']
df['abs_mag_g'] = df['phot_g_mean_mag'] + 5 * np.log10(df['parallax'] / 100)
df['rel_std'] = df['std_dev_mag_g_fov'] / df['mean_mag_g_fov']
df['proper_motion'] = np.sqrt(df['pmra']**2 + df['pmdec']**2)

features = ['amplitude', 'std_dev_mag_g_fov', 'skewness_mag_g_fov',
            'kurtosis_mag_g_fov', 'bp_rp', 'abs_mag_g', 'rel_std', 'proper_motion']

df_clean = df.dropna(subset=features).copy()
print(f"  After cleaning: {len(df_clean)} sources ({len(df) - len(df_clean)} dropped)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# --- Isolation Forest ---
contamination = CONFIG['anomaly_percentile'] / 100
print(f"\n  Training Isolation Forest (contamination={contamination})...")
iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
iso_labels = iso.fit_predict(X_scaled)
iso_scores = -iso.score_samples(X_scaled)
n_iso = (iso_labels == -1).sum()
print(f"    Anomalies: {n_iso}")

# --- One-Class SVM ---
print(f"  Training One-Class SVM...")
svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
svm_labels = svm.fit_predict(X_scaled)
svm_scores = -svm.score_samples(X_scaled)
n_svm = (svm_labels == -1).sum()
print(f"    Anomalies: {n_svm}")

# --- Combined scoring ---
iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
svm_norm = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min())
df_clean['combined_score'] = (iso_norm + svm_norm) / 2
df_clean['iso_anomaly'] = iso_labels == -1
df_clean['svm_anomaly'] = svm_labels == -1
df_clean['both_anomaly'] = (iso_labels == -1) & (svm_labels == -1)
n_both = df_clean['both_anomaly'].sum()
print(f"  Consensus (both models): {n_both}")

# --- Select candidates ---
threshold = CONFIG['anomaly_score_min']
if CONFIG['require_consensus']:
    mask = (df_clean['combined_score'] >= threshold) & df_clean['both_anomaly']
else:
    mask = df_clean['combined_score'] >= threshold

candidates = df_clean[mask].sort_values('combined_score', ascending=False).copy()
print(f"\n  Selected: {len(candidates)} candidates (score >= {threshold}, consensus={CONFIG['require_consensus']})")
print(f"  Score range: {candidates['combined_score'].min():.3f} - {candidates['combined_score'].max():.3f}")

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
ax.scatter(df_clean['bp_rp'], df_clean['abs_mag_g'], c=df_clean['combined_score'],
           cmap='viridis', s=5, alpha=0.5)
ax.scatter(candidates['bp_rp'], candidates['abs_mag_g'], c='red', s=50, marker='*', zorder=10)
ax.invert_yaxis()
ax.set_xlabel('BP-RP'); ax.set_ylabel('Abs G mag'); ax.set_title('CMD')

ax = axes[0, 1]
ax.scatter(df_clean['amplitude'], df_clean['skewness_mag_g_fov'],
           c=df_clean['combined_score'], cmap='viridis', s=5, alpha=0.5)
ax.scatter(candidates['amplitude'], candidates['skewness_mag_g_fov'], c='red', s=50, marker='*', zorder=10)
ax.set_xlabel('Amplitude'); ax.set_ylabel('Skewness'); ax.set_title('Amplitude vs Skewness')

ax = axes[1, 0]
ax.scatter(df_clean['skewness_mag_g_fov'], df_clean['kurtosis_mag_g_fov'],
           c=df_clean['combined_score'], cmap='viridis', s=5, alpha=0.5)
ax.scatter(candidates['skewness_mag_g_fov'], candidates['kurtosis_mag_g_fov'], c='red', s=50, marker='*', zorder=10)
ax.set_xlabel('Skewness'); ax.set_ylabel('Kurtosis'); ax.set_title('Shape')

ax = axes[1, 1]
ax.hist(df_clean['combined_score'], bins=50, alpha=0.7)
ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold ({len(candidates)} sel)')
ax.set_xlabel('Combined Score'); ax.legend(); ax.set_title('Score Distribution')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'anomaly_detection.png'), dpi=150)
plt.close()
print(f"  Plot: {PLOT_DIR}/anomaly_detection.png")

out = os.path.join(DATA_DIR, '02_candidates.csv')
candidates.to_csv(out, index=False)
print(f"  Saved {len(candidates)} candidates to {out}")
