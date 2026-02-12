# Plan: Multi-Modal CV Detection (Gaia Variability + XP Spectra + TESS)

## Goal
Build a new pipeline that combines three data modalities for CV detection:
1. **Gaia variability statistics** (existing) — outburst amplitude, skewness, kurtosis
2. **Gaia XP low-resolution spectra** (new) — blue excess from accretion disc, H-alpha excess
3. **TESS light curve features** (existing) — outburst morphology, period detection

This combination has never been done for CV detection (confirmed via literature review).

## Architecture

New directory: `pipeline/v3_multimodal/` with 6 scripts sharing a config.

### Step 1: `01_query_sample.py` — Sample selection

Query Gaia DR3 for two overlapping populations:
- **CMD bridge sources**: Stars in the main-sequence–white-dwarf bridge region where CVs live (M_G between 4–12, BP-RP between -0.5–1.5, excluding the dense main sequence via a polygon cut). This is the approach of Ranaivomanana et al. (2025) but we add the XP requirement.
- **High-variability sources**: Current vari_summary query (top 5000 by range)
- **Require** `has_xp_continuous = 'True'` for all sources

ADQL query joins `gaia_source` + `vari_summary`, selects sources with XP spectra available.
Target: ~5000–10000 sources. Save source_ids + astrometry + photometry + vari_summary stats.

### Step 2: `02_xp_spectra.py` — Download & calibrate XP spectra

- Install GaiaXPy (`pip install GaiaXPy`)
- Use `gaiaxpy.calibrate(source_ids)` to download and calibrate XP spectra in bulk
- Calibrated output: flux vs wavelength at ~2nm sampling across 336–1020nm
- Cache calibrated spectra to pickle (avoid re-downloading on reruns)
- Batches of 500 source IDs to stay within DataLink limits

### Step 3: `03_spectral_features.py` — Extract CV-specific spectral features

From each calibrated XP spectrum, compute:

| Feature | Wavelength range | CV signature |
|---------|-----------------|--------------|
| `blue_red_ratio` | flux(400–500nm) / flux(700–800nm) | Accretion disc = flat/blue SED |
| `balmer_jump` | flux(370–400nm) / flux(400–430nm) | Disc Balmer discontinuity |
| `halpha_excess` | flux(650–665nm) / interpolated continuum | H-alpha emission from accretion |
| `spectral_slope` | Linear fit slope to log(flux) vs λ | Blue = negative slope |
| `uv_excess` | flux(340–400nm) vs expected from Teff | Hot WD/disc contribution |
| `teff_residual` | RMS of (spectrum - blackbody at Gaia Teff) | Composite spectrum = high residual |

Also compute 5 PCA components of the normalized spectral shape for unsupervised structure discovery.

### Step 4: `04_feature_combine.py` — Merge all feature sets

Combine into a single feature matrix per source:
- **Variability features** (8): amplitude, std_dev, skewness, kurtosis, rel_std, num_epochs, abs_mag, proper_motion
- **Spectral features** (11): 6 physics-motivated + 5 PCA components
- **CMD features** (2): absolute G mag, BP-RP color

Total: ~21 features. StandardScaler normalization. Handle NaNs (drop or impute).

### Step 5: `05_anomaly_detect.py` — Multi-modal anomaly detection

Run Isolation Forest + One-Class SVM on three feature subsets:
- **(A) Variability-only**: The 8 variability features (baseline, replicates v2 pipeline)
- **(B) Spectral-only**: The 11 spectral features (new modality)
- **(C) Combined**: All 21 features (the novel approach)

For each, identify anomalies and compute combined scores.
Cross-match all anomaly sets with known CV catalogs (VSX CV types + SIMBAD CV types) to measure **recall** (fraction of known CVs detected) and **precision**.

Key output: a comparison showing combined > spectral-only > variability-only for CV recovery.

Plot:
- Precision-recall curves for each modality
- 2D UMAP projections of the combined feature space, colored by known CV / non-CV
- Spectral feature distributions for known CVs vs. non-CVs

### Step 6: `06_rank_candidates.py` — Rank and validate novel candidates

- Rank novel (non-catalogued) sources by combined anomaly score
- Cross-match top candidates with ROSAT, GALEX, eROSITA catalogs
- For top ~20 candidates: extract TESS light curves (reuse lightkurve infrastructure from v2)
- Generate publication figures:
  - XP spectra of top candidates overlaid with known CV spectra
  - CMD position of candidates relative to known CV locus
  - TESS light curves for candidates with outburst signatures

## Dependencies

New: `GaiaXPy>=2.1.0`, `umap-learn`
Existing: `astroquery`, `scikit-learn`, `lightkurve`, `matplotlib`, `astropy`, `numpy`, `pandas`

## Files to create

```
pipeline/v3_multimodal/
  config.py          — shared config (sample size, CMD cuts, feature lists)
  01_query_sample.py
  02_xp_spectra.py
  03_spectral_features.py
  04_feature_combine.py
  05_anomaly_detect.py
  06_rank_candidates.py
  data/              — intermediate outputs
```

## What makes this publishable

1. **First multi-modal (variability + spectral) CV detection using Gaia XP** — confirmed novel
2. **Quantitative comparison** showing which modality contributes what to CV identification
3. **Known CV spectral atlas** — calibrated XP spectra for 35+ known CVs from our sample, showing the blue excess signature
4. **Novel candidates** backed by both variability AND spectral evidence — much stronger than variability alone
