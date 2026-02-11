# Characterization of TIC 22888126 as an Active Dwarf Nova Candidate from TESS Photometry

---

## Abstract

We characterize TIC 22888126 (Gaia DR3 5947829831449228800) as an active dwarf nova candidate based on TESS photometry revealing seven distinct outbursts across 6 years (Sectors 13, 39, 66, 93). Outburst amplitudes range from 1--5 mag with classic dwarf nova morphology. A Lomb-Scargle analysis of background-subtracted quiescent data finds a candidate ~90-min photometric period (FAP = 2.5e-85). The Gaia/VSX 57.3-min period is not recovered (FAP = 1). Spectroscopic follow-up is recommended to confirm the orbital period and CV nature.

---

## 1. Introduction

The International Variable Star Index (VSX) contains millions of variable stars lacking physical classification. We applied Isolation Forest anomaly detection to Gaia DR3 `vari_summary` statistics to identify unusual objects. TIC 22888126, catalogued as generic "VAR" with P=57.3 min, showed extreme skewness (-3.94) and kurtosis (+18.5), prompting archival investigation.

## 2. Observations

TIC 22888126 (RA=17:55:28.37, Dec=-47:35:34.1, G=16.58) is detected in the ROSAT All-Sky Survey but absent from SIMBAD and CV catalogs. TIC parameters (Teff=4828 K, M=0.78 solar masses, d=1171 pc) likely reflect the donor star. We extracted TESS FFI photometry from Sectors 13, 39, 66, and 93 using lightkurve with threshold aperture masks and median background subtraction.

### 2.1 TESS Outbursts

Seven distinct outbursts are detected across 6 years of TESS coverage. Amplitudes range from 1--5 mag with classic dwarf nova morphology (rapid rise, exponential decline). Sector 93 shows two large-amplitude events (~5 mag) suggestive of superoutbursts. The ~2-week recurrence during active states is typical of short-period dwarf novae.

![TESS Light Curve](figs/tess_ffi_tic22888126.png)
*Figure 1: TESS Sector 13 FFI photometry of TIC 22888126 showing a ~2.5 magnitude outburst consistent with dwarf nova behavior. Red points indicate measurements >3 sigma above quiescence.*

### 2.2 Multi-wavelength Detections

| Catalog | Detection |
|---------|-----------|
| Gaia DR3 | G = 16.58, classified as VAR |
| TIC v8.2 | TIC 22888126, Teff = 4828 K |
| ROSAT | X-ray source detected |
| 2MASS | J = 15.03 |
| AllWISE | W1 = 14.19 |
| VSX | P = 0.0398 d (57.3 min) |

## 3. Classification

### 3.1 Evidence Supporting Dwarf Nova Classification

| Evidence | Observation | DN Consistent? |
|----------|-------------|----------------|
| Outburst amplitude | ~2.5 mag | Yes (typical 2-6 mag) |
| Rise time | <1 day | Yes (fast rise expected) |
| Decline time | ~5-7 days | Yes (thermal timescale) |
| X-ray emission | ROSAT detected | Yes (boundary layer) |
| Candidate period | ~90 min (TESS L-S) | Yes (period gap boundary) |
| Quiescent magnitude | G=16.6 | Yes (faint CV) |

### 3.2 Period Analysis

The Gaia/VSX 57.3-min period is **not recovered** in background-subtracted TESS data (FAP = 1.0) and is likely a sampling alias from ~31 sparse Gaia epochs.

A candidate photometric period of **~90 min** is detected at high significance (FAP = 2.5e-85) in the combined quiescent data, and independently in Sectors 39, 66, and 93. If confirmed as the orbital period, this places TIC 22888126 at the upper edge of the CV period gap (~75--130 min; Knigge et al. 2011), consistent with an SU UMa-type system re-emerging from the period gap.

## 4. Machine Learning Method

We queried 300 classified variable stars from Gaia DR3 `vari_summary` and extracted light curve statistics: mean magnitude, standard deviation, amplitude (range), skewness, and kurtosis. An Isolation Forest algorithm (scikit-learn; contamination=0.05) identified outliers based on these features.

TIC 22888126 was flagged with extreme values:

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Skewness | -3.94 | Strong negative (brief brightenings) |
| Kurtosis | +18.5 | Extremely peaked (impulsive events) |
| Amplitude | 0.07 mag | Low in Gaia (sparse sampling missed outburst) |

This morphology (negative skewness with high kurtosis) is characteristic of dwarf nova outbursts sampled at low cadence.

![Anomaly Detection Results](figs/anomaly_detection_results.png)
*Figure 2: Isolation Forest anomaly detection results. Red points indicate flagged anomalies. TIC 22888126 shows extreme negative skewness and high kurtosis.*

![Discovery Summary](figs/discovery_summary.png)
*Figure 3: Summary of TIC 22888126 characterization. Top left: Variability amplitude comparison. Top right: Light curve shape space. Bottom left: Simulated light curve patterns. Bottom right: Discovery statistics.*

## 5. Why Was This Object Overlooked?

Several factors contributed to this dwarf nova candidate remaining uncharacterized:

1. Faint quiescent magnitude (G=16.6): Below threshold for many surveys
2. Southern declination (-47 deg): Less coverage by northern facilities
3. Generic VSX classification: "VAR" prompted no follow-up
4. No SIMBAD entry: Not cross-matched to other catalogs
5. Gaia period restriction: Eclipsing binary catalog excludes P<0.2 days
6. No dedicated TESS light curve: Required manual FFI extraction
7. Outburst timing: Sector 13 (July 2019) predates systematic transient monitoring

## 6. Conclusions

1. TIC 22888126 is a strong dwarf nova candidate based on seven TESS-detected outbursts with classic disk instability morphology. Spectroscopic confirmation is required.

2. A candidate ~90-min photometric period is detected at high significance across three TESS sectors, placing this system at the upper edge of the CV period gap if confirmed.

3. The Gaia/VSX 57.3-min period is spurious -- not recovered in background-subtracted TESS data.

4. Machine learning anomaly detection on Gaia DR3 statistics successfully identified this overlooked object.

5. Spectroscopic confirmation is essential to:
   - Verify CV nature via emission lines (H-alpha, He II)
   - Confirm the orbital period via radial velocities
   - Determine donor composition and subtype (SU UMa vs other)

## 7. Recommended Follow-Up

| Observation | Purpose | Priority |
|-------------|---------|----------|
| Optical spectroscopy | Confirm CV (H/He emission) | High |
| High-speed photometry | Detect superhumps, eclipses | High |
| UV photometry (Swift) | Characterize WD | Medium |
| Monitor for next outburst | Determine recurrence time | Ongoing |

---

## Acknowledgments

This research made use of data from:
- ESA Gaia mission (Gaia DR3)
- NASA TESS mission (Sectors 13, 39, 66, 93)
- AAVSO International Variable Star Index (VSX)
- ROSAT All-Sky Survey
- VizieR catalogue access tool (CDS, Strasbourg)

Computational resources provided by Google Colab.

Project inspiration from xAI Grok. Analysis assistance provided by Anthropic Claude Opus 4.5.

Machine learning analysis used scikit-learn.

---

## Data Availability

Full analysis notebook and figures available at:
https://github.com/toadlyBroodle/science/tree/main/astronomy/Gaia-light-curve-anom-detect
