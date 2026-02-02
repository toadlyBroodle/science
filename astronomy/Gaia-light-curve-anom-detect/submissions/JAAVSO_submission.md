# Machine Learning Identification of a Dwarf Nova Candidate with Ultra-Short Orbital Period from Gaia DR3

---

## Abstract

We report TIC 22888126 (Gaia DR3 5947829831449229312) as a dwarf nova candidate identified through machine learning analysis of Gaia DR3 variability statistics. TESS Sector 13 photometry reveals a ~2.5 mag outburst with classic dwarf nova morphology. The 57.3-minute period from VSX places this below the CV period gap if confirmed. Spectroscopic follow-up is recommended.

---

## 1. Introduction

The International Variable Star Index (VSX) contains millions of variable stars lacking physical classification. We applied Isolation Forest anomaly detection to Gaia DR3 `vari_summary` statistics to identify unusual objects. TIC 22888126, catalogued as generic "VAR" with P=57.3 min, showed extreme skewness (-3.94) and kurtosis (+18.5), prompting archival investigation.

## 2. Observations

TIC 22888126 (RA=17:55:28.37, Dec=-47:35:34.1, G=16.58) is detected in the ROSAT All-Sky Survey but absent from SIMBAD and CV catalogs. TIC parameters (Teff=4828 K, M=0.78 solar masses, d=1171 pc) likely reflect the donor star. We extracted TESS FFI photometry from Sectors 13, 39, 66, and 93 using 3x3 pixel apertures.

### 2.1 TESS Outburst

Sector 13 (2019 July) captured a dramatic outburst:
- Amplitude: ~2.5 mag (flux ratio ~10x)
- Rise time: <1 day
- Decline time: ~5-7 days

This morphology (rapid rise with exponential decline) is characteristic of dwarf nova disk instability outbursts.

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
| Orbital period | 57.3 min | Yes (below period gap) |
| Quiescent magnitude | G=16.6 | Yes (faint CV) |

### 3.2 Ultra-Short Period Significance

The 57.3-minute period places this system below the cataclysmic variable period gap (~75-115 minutes). CVs in this regime:

- Have evolved past the period minimum (~80 min)
- Contain degenerate or semi-degenerate donors
- Are relatively rare (~150 confirmed systems)
- Include WZ Sge stars, SU UMa stars, and AM CVn candidates

If confirmed spectroscopically, TIC 22888126 would join this scientifically valuable population.

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

1. TIC 22888126 is a strong dwarf nova candidate based on TESS-detected outburst morphology consistent with disk instability model predictions. Spectroscopic confirmation is required for definitive classification.

2. The 57.3-minute orbital period would place this system below the CV period gap, among the scientifically valuable ultra-short period population if confirmed.

3. Machine learning anomaly detection on Gaia DR3 statistics successfully identified this overlooked object from its unusual light curve morphology.

4. Spectroscopic confirmation is essential to:
   - Verify CV nature via emission lines (H-alpha, He II)
   - Measure orbital period precisely via radial velocities
   - Determine donor composition (H-rich vs He-rich AM CVn)
   - Establish definitive subtype classification

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
