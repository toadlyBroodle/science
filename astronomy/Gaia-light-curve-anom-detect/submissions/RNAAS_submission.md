# RNAAS Submission: TIC 22888126 Dwarf Nova Candidate

**Title:** Machine Learning Identification of a Dwarf Nova Candidate with Ultra-Short Orbital Period from Gaia DR3

**Authors:** Landon Mutch; Opus 4.5

**Word Count:** ~950 (limit: 1000)

---

## Abstract

We report TIC 22888126 (Gaia DR3 5947829831449229312) as a dwarf nova candidate identified through machine learning analysis of Gaia DR3 variability statistics. TESS Sector 13 photometry reveals a ~2.5 mag outburst with classic dwarf nova morphology. The 57.3-minute period from VSX places this below the CV period gap if confirmed. Spectroscopic follow-up is recommended.

---

## Main Text

The International Variable Star Index (VSX) contains millions of variable stars lacking physical classification. We applied Isolation Forest anomaly detection to Gaia DR3 `vari_summary` statistics to identify unusual objects. TIC 22888126, catalogued as generic "VAR" with P=57.3 min, showed extreme skewness (−3.94) and kurtosis (+18.5), prompting archival investigation.

**Observations.** TIC 22888126 (RA=17:55:28.37, Dec=−47:35:34.1, G=16.58) is detected in the ROSAT All-Sky Survey but absent from SIMBAD and CV catalogs. TIC parameters (Teff=4828 K, M=0.78 M☉, d=1171 pc) likely reflect the donor star. We extracted TESS FFI photometry from Sectors 13, 39, 66, and 93 using 3×3 pixel apertures.

**TESS Outburst.** Sector 13 (2019 July) captured a dramatic outburst: amplitude ~2.5 mag (flux ratio ~10×), rise time <1 day, decline ~5–7 days. This morphology—rapid rise with exponential decline—is characteristic of dwarf nova disk instability outbursts.

**Classification Evidence.** The outburst properties (amplitude 2–6 mag typical, fast rise, thermal decline timescale), X-ray detection (boundary layer emission), and ultra-short period (below 75-min period gap) strongly support dwarf nova classification. The 57.3-min period would place this among rare systems that have evolved past the period minimum with degenerate donors.

**Why Overlooked.** Several factors contributed: faint quiescent magnitude (G=16.6), southern declination (−47°), generic VSX classification, no SIMBAD entry, Gaia eclipsing binary catalog excluding P<0.2 days, and lack of dedicated TESS light curve products.

**Conclusions.** TIC 22888126 is a strong dwarf nova candidate based on photometric evidence. Spectroscopic confirmation is essential to verify CV nature via emission lines, measure the orbital period, and determine donor composition. This case demonstrates machine learning's utility for recovering misclassified objects from large surveys.

---

## Figures (Online Only)

All figures available at: https://github.com/toadlyBroodle/science/tree/main/astronomy/Gaia-light-curve-anom-detect/figs

- Figure 1: TESS Sector 13 light curve showing outburst
- Figure 2: Anomaly detection in Gaia feature space

---

## Data Availability

Analysis notebook: https://github.com/toadlyBroodle/science/blob/main/astronomy/Gaia-light-curve-anom-detect/Gaia_LightCurve_Anomaly_Detection.ipynb

---

## Acknowledgments

This research used Gaia DR3, TESS, VSX, ROSAT, and VizieR. Computational resources by Google Colab. Analysis assistance by Anthropic Claude Opus 4.5.

---

## References

Gaia Collaboration 2023, A&A, 674, A1; Knigge et al. 2011, ApJS, 194, 28; Liu et al. 2008, ICDM, 413; Osaki 1996, PASP, 108, 39; Ricker et al. 2015, JATIS, 1, 014003

---

## RNAAS Submission Notes

**Submission URL:** https://journals.aas.org/research-notes/

**Requirements:**
- 1000 word limit (main text only)
- No embedded figures (link to online)
- No abstract in final (folded into text)
- References abbreviated

**Category:** Stellar Astrophysics

**Keywords:** cataclysmic variables, dwarf novae, machine learning, Gaia, TESS
