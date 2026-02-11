# RNAAS Submission: TIC 22888126 Dwarf Nova Candidate

**Title:** Characterization of TIC 22888126 as an Active Dwarf Nova Candidate from TESS Photometry

**Authors:** Landon Mutch; Opus 4.5

**Word Count:** ~950 (limit: 1000)

---

## Abstract

We characterize TIC 22888126 (Gaia DR3 5947829831449228800) as an active dwarf nova candidate based on TESS photometry revealing seven distinct outbursts across 6 years. A candidate ~90-min photometric period is detected at high significance (FAP = 2.5e-85). The Gaia/VSX 57.3-min period is not recovered. Spectroscopic follow-up is recommended.

---

## Main Text

The International Variable Star Index (VSX) contains millions of variable stars lacking physical classification. We applied Isolation Forest anomaly detection to Gaia DR3 `vari_summary` statistics to identify unusual objects. TIC 22888126, catalogued as generic "VAR" with P=57.3 min, showed extreme skewness (−3.94) and kurtosis (+18.5), prompting archival investigation.

**Observations.** TIC 22888126 (RA=17:55:28.37, Dec=−47:35:34.1, G=16.58) is detected in the ROSAT All-Sky Survey but absent from SIMBAD and CV catalogs. TIC parameters (Teff=4828 K, M=0.78 M☉, d=1171 pc) likely reflect the donor star. We extracted TESS FFI photometry from Sectors 13, 39, 66, and 93 using `lightkurve` with threshold aperture masks and median background subtraction.

**TESS Outbursts.** Seven distinct outbursts are detected across 6 years. Amplitudes range from 1--5 mag with classic dwarf nova morphology (rapid rise, exponential decline). Sector 93 shows two large-amplitude events (~5 mag) suggestive of superoutbursts. The ~2-week recurrence during active states is typical of short-period dwarf novae.

**Period Analysis.** A Lomb-Scargle analysis of ~23,500 background-subtracted quiescent data points finds a candidate period of ~90 min (FAP = 2.5e-85), consistent across Sectors 39, 66, and 93. The Gaia/VSX 57.3-min period is not recovered (FAP = 1) and is likely a sampling alias. If the 90-min period reflects the orbital period, this system lies at the upper edge of the CV period gap (~75--130 min).

**Conclusions.** TIC 22888126 is a strong dwarf nova candidate based on photometric evidence. Spectroscopic confirmation is essential to verify CV nature, confirm the orbital period, and determine donor composition. This case demonstrates machine learning's utility for recovering misclassified objects from large surveys.

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

**Keywords:** cataclysmic variables, dwarf novae, machine learning, Gaia, TESS, lightkurve
