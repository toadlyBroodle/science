# ATel Submission: TIC 22888126 Dwarf Nova Candidate

**Submission URL:** https://www.astronomerstelegram.org/

---

## ATel Draft

**Title:** TESS Detection of Dwarf Nova Candidate Outburst in TIC 22888126

**Authors:** L. Mutch, Opus 4.5

**Wavebands:** Optical, X-ray, Infrared

**Subjects:** Cataclysmic Variables, Transients, Variables

---

We report the identification of TIC 22888126 (Gaia DR3 5947829831449228800) as a dwarf nova candidate based on archival TESS photometry.

**Coordinates (J2000):** RA = 17:55:28.37, Dec = -47:35:34.1  
**Galactic:** l = 344.75 deg, b = -11.04 deg

The object is catalogued in VSX as a generic variable ("VAR") with period P = 0.0398 d (57.3 min) but was not previously classified. It is detected in the ROSAT All-Sky Survey but absent from SIMBAD and major CV catalogs.

**TESS Observations:** We extracted FFI photometry from Sectors 13, 39, 66, and 93 using lightkurve with background subtraction. Seven distinct outbursts are detected across 6 years:

- Amplitudes: 1--5 mag with classic dwarf nova morphology
- Rise time: <1 day; Decline time: ~5-7 days
- Sector 93 shows two large-amplitude events (~5 mag, possible superoutbursts)
- ~2-week recurrence during active states

**Period Analysis:** A Lomb-Scargle search of ~23,500 background-subtracted quiescent data points finds a candidate photometric period of ~90 min (FAP = 2.5e-85), consistent across 3/4 sectors. The VSX/Gaia 57.3-min period is NOT recovered (FAP = 1) and is likely a sampling alias. If the 90-min period is the orbital period, this system lies at the upper edge of the CV period gap (~75-130 min).

**Classification:** The combination of: (1) seven outbursts with dwarf nova morphology and 1--5 mag amplitudes, (2) X-ray detection suggesting accretion, and (3) candidate period at the CV period gap boundary, strongly supports SU UMa-type dwarf nova classification.

**Follow-up Recommended:** Optical spectroscopy to confirm CV nature via emission lines (H-alpha, He II) and radial velocity monitoring to confirm the orbital period.

This object was identified through machine learning (Isolation Forest) anomaly detection applied to Gaia DR3 light curve statistics.

Analysis notebook and figures: https://github.com/toadlyBroodle/science/tree/main/astronomy/Gaia-light-curve-anom-detect

---

## ATel Submission Notes

**Submission Process:**
1. Go to https://www.astronomerstelegram.org/
2. Click "Submit a Telegram"
3. Create account if needed
4. Paste text, select subjects
5. Submit (appears within hours)

**Tips:**
- Keep brief (this is ~300 words, typical for ATel)
- Include coordinates prominently
- State what follow-up is needed
- Link to supporting data

**Expected turnaround:** Same day to 24 hours
