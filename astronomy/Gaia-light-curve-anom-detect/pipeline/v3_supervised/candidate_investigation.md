# Deep Investigation: Top 3 CV Candidates from v3 Supervised Pipeline

Date: 2026-02-12
Pipeline: Gaia CV Hunter v3 (supervised classification: Gaia variability + XP spectra + TESS follow-up)

---

## Overview

The v3 supervised pipeline identified 50 novel CV candidates ranked by out-of-fold CV probability. Step 7 (TESS light curve fetch + Lomb-Scargle period analysis) found significant periodic signals (FAP < 0.01) in 5 of the top 20 candidates. The three strongest detections were selected for deep investigation: cross-matching against SIMBAD, VSX, traditional CV catalogs (Downes, Ritter & Kolb), X-ray surveys (ROSAT, eROSITA, XMM-Newton), UV surveys (GALEX), the TESS Input Catalog, hot subdwarf catalogs, and the published literature.

### Key Finding

All three candidates have been **misclassified or overlooked** by existing surveys. One (Gaia DR3 3041858605408672128) is a genuinely novel CV identification with eROSITA X-ray confirmation. The other two were auto-classified as CVs by the Gaia DR3 variability pipeline but never individually studied, and one is actively misclassified as a YSO in VSX due to photometric blending.

---

## Candidate 1: Gaia DR3 2892633608838499712

### Identity

| Property | Value |
|----------|-------|
| Gaia DR3 source_id | 2892633608838499712 |
| RA, Dec (J2000) | 06:20:53.80, -32:46:44.0 (95.2242, -32.7789) |
| Galactic coords | l=240.3, b=-20.3 (Canis Major) |
| TIC ID | 124920928 |
| 2MASS | J06205379-3246442 |
| G mag | 17.02 |
| BP-RP | 0.39 (blue) |
| Parallax | 0.736 +/- 0.043 mas |
| Distance | ~1360 pc |
| Abs G | 6.35 |
| RUWE | 1.17 |

### TESS Period Detection

| Property | Value |
|----------|-------|
| TESS sectors | 87 |
| Data source | SPOC |
| N data points | 14,743 |
| Best period | 209.2 min (0.1452 d) |
| FAP | 3.12e-157 |
| Amplitude | 21.1% |
| N quiescent | 12,680 |
| N outburst | 464 |

The 209.2-minute (~3.49 hr) period is above the CV period gap (2-3 hr), consistent with an intermediate polar or AM Her system. The 21% amplitude sinusoidal modulation is visible by eye in the raw TESS light curve. 464 points (~3.2%) flagged as outbursts suggest dwarf nova activity.

### Pipeline Scores

- CV probability (out-of-fold): 0.957
- Rank: #5 of 50 candidates
- In CMD bridge: yes
- Blue/red spectral ratio: 2.43 (very blue, accretion disc signature)
- H-alpha excess: 1.20 (elevated emission)
- Balmer jump: 1.25 (disc contribution)
- Gaia variability range: 3.50 mag

### Cross-Match Results

| Catalog | Result |
|---------|--------|
| SIMBAD | "Star" (blue notation). No CV type. 2 refs. |
| VSX | **Listed as type "CV"** (Gaia DR3 designation used as name) |
| Gaia DR3 vari_classifier | **best_class = CV** |
| Downes CV catalog | Not present |
| Ritter & Kolb catalog | Not present |
| ROSAT (2RXS, RASS-FSC) | Not detected |
| eROSITA (eRASS1) | Not detected |
| XMM-Newton (4XMM-DR13) | Not detected |
| GALEX (GUVcat) | Not detected (FUV and NUV null) |
| SDSS | No coverage |

### Literature

No published papers discuss this source individually. The SIMBAD bibliography lists 2 references:
1. Ranaivomanana et al. 2025 (A&A 693, A268) -- ML analysis of hot subdwarf variability in Gaia DR3. Source may be in extended CDS tables but is not discussed in the paper.
2. A Gaia DR3 variability processing paper (2022).

### Assessment

**Gaia DR3 auto-classified as CV; VSX entry derived from that classification. Never individually studied.**

The Gaia DR3 variability classifier identified this as a CV and VSX ingested the label, but no human has ever examined this object. There is no spectroscopy, no dedicated photometric study, no X-ray detection, and no published characterization. The TESS period detection (209 min, FAP=3e-157) and XP spectral features (blue excess, H-alpha elevation) from this pipeline constitute the **first detailed characterization** of this source.

For publication: must acknowledge existing Gaia/VSX CV label. Framing should be "first characterization" not "new discovery."

---

## Candidate 2: Gaia DR3 5795899250312781056

### Identity

| Property | Value |
|----------|-------|
| Gaia DR3 source_id | 5795899250312781056 |
| RA, Dec (J2000) | 15:27:07.01, -70:51:17.8 (231.7792, -70.8549) |
| Constellation | Apus |
| TIC ID | 263952679 |
| 2MASS | J15270701-7051175 |
| UCAC4 | 096-065997 |
| VSX name | ASASSN-V J152707.02-705117.5 |
| G mag | 17.46 |
| BP-RP | 0.54 (blue) |
| Parallax | 0.758 +/- 0.062 mas |
| Distance | ~1320 pc |
| Abs G | 6.9 |
| RUWE | 0.99 |

### TESS Period Detection

| Property | Value |
|----------|-------|
| TESS sectors | 93 |
| Data source | SPOC |
| N data points | 12,654 |
| Best period | 235.5 min (0.1635 d) |
| FAP | 1.34e-118 |
| Amplitude | 75.9% |
| N quiescent | 6,567 |
| N outburst | 1,906 |

The 75.9% amplitude is far too large for an orbital hump -- this is almost certainly a **deeply eclipsing system**. The 235.5-minute (~3.93 hr) period is above the CV period gap, consistent with an eclipsing dwarf nova (comparable to OY Car, Z Cha). The large fraction of outburst points (1906/12654 = 15%) confirms active dwarf nova behavior.

### Gaia DR3 Variability

| Property | Value |
|----------|-------|
| vari_classifier best_class | **CV (90.3% confidence)** |
| phot_variable_flag | VARIABLE |
| N G-band observations | 49 over 962 days |
| G range | 16.159 -- 19.351 (3.19 mag) |
| G std_dev | 0.786 mag |
| G skewness | +0.79 (positive: bright baseline with deep fades/eclipses) |
| in_vari_eclipsing_binary | False |
| in_vari_compact_companion | False |

### Pipeline Scores

- CV probability (out-of-fold): 0.89
- Rank: #11 of 50 candidates

### Cross-Match Results

| Catalog | Result |
|---------|--------|
| SIMBAD | "Star" only. No CV type. |
| VSX | **ASASSN-V J152707.02-705117.5, type "YSO"** (WRONG -- see below) |
| Gaia DR3 vari_classifier | **best_class = CV (90.3% confidence)** |
| Downes CV catalog | Not present |
| Ritter & Kolb catalog | Not present |
| ROSAT | Not detected |
| eROSITA | Position is in eROSITA-East (l=315); Russian data not public |
| XMM-Newton | Not detected |
| GALEX | Not detected |
| SDSS | No coverage |

### VSX YSO Misclassification: Source Blending

The VSX entry classifies this as a YSO with V = 14.67-15.08 (amplitude 0.41 mag). This is **demonstrably wrong**:

- **Magnitude discrepancy**: VSX V~14.9 vs Gaia G=17.46. The 2.5-mag difference is explained by ASAS-SN aperture blending.
- **Blending source**: Gaia DR3 5795993086759527040 (G=15.27) lies only 16 arcsec away. A second star at G=13.45 is within 24 arcsec. ASAS-SN uses ~8 arcsec pixels; the aperture captures both sources.
- **Diluted variability**: The true 3.2-mag variability (from Gaia) is diluted to 0.41 mag by the dominant flux of the brighter neighbor.
- **Wrong classification follows**: The diluted, irregular-looking light curve was classified as YSO by the ASAS-SN automated pipeline.

This is a textbook case of **photometric blending leading to misclassification in wide-pixel surveys**.

### Literature

No papers discuss this source individually. Not in any published CV catalog, CV candidate list, or eclipsing binary catalog. Gaia's own eclipsing binary pipeline did not flag it (likely due to sparse 49-epoch sampling missing the narrow eclipses).

### Assessment

**Gaia DR3 classified as CV (90% confidence). VSX misclassified as YSO due to ASAS-SN blending. Never individually studied.**

This is the most scientifically interesting candidate for a paper:
1. The 76% amplitude TESS modulation reveals an eclipsing CV that was hidden by photometric blending
2. The VSX YSO misclassification actively prevented this source from being recognized as a CV
3. Gaia's classifier correctly identified it as a CV, but this was overlooked by the community
4. The 3.93-hr period and deep eclipses make it a valuable system for CV parameter measurement (mass ratio, inclination)

For publication: strong narrative of "misclassified by wide-pixel surveys, recovered by Gaia + TESS." Eclipsing CVs are rare and scientifically valuable.

---

## Candidate 3: Gaia DR3 3041858605408672128

### Identity

| Property | Value |
|----------|-------|
| Gaia DR3 source_id | 3041858605408672128 |
| RA, Dec (J2000) | 07:42:55.69, -08:16:16.4 (115.7320, -8.2712) |
| Constellation | Monoceros |
| TIC ID | 95119677 |
| 2MASS | J07425570-0816164 |
| ATLAS designation | ATO J115.7320-08.2712 |
| VSX name | ASASSN-V J074255.70-081616.4 |
| G mag | 16.60 |
| BP-RP | 0.33 (very blue) |
| Parallax | 0.714 +/- 0.056 mas |
| Distance | ~1400 pc (TIC: 1096 pc) |
| Abs G | 5.86 |
| RUWE | 1.00 |

### TESS Period Detection

| Property | Value |
|----------|-------|
| TESS sectors | 34, 61, 88 |
| Data source | **SPOC 2-min cadence** |
| N data points | 49,641 |
| Best period | 90.9 min (0.0631 d) |
| FAP | 5.34e-18 |
| Amplitude | 2.6% |
| N quiescent | 36,945 |
| N outburst | 2,268 |

The 90.9-minute period places this system **below the CV period gap** (~80-130 min boundary), in the territory of short-period CVs (WZ Sge / SU UMa types). The period is detected across 3 independent SPOC sectors. The low amplitude (2.6%) is typical for orbital humps in low-inclination short-period systems. The 2268 outburst points across 3 sectors confirm recurrent brightening episodes.

### Gaia DR3 Variability

| Property | Value |
|----------|-------|
| vari_classifier | **NOT classified as CV** (not in vari_classifier_result for CV) |
| Gaia G range | 4.09 mag |
| G std_dev | 1.286 mag |
| G skewness | +1.83 |
| G kurtosis | +1.76 |
| N epochs | 56 over 888 days |

Despite extreme variability (4.1 mag range), the Gaia ML classifier did NOT flag this as a CV -- a failure of the automated pipeline that our supervised classifier corrects.

### Pipeline Scores

- CV probability (out-of-fold): 0.98 (highest of the three)
- Rank: #2 of 50 candidates
- In CMD bridge: yes
- Blue/red spectral ratio: very blue
- Gaia variability range: 4.09 mag (largest of the three)

### Cross-Match Results

| Catalog | Result |
|---------|--------|
| SIMBAD | **"LPV" (Long Period Variable)** via ATLAS. P=303.7d (spurious). |
| VSX | **ASASSN-V J074255.70-081616.4, type "YSO"** (WRONG) |
| Gaia DR3 vari_classifier | NOT classified as CV |
| Downes CV catalog | Not present |
| Ritter & Kolb catalog | Not present |
| **eROSITA (eRASS1)** | **DETECTED: 1eRASS J074256.4-081623** (12" offset, 7.35 cts) |
| ROSAT | Not detected |
| XMM-Newton | Not detected |
| GALEX | No coverage at this position |
| SDSS | No coverage (outside SDSS footprint) |

### eROSITA X-ray Detection

| Property | Value |
|----------|-------|
| eROSITA designation | 1eRASS J074256.4-081623 |
| Position | RA=115.7351, Dec=-8.2731 |
| Offset from Gaia | ~12 arcsec (within eROSITA positional uncertainty) |
| Band | 0.2-2.3 keV |
| Counts | 7.35 |
| Flux | 7.76e-14 erg/s/cm2 |

X-ray emission is a hallmark of accretion in cataclysmic variables. This detection provides independent physical evidence of the CV nature beyond photometric variability.

### Existing Catalog Entries (All Misclassifications)

1. **SIMBAD: LPV (Long Period Variable)** -- Classification from ATLAS (Heinze et al. 2018) with a spurious period of 303.7 days. A 91-min period system is obviously not an LPV. The ATLAS pipeline was confused by sparse sampling and large-amplitude aperiodic variability.

2. **VSX: YSO (Young Stellar Object)** -- From ASAS-SN automated classification. Same blending/misclassification pattern as Candidate 2: faint blue source with large intrinsic variability, diluted by wide-pixel aperture photometry, classified as irregular YSO.

3. **Hot subdwarf candidate catalogs**:
   - Geier et al. 2019 (A&A 621, A38): In catalog of 39,800 hot subdwarf candidates from Gaia DR2. Photometric selection only.
   - Culpan et al. 2022 (A&A 662, A40): In Gaia EDR3 hot subluminous star catalog. NOT in the "known hot subdwarf" sub-catalog (no spectroscopic confirmation).
   - Barlow et al. 2022 (ApJ 928, 20): In Table 1 of 1,208 candidate variable hot subdwarfs selected from anomalous Gaia flux errors. NOT in their Table 5 of sources actually analyzed with TESS.

All three hot subdwarf catalog entries are based on photometric color/magnitude selection. CVs and hot subdwarfs overlap in color-magnitude space (both are blue and subluminous). Without spectroscopy, the hot subdwarf classification is unconfirmed and almost certainly wrong given the totality of evidence.

### Literature

No published paper identifies this source as a CV. It appears only in:
- Heinze et al. 2018 (ATLAS variable star catalog, as LPV)
- Geier et al. 2019 (hot subdwarf candidate catalog)
- Culpan et al. 2022 (hot subluminous star catalog)
- Barlow et al. 2022 (variable hot subdwarf candidates)
- Ranaivomanana et al. 2025 (SIMBAD reference, likely in extended tables)

### Assessment

**Genuinely novel CV identification. Not classified as CV anywhere. Three existing misclassifications corrected.**

This is the strongest candidate of the three and the lead result for publication:

1. **Novel**: No existing CV classification in any catalog or database
2. **X-ray confirmed**: eROSITA detection provides independent accretion evidence
3. **Short-period**: 90.9 min below the period gap (scientifically valuable regime)
4. **Multi-modal evidence**: Blue color (BP-RP=0.33) + X-ray + TESS period + 4.1 mag outbursts + CMD bridge position + XP spectral features
5. **Triple misclassification corrected**: LPV (SIMBAD/ATLAS), YSO (VSX/ASAS-SN), and hot subdwarf candidate (3 papers) are all wrong
6. **Gold-standard TESS data**: 3 sectors of SPOC 2-min cadence (49.6k points)
7. **Bright enough for spectroscopy**: G=16.6 is accessible to 2-4m class telescopes

---

## Comparative Summary

| Property | Candidate 1 | Candidate 2 | Candidate 3 |
|----------|-------------|-------------|-------------|
| Gaia DR3 ID | 2892633608838499712 | 5795899250312781056 | 3041858605408672128 |
| G mag | 17.02 | 17.46 | 16.60 |
| BP-RP | 0.39 | 0.54 | 0.33 |
| Distance | ~1360 pc | ~1320 pc | ~1400 pc |
| TESS period | 209.2 min | 235.5 min | 90.9 min |
| FAP | 3.1e-157 | 1.3e-118 | 5.3e-18 |
| Amplitude | 21% | 76% | 2.6% |
| TESS source | SPOC (1 sector) | SPOC (1 sector) | SPOC (3 sectors, 2-min) |
| Gaia CV class | Yes | Yes (90%) | **No** |
| VSX type | CV (from Gaia) | **YSO (wrong)** | **YSO (wrong)** |
| SIMBAD type | Star | Star | **LPV (wrong)** |
| X-ray | None | None | **eROSITA detected** |
| In CV catalogs | No | No | No |
| Literature | None | None | In hot subdwarf catalogs |
| Novelty | First characterization | Misclassification corrected | **Genuinely novel** |
| Period regime | Above gap (3.49 hr) | Above gap (3.93 hr) | **Below gap (1.52 hr)** |
| CV subtype (likely) | IP or polar | Eclipsing DN | Short-period DN (SU UMa / WZ Sge) |

---

## Publication Framing

### Lead result
Gaia DR3 3041858605408672128: a novel X-ray-detected short-period CV below the period gap, misclassified as LPV/YSO/hot subdwarf by three independent surveys, identified by multi-modal ML pipeline.

### Supporting results
- Gaia DR3 5795899250312781056: a deeply eclipsing CV (76% amplitude) hidden by ASAS-SN photometric blending, misclassified as YSO in VSX
- Gaia DR3 2892633608838499712: first TESS characterization of a Gaia-classified CV, confirming 3.49 hr period with outburst activity

### Narrative
The multi-modal pipeline (Gaia variability + XP spectra + supervised classification + TESS follow-up) succeeds where individual automated surveys failed:
- ATLAS misclassified the lead candidate as an LPV (wrong period by 4800x)
- ASAS-SN/VSX misclassified two candidates as YSOs (photometric blending)
- Three hot subdwarf catalogs included the lead candidate based on color alone
- Even Gaia's own ML classifier missed the lead candidate
- Only the combination of variability statistics, spectral features, and TESS time-domain analysis correctly identifies all three as CV candidates

### Recommended follow-up
1. **Spectroscopy**: All three targets need optical spectra (H-alpha, He I, He II emission lines) for definitive CV confirmation. G=16.6-17.5 is accessible to 2-4m class telescopes.
2. **Time-resolved photometry**: Candidate 2 (eclipsing) would yield mass ratio, inclination, and white dwarf temperature from eclipse modeling.
3. **X-ray follow-up**: Pointed XMM-Newton or Swift observation of Candidate 3 to characterize the accretion spectrum.

---

## Aladin/ESASky Investigation Links

### Candidate 1: Gaia DR3 2892633608838499712
- Aladin: https://aladin.cds.unistra.fr/AladinLite/?target=95.224157+-32.778836&fov=0.05
- ESASky: https://sky.esa.int/?target=95.224157%20-32.778836&hips=DSS2+color&fov=0.05

### Candidate 2: Gaia DR3 5795899250312781056
- Aladin: https://aladin.cds.unistra.fr/AladinLite/?target=231.779193+-70.854932&fov=0.05
- ESASky: https://sky.esa.int/?target=231.779193%20-70.854932&hips=DSS2+color&fov=0.05

### Candidate 3: Gaia DR3 3041858605408672128
- Aladin: https://aladin.cds.unistra.fr/AladinLite/?target=115.732041+-8.271213&fov=0.05
- ESASky: https://sky.esa.int/?target=115.732041%20-8.271213&hips=DSS2+color&fov=0.05
