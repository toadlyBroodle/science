---
id: organ-proteomic-clocks-2025
title: "Organ-specific proteomic aging clocks predict disease and longevity across diverse populations"
year: 2025
venue: "Nature Aging"
url: https://www.nature.com/articles/s43587-025-01016-8
pubmed: https://pubmed.ncbi.nlm.nih.gov/41299092/
access: gated
kind: paper
topics: [aging-clocks, proteomics, organ-specific, uk-biobank]
---

# Organ-specific proteomic aging clocks

## Summary
11 clocks (organismal + **10 organ-specific**) trained on UK Biobank
plasma proteomics and externally validated in Chinese and US cohorts.

## Data
- **UK Biobank** training: **n = 43,616** (Olink antibody proteomics)
- External: **China Kadoorie Biobank (n = 3,977)**, **Nurses' Health
  Study (n = 800)**

## Methods
Nonlinear ML (tree ensembles / DL) to regress chronological age from
organ-annotated protein panels; accelerated aging = residual.

## Key findings
- Organ-specific accelerated aging predicts disease onset/progression
  and mortality **beyond** clinical + genetic risk factors.
- **Brain aging is the strongest mortality predictor.**
- Brain + artery clocks link synaptic loss, vascular dysfunction, and
  glial activation to cognitive decline and dementia.
- Brain aging associates with lifestyle, **GABBR1** and **ECM1** genes,
  and brain structure.
- Positively linked diseases: hypertension, MI, stroke, COPD, CKD, CLD,
  T2D, neurodegeneration.

## Why a CS person should care
UK Biobank access is gated but approvable. Once in, organ-cross-talk
graph models / causal inference on the proteomic panel is open territory.

## Related
- [[papers/ukb-nmr-metabolomic-2024]] — sibling UKB study on metabolomics
- [[topics/uk-biobank]], [[topics/proteomics]]
