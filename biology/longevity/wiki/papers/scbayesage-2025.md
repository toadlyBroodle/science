---
id: scbayesage-2025
title: "Determining the age of single cells using scBayesAge"
year: 2025
venue: "bioRxiv 2025.12.04.692166"
url: https://www.biorxiv.org/content/10.64898/2025.12.04.692166v1.full
access: open
kind: paper
topics: [aging-clocks, single-cell, transcriptomics, machine-learning]
---

# scBayesAge

## Summary
Bayesian statistical framework to **predict age per cell** from
single-cell transcriptomic profiles. Applied to [[topics/transcriptomics]]
data from **Tabula Muris Senis** to tease out organ- and cell-type-specific
aging signatures.

## Why this is useful
Aging clocks built on bulk data throw away the cell-type signal — the
very signal you want to preserve when scoring a rejuvenation strategy
that acts differently on different cell types. scBayesAge operationalises
per-cell biological age.

## Why a CS person should care
Directly plug-compatible with:
- [[papers/singular-rejuv-atlas-2024]] — use scBayesAge to score
  SINGULAR-identified master regulators.
- [[papers/clockbase-agent-2025]] — extend the ClockBase ensemble to
  single-cell resolution.

Open method, public data (Tabula Muris Senis is free on Figshare / GCP).

## Related
- [[papers/mesenchymal-drift-cell-2025]], [[papers/shift-sb000-2025]]
- [[topics/aging-clocks]], [[topics/single-cell]]
