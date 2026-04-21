---
id: pathwayage
title: "Decoding disease-specific ageing mechanisms through pathway-level epigenetic clock (PathwayAge)"
year: 2025
venue: "eBioMedicine (The Lancet)"
url: https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(25)00273-7/fulltext
sciencedirect: https://www.sciencedirect.com/science/article/pii/S2352396425002737
access: open
kind: paper
topics: [aging-clocks, pathway-analysis, interpretability]
---

# PathwayAge

## Summary
Interpretable epigenetic aging clock that aggregates CpG sites into
**GO / KEGG pathway-level features** instead of modeling isolated CpGs.
Two-stage ML pipeline.

## Data
- Genome-wide DNA methylation: **10,615 individuals / 19 cohorts**
- External validation: **3,413 Han Chinese** participants
- Cross-omics: **3,384** transcriptomic samples

## Performance
| Setting | Rho | MAE (years) |
|---|---|---|
| Cross-validation | 0.977 | 2.350 |
| 15 external blood cohorts | 0.677–0.979 | 2.113–6.837 |
| Chinese population | 0.972 | 2.302 |
| Cross-omics (transcriptomics) | 0.70 | 7.21 |

## Biology recovered
Top implicated pathways: **autophagy, cell adhesion, synaptic signalling,
metabolic regulation**. GO clustering reveals consistent ageing
signatures across neuropsychiatric, immune, metabolic, and cancer
conditions.

## Why a CS person should care
Interpretability first — each feature has a direct biological meaning.
Strong cross-omics generalization hints this pattern is not a
methylation-specific artifact.

## Related
- [[papers/epinflammage]], [[papers/computagebench]]
- [[topics/interpretability]]
