---
id: epinflammage
title: "EpInflammAge: Epigenetic-Inflammatory Clock for Disease-Associated Biological Aging Based on Deep Learning"
year: 2025
venue: "Int. J. Mol. Sci., MDPI"
url: https://www.mdpi.com/1422-0067/26/13/6284
pmc: https://pmc.ncbi.nlm.nih.gov/articles/PMC12249966/
preprint: https://www.biorxiv.org/content/biorxiv/early/2025/03/14/2025.03.11.642648.full.pdf
access: open
kind: paper
topics: [aging-clocks, deep-learning, inflammation]
---

# EpInflammAge

## Summary
Deep-learning aging clock that **combines epigenetic and inflammatory
markers**. Bridges two hallmarks of aging: epigenetic alterations and
immunosenescence.

## Data
- **17,195 healthy participants** across **72 datasets**
- Age range: months → 101 years

## Method
Two-stage deep neural network on tabular data:
1. One DNN per inflammatory marker, predicting levels from DNA
   methylation input.
2. Feature selection: keep only markers with predicted vs. observed
   Pearson r > 0.5; use those to predict biological age.

DNN architectures beat ElasticNet (the default in the field) on tabular
epigenetic data.

## Performance
- MAE ≈ **7 years** on healthy controls.
- Pearson r ≈ **0.85**.
- Competitive vs. **34** existing epigenetic clocks.
- Strong sensitivity across multiple disease categories — trades some
  healthy-control accuracy for disease sensitivity.

## Tooling
Released as an **easy-to-use web tool** that returns an age estimate plus
per-sample inflammatory-marker levels and feature contributions.

## Related
- [[papers/pathwayage]] — another interpretable 2025 clock
- [[papers/computagebench]] — would benchmark this directly
- [[topics/aging-clocks]], [[topics/inflammation]]
