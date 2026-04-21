---
id: smerbarreto2023-senolytics
title: "Discovery of senolytics using machine learning"
authors: "Smer-Barreto et al."
year: 2023
venue: "Nature Communications 14:3445"
url: https://www.nature.com/articles/s41467-023-39120-1
pmc: https://pmc.ncbi.nlm.nih.gov/articles/PMC10257182/
code: https://zenodo.org/records/7870357
access: open
kind: paper
topics: [senolytics, drug-discovery, machine-learning]
---

# Discovery of senolytics using machine learning

## Summary
Demonstrates that an ML model trained on published senolytic data can
surface new bona-fide [[topics/senolytics]] — three hits (**ginkgetin,
periplocin, oleandrin**) validated in human cell lines across multiple
senescence modalities.

## Data
- **58** known senolytics mined from papers + patent
- Diverse non-senolytics from **LOPAC-1280** and **Prestwick FDA-approved-1280**
  libraries
- Structures featurised with **200 physicochemical descriptors** (RDKit)
- Binary labels (senolytic / not)

## Method
Multiple classifiers; reported screens virtually triaged **2,352** test
compounds and scaled predictions to **>800k** molecules via GNNs.

## Impact
~100× cost reduction vs. brute-force phenotypic screening. Proof that
ML handles small-data drug-class discovery gracefully.

## Code / data
[Zenodo 7870357](https://zenodo.org/records/7870357) — ships with
full training set and trained models.

## Related
- [[papers/senolytic-predictor-2025]] — 2025 follow-up using
  MoLFormer embeddings
- [[papers/network-repurposing-aging]]
- [[topics/senolytics]], [[topics/drug-discovery]]
