---
id: senolytic-predictor-2025
title: "Development and Application of a Senolytic Predictor for Discovery of Novel Senolytic Compounds and Herbs"
year: 2025
venue: "Molecules, MDPI, 30(12):2653"
url: https://www.mdpi.com/1420-3049/30/12/2653
pmc: https://pmc.ncbi.nlm.nih.gov/articles/PMC12196162/
access: open
kind: paper
topics: [senolytics, drug-discovery, cheminformatics]
---

# Senolytic Predictor (2025)

## Summary
Follow-up to [[papers/smerbarreto2023-senolytics]] with a bigger, cleaner
training set and modern molecular representations.

## Data
- **111 positive** + **3,951 negative** compounds, curated from literature
- Featurisation: classical fingerprints, descriptors, and
  **MoLFormer** molecular embeddings

## Models
- Support Vector Machine **AUC 0.998** (F1 0.948)
- Multilayer Perceptron **AUC 0.997** (F1 0.941)
- MoLFormer embeddings outperform classical fingerprints.

## Virtual screens
- **DrugBank** → **98** structurally novel candidates
- **TCMbank** (traditional Chinese medicine) → **714** compounds, **81
  medicinal herbs** with possible senolytic activity

## Why a CS person should care
Training set is released; reproducing and extending to newer foundation
models (MolE, Uni-Mol2, ChemBERTa-2) is a self-contained project.
Natural product candidate pool is unusually large and unexplored.

## Related
- [[papers/smerbarreto2023-senolytics]]
- [[papers/network-repurposing-aging]]
- [[topics/cheminformatics]], [[topics/senolytics]]
