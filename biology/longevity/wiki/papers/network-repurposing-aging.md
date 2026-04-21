---
id: network-repurposing-aging
title: "Network-driven discovery of repurposable drugs targeting hallmarks of aging"
year: 2025
venue: "arXiv 2509.03330 / PMC"
url: https://arxiv.org/abs/2509.03330
html: https://arxiv.org/html/2509.03330
pmc: https://pmc.ncbi.nlm.nih.gov/articles/PMC12425021/
access: open
kind: paper
topics: [drug-repurposing, network-medicine, hallmarks-of-aging]
---

# Network-driven drug repurposing for hallmarks of aging

## Summary
Network-medicine framework that embeds **2,358 longevity-associated
genes** onto the human interactome and measures proximity of **6,442
approved / experimental compounds** to each hallmark of aging.

## Method — SHARP (Systematic Hallmark-based Aging Repurposing Pipeline)
1. Hallmark modules = connected subgraphs of longevity genes per hallmark.
2. **Network proximity** of each drug's protein targets to each module.
3. **pAGE** (Pro-Age): transcription-based metric from LINCS-style
   expression data — does the drug shift expression back toward the
   young signature?

## Validation
- Captures **82.4%** of already-clinically-tested anti-aging compounds.
- Captures **90.9%** of mouse-lifespan-extending compounds (i.e., ITP
  hits — see [[papers/itp-nia]]).
- For each captured compound, pAGE direction is consistent with age
  restoration in the relevant hallmark.

## Output
**370 drugs** with significant proximity to ≥1 hallmark of aging.

## Why a CS person should care
The underlying ingredients are all public: interactome (STRING, BioGRID,
HuRI), longevity genes (GenAge), drug-target maps (DrugBank, ChEMBL),
and LINCS L1000 signatures. SHARP is reimplementable in a notebook.
Natural extension: cross-check SHARP+pAGE hits against
[[papers/itp-mpd-portal]] survival — a publishable validation study.

## Related
- [[papers/smerbarreto2023-senolytics]], [[papers/senolytic-predictor-2025]]
  — complementary target-class-specific ML approaches
- [[topics/drug-repurposing]], [[topics/network-medicine]],
  [[topics/hallmarks-of-aging]]
