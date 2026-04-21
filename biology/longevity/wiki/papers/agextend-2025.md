---
id: agextend-2025
title: "Discovering geroprotectors through the explainable AI-based platform AgeXtend"
year: 2025
venue: "Nature Aging, Jan 2025"
url: https://www.nature.com/articles/s43587-024-00763-4
pubmed: https://pubmed.ncbi.nlm.nih.gov/39627462/
access: gated
kind: paper
topics: [machine-learning, drug-discovery, interventions, cheminformatics]
---

# AgeXtend

## Summary
Multimodal explainable-AI platform for **geroprotector prediction**
built at IIIT-Delhi. Predicts geroprotective potential, toxicity, and
target proteins/pathways for a candidate molecule.

## Scale
**Screened ~1.1 billion compounds.** Hits were validated in yeast,
C. elegans, and human cell models.

## Sanity checks
Recovered known pro-longevity compounds held out of training, including
**metformin** and **taurine**.

## Scope
- Predicts geroprotective activity
- Assesses toxicity (ADMET-style)
- Identifies target proteins
- Probes natural metabolites from the human microbiome for
  senescence-modulating potential

## Why a CS person should care
Closest analogue to AlphaFold-style scale for anti-aging drug
discovery. The paper is gated but the methods are fully described,
and a public version of the platform is available; a good lab-based
follow-up would be to rebuild the pipeline on open model backbones.

## Related
- [[papers/smerbarreto2023-senolytics]], [[papers/senolytic-predictor-2025]]
- [[papers/network-repurposing-aging]]
- [[papers/clockbase-agent-2025]]
- [[topics/cheminformatics]], [[topics/drug-discovery]]
