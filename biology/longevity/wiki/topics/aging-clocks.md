---
id: topic-aging-clocks
topic: aging-clocks
---

# Aging clocks

Computational models estimating biological age from omics data.

## Three generations (see [[papers/deep-aging-clocks-review-2025]])
1. **1st gen** — trained on chronological age (Horvath, Hannum).
2. **2nd gen** — phenotype / mortality aware (PhenoAge, GrimAge,
   DunedinPACE).
3. **3rd gen** — causality-enriched (CausAge, DamAge, AdaptAge).

## Benchmarks
- [[papers/computagebench]] — 66 datasets, 13 models, open code
- [[papers/nc-2025-14clocks]] — 14 clocks × 174 diseases in 18,859
  individuals

## Recent clock variants
- [[papers/epinflammage]] — epigenetic + inflammatory
- [[papers/pathwayage]] — pathway-level interpretable clock
- [[papers/organ-proteomic-clocks-2025]] — 10 organ clocks from plasma
  proteomics
- [[papers/scbayesage-2025]] — Bayesian per-cell scRNA-seq age

## As scoring function for rejuvenation
- [[papers/clockbase-agent-2025]] — clocks as a ranking metric over
  ~43k published interventions
- [[papers/shift-sb000-2025]] — transcriptomic clock as a screen
  objective for single-gene rejuvenators

## Open problems
- Disease-aware mixture-of-experts over existing clocks
- Causal vs. correlational feature disentanglement at scale
- Cross-species transfer
- Longitudinal (rate-of-aging) modeling
