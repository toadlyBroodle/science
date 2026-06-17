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
- [[papers/plasma-proteomics-brain-immune-2025]] — UKB plasma proteomic
  ages, 11 organs, brain and immune are the load-bearing predictors
- [[papers/scbayesage-2025]] — Bayesian per-cell scRNA-seq age
- [[papers/scageclock-2026]] — gated multi-head attention NN over 16M
  single cells, 40+ tissues, 400+ cell types
- [[papers/scimmuaging-immune-clocks-2025]] — per-cell-type immune
  clocks, 1,081 donors
- [[papers/spatial-aging-clocks-brain-2024]] — first spatial aging
  clock; cell-proximity effects in mouse brain
- [[papers/plasma-cellular-aging-proteomic-clock-2026]] — cell-type-
  resolved plasma proteomic ages of 40+ cell types in 60,542 people

## Cross-species
- [[papers/universal-transcriptomic-aging-clock-2026]] — conserved
  transcriptomic age and mortality clocks across mouse, rat, macaque,
  human; universal aging genes (GPNMB, CDKN1A, LGALS3)

## Foundation-model framing
- [[papers/longevity-llm-2026]] — Qwen3-14B fine-tuned, beats Horvath
  on epigenetic-age MAE
- [[papers/longevity-bench-2026]] — companion benchmark suite

## As scoring function for rejuvenation
- [[papers/clockbase-agent-2025]] — clocks as a ranking metric over
  ~43k published interventions
- [[papers/shift-sb000-2025]] — transcriptomic clock as a screen
  objective for single-gene rejuvenators
- [[papers/x-atlas-orion-perturbseq-2025]] — 8M-cell Perturb-seq atlas
  to feed clock-as-objective screens

## Open problems
- Disease-aware mixture-of-experts over existing clocks
- Causal vs. correlational feature disentanglement at scale
- Cross-species transfer
- Longitudinal (rate-of-aging) modeling
