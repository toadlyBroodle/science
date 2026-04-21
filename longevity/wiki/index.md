---
id: index
title: "Longevity Research — Knowledge Tree"
---

# Longevity Research — Knowledge Tree

Knowledge base of computational entry points into longevity and
reverse-aging research for a CS person. Karpathy-style:
markdown-as-source, wikilinks, linting, and indexed querying.

## Start here
- [[topics/aging-research]] — umbrella, full topic map
- [[topics/biomarkers-of-aging]] / [[topics/aging-clocks]] — how we
  measure aging
- [[topics/reprogramming]] / [[topics/partial-reprogramming]] /
  [[topics/chemical-reprogramming]] — how we reverse it

## Analyses
- [[analysis/promising-reverse-aging]] — tier ranking of current
  reverse-aging technologies (April 2026)

## Highest-leverage projects
1. **Extend / stress-test [[papers/clockbase-agent-2025]]'s agent
   architecture** — the dataset is public, the clocks are swappable,
   and most of the ~43k interventions have not been re-scored with
   newer clocks.
2. **SHARP × ITP validation** — cross-check
   [[papers/network-repurposing-aging]] repurposing hits against mouse
   lifespan data at [[papers/itp-mpd-portal]].
3. **Reimplement [[papers/shift-sb000-2025]]-style single-gene screen
   on a published clock** — optimise a transcriptomic clock over gene
   perturbations (Perturb-seq / LINCS) to find safer rejuvenators.
4. **Biomarkers of Aging Challenge Phase 3** — active competition
   ([[papers/biomarkers-aging-challenge]]).
5. **[[papers/xprize-healthspan]]** — $101M, 7 years, biomarker-measured
   function restoration; team formation underway.

## Topic index

### Measurement
- [[topics/aging-clocks]], [[topics/biomarkers-of-aging]],
  [[topics/disease-prediction]]

### Methods / modelling
- [[topics/deep-learning]], [[topics/machine-learning]],
  [[topics/cheminformatics]], [[topics/network-medicine]],
  [[topics/interpretability]], [[topics/pathway-analysis]],
  [[topics/single-cell]]

### Biology / data domain
- [[topics/hallmarks-of-aging]], [[topics/epigenetics]],
  [[topics/proteomics]], [[topics/metabolomics]],
  [[topics/transcriptomics]], [[topics/inflammation]],
  [[topics/organ-specific]]

### Rejuvenation / age reversal
- [[topics/reprogramming]] (umbrella)
  - [[topics/partial-reprogramming]]
  - [[topics/chemical-reprogramming]]
  - [[topics/yamanaka]]
- [[topics/senolytics]]
- [[topics/parabiosis-blood-factors]]
- [[topics/brain-rejuvenation]]
- [[topics/nad-mitophagy]]
- [[topics/gene-therapy]]
- [[topics/telomeres]]
- [[topics/exosomes-extracellular-vesicles]]
- [[topics/stem-cells]]
- [[topics/immune-rejuvenation]]
- [[topics/mtor]]
- [[topics/caloric-restriction]]

### Interventions / discovery
- [[topics/interventions]], [[topics/drug-repurposing]],
  [[topics/drug-discovery]]

### Cohorts / resources
- [[topics/uk-biobank]], [[topics/itp-mice]], [[topics/datasets]]

### Evaluation
- [[topics/benchmarks]], [[topics/competitions]],
  [[topics/clinical-trials]], [[topics/disease-prediction]],
  [[topics/lifespan]], [[topics/sex-specific]], [[topics/safety]]

### Syntheses
- [[topics/review]]

## All papers
See `wiki/papers/` — one file per source, enumerated in
`sources.json` at the project root.

## Tooling
- `scripts/download.py` — pull sources into `sources/{html,pdf}/`
  (requires unrestricted network egress).
- `scripts/convert.py` — HTML/PDF → markdown in `sources/md/`.
- `scripts/index.py` — build TF-IDF + keyword index over the wiki
  into `wiki/build/`.
- `scripts/lint.py` — check broken wikilinks, missing metadata,
  orphan pages, sources.json ↔ wiki consistency.
