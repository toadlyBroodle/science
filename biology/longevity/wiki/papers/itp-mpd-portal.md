---
id: itp-mpd-portal
title: "ITP Data Portal — Mouse Phenome Database"
url: https://phenome.jax.org/projects/ITP1
access: open
kind: resource
topics: [itp-mice, datasets]
---

# Mouse Phenome Database — ITP portal

Public per-cohort portal for the [[papers/itp-nia]]. Per-cohort results,
supplementary files, and metadata for every compound tested since 2004.

## What's there
- Kaplan–Meier tables for each compound/dose/sex combination
- Body-weight trajectories
- Health-span assay data
- Links back to the ITP primary papers

## Downloads
Tabular files are small enough to fit in memory; a simple pandas
reanalysis + mixed-effects survival model is feasible without specialized
infra.

## Related
- [[papers/dr-960-mice-nature-2024]]
- [[papers/itp-sex-specific-2025]]
- [[topics/datasets]], [[topics/itp-mice]]
