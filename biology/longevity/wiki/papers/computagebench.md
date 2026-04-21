---
id: computagebench
title: "ComputAgeBench: Epigenetic Aging Clocks Benchmark"
authors: "Kriukov et al."
year: 2024
venue: "KDD 2025 / bioRxiv 2024.06.06.597715"
url: https://openreview.net/forum?id=0ApkwFlCxq
biorxiv: https://www.biorxiv.org/content/10.1101/2024.06.06.597715v2
code: https://github.com/ComputationalAgingLab/ComputAge
huggingface: https://huggingface.co/datasets/computage/computage_bench
access: open
kind: paper
topics: [aging-clocks, benchmarks]
---

# ComputAgeBench: Epigenetic Aging Clocks Benchmark

## Summary
First unified framework for benchmarking epigenetic [[topics/aging-clocks]].
Core idea: a good clock must discriminate healthy individuals from those
with aging-accelerating conditions.

## Dataset
- **Benchmark**: 10,404 samples × 900,449 CpGs across 65 studies
- **Training**: 7,419 samples × 907,766 CpGs across 46 studies
- 66 harmonized public blood-DNAm datasets covering **19
  aging-accelerating conditions**
- All on HuggingFace ([computage/computage_bench](https://huggingface.co/datasets/computage/computage_bench))

## What's compared
13 published clock models benchmarked head-to-head (Horvath, Hannum,
PhenoAge, GrimAge, DunedinPACE, etc.).

## Code / Tools
[`ComputationalAgingLab/ComputAge`](https://github.com/ComputationalAgingLab/ComputAge)
— "full-stack aging clocks design and benchmarking." Includes reproducibility
notebook for the paper.

## Why a CS person should care
Drop-in package + harmonized data = you can train a new clock or ablate
an existing one in an afternoon. Benchmark gaps are explicit research
targets.

## Related
- [[papers/nc-2025-14clocks]] — independent 14-clock vs. disease comparison
- [[papers/epinflammage]], [[papers/pathwayage]] — recent clock variants
- [[topics/aging-clocks]], [[topics/benchmarks]]
