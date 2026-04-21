---
id: clockbase-agent-2025
title: "Autonomous AI Agents Discover Aging Interventions from Millions of Molecular Profiles (ClockBase Agent)"
authors: "Ying, Tyshkovskiy, Gladyshev et al."
year: 2025
venue: "bioRxiv / PMC"
url: https://pmc.ncbi.nlm.nih.gov/articles/PMC12667862/
biorxiv: https://www.biorxiv.org/content/10.1101/2023.02.28.530532v2
pubmed: https://pubmed.ncbi.nlm.nih.gov/41332661/
access: open
kind: paper
topics: [machine-learning, interventions, aging-clocks, drug-repurposing, benchmarks]
---

# ClockBase Agent

## Summary
Publicly accessible platform that **reanalyses ~2M public molecular
samples** (methylation + RNA-seq, human + mouse) through **>40 aging
clocks** to surface which published perturbations actually modulate
biological age.

## Scale (the headline)
- **2,048,729 samples** total.
  - 230,516 human DNA methylation
  - 1,749 mouse DNA methylation
  - 852,381 human RNA-seq
  - 964,083 mouse RNA-seq
- **43,529 interventions** surveyed (genetic, disease, drug, environment).
- **5,756** statistically likely age-modifying candidates.

## Key empirical findings
- More interventions **accelerate** aging than decelerate it.
- Disease states consistently accelerate biological age.
- **Loss-of-function** genetic perturbations outperform gain-of-function
  for decelerating aging — actionable prior for project design.

## Validation
Ouabain, a top-scoring AI-predicted candidate, was tested in aged mice
and showed reduced frailty progression, decreased neuroinflammation,
and improved cardiac function.

## Why a CS person should care
This is the closest thing in longevity to a full automated research
agent. The data store + reanalysis pipeline is public; you can plug in
your own clock or your own hypothesis and score it against the same
corpus. Direct extension: build a better agent / swap in better clocks.

## Related
- [[papers/network-repurposing-aging]], [[papers/agextend-2025]] —
  complementary AI-driven intervention discovery.
- [[papers/computagebench]] — the clocks that feed this.
- [[topics/machine-learning]], [[topics/drug-repurposing]]
