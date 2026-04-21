---
id: topic-drug-repurposing
topic: drug-repurposing
---

# Drug repurposing for aging

Finding existing drugs that can be redeployed against aging hallmarks.

## Current frontier
- [[papers/network-repurposing-aging]] — SHARP pipeline, pAGE metric,
  370 hits against aging hallmarks.
- [[papers/clockbase-agent-2025]] — autonomous AI agents over ~2M
  molecular profiles, 43k perturbations.
- [[papers/agextend-2025]] — explainable-AI geroprotector platform,
  1.1B compounds screened.

## Orthogonal target-class approaches
- [[papers/smerbarreto2023-senolytics]] / [[papers/senolytic-predictor-2025]]
  — senolytics via chemistry-only ML
- [[papers/itp-nia]] — the gold-standard in-vivo validation pipeline
  (UM-HET3 mouse lifespan)

## Building blocks (all public)
- GenAge longevity-gene list
- STRING / BioGRID / HuRI interactome
- LINCS L1000 perturbation signatures (~1M)
- DrugBank / ChEMBL drug-target maps

## Open problems
- Does SHARP network proximity actually predict ITP lifespan extension
  on held-out compounds? (Candidate first project.)
- Combine SHARP with ComputAgeBench-scored transcriptomic shifts.
- Population-stratified drug repurposing on UKB.
