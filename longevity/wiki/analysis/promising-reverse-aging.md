---
id: promising-reverse-aging-2026
title: "Most promising reverse-aging technologies (April 2026)"
date: 2026-04-21
kind: analysis
---

# Most promising reverse-aging technologies — April 2026

Ranking by **evidence strength × translational readiness × computational
tractability**. Every claim here is backed by a paper page in this
wiki; chase the wikilinks for primary sources.

## Tier 1 — Strongest evidence, translation already underway

### 1. Partial reprogramming (OSK / OSKM)
The only intervention class with *both* clean single-cell rejuvenation
data ([[papers/mesenchymal-drift-cell-2025]]) *and* a lifespan-level
readout in old mice — [[papers/aav-osk-lifespan-2024]] reports **+109%
remaining lifespan** with AAV-OSK in 124-week-old mice.

Altos Labs, NewLimit, and Retro Biosciences are all converging on this
class. The historical bottleneck — pluripotency / cancer risk — is
actively being solved: [[papers/shift-sb000-2025]] identified **SB000**,
a *single-gene* alternative discovered via ML-clock-directed screens
that rivals OSK(M) potency **without** activating pluripotency pathways.

See also: [[papers/paine-partial-reprog-review-2024]],
[[papers/reprogramming-rejuv-review-ncomms-2024]] for reviews.

### 2. Senolytics in humans
The class has made the jump to real clinical endpoints:

- [[papers/senolytic-mci-ebiomed-2025]] — **STAMINA** pilot dosed
  D+Q against cognition + mobility in pre-Alzheimer's adults.
- [[papers/senolytic-methylation-2024]] — D+Q±fisetin vs. epigenetic
  clocks over 6 months.

The ML discovery pipeline ([[papers/smerbarreto2023-senolytics]],
[[papers/senolytic-predictor-2025]]) is cheap to extend and already
gave 3 validated hits from <60 training positives.

### 3. Rapamycin (mTOR inhibition)
The single most reproducible longevity intervention across species
([[papers/itp-nia]]: +20–25% lifespan in both sexes of UM-HET3 mice).

[[papers/pearl-rapamycin-2025]] — the first large human RCT with
healthspan primary endpoints — showed **safe, dose-dependent
benefits at 1 year** in a normative-aging cohort. Boring but real.

## Tier 2 — Strong mechanism, promising human data, still scaling

### 4. Mitophagy induction (urolithin A, etc.)
[[papers/urolithin-a-immune-2025]] is a clean *Nature Aging* RCT
showing UA expands naive-like CD8+ T cells and increases CD8+
fatty-acid oxidation in humans. Gut-microbial metabolite → shelf-stable
pill. Mechanism is well understood. Part of the broader
[[topics/nad-mitophagy]] axis.

### 5. Young-blood / young-CSF factors
After a decade of parabiosis work, the field has identified real
**single factors** that reproduce much of the young-blood effect:

- [[papers/pedf-parabiosis-2024]] — **PEDF** extends fibroblast
  replicative lifespan and reverses age-related pathology across
  multiple tissues in aged mice.
- [[papers/fgf17-young-csf-2022]] — **FGF17** in young CSF rejuvenates
  aged hippocampus via oligodendrocyte progenitor proliferation.
- [[papers/hcpb-review-2024]] — field overview.

These are now recombinant-protein / AAV-gene-therapy candidates.

### 6. Young stem-cell exosomes
[[papers/adsc-exosomes-2022]] (*Sci Adv*): ADSC-derived small EVs from
young mice **improve healthspan, reduce frailty, lower epigenetic age**
in old mice. Cell-free, shelf-stable — avoids the regulatory complexity
of whole-cell stem-cell therapies.
See [[topics/exosomes-extracellular-vesicles]].

## Tier 3 — Longer-horizon but high-ceiling

### 7. Chemical reprogramming cocktails
[[papers/yang-chemical-cocktails-2023]] (Sinclair lab) demonstrated 6
cocktails reversing transcriptomic age without gene therapy;
[[papers/chemical-reprog-lifespan-2025]] extended to actual lifespan in
mice. **But** [[papers/lipid-droplets-reprog-2025]] is a serious
negative result — in-vivo cocktails cause **lipid-droplet toxicity in
liver/kidney**, hindering rejuvenation. Therapeutic window is narrow
and unresolved.

### 8. Telomerase gene therapy
[[papers/tert-knockin-2025]] shows TERT knock-in extends lifespan
without tumorigenicity. Older AAV9-TERT work (~+13–24% lifespan) is
real. **Warning:** the 2022 PNAS MCMV-TERT paper (+41.4% lifespan)
was **retracted in August 2025** — do not cite it. See
[[topics/telomeres]].

### 9. AI-driven intervention discovery (meta-technology)
Not a therapy itself, but the **highest-leverage thing a
computer-scientist can work on** — these tools will 10× the discovery
rate for everything above:

- [[papers/clockbase-agent-2025]] — autonomous agent reanalysing ~2M
  public molecular samples across ~43k perturbations through ~40
  aging clocks.
- [[papers/agextend-2025]] — explainable AI platform, **1.1 billion
  compounds** screened for geroprotective activity.
- [[papers/singular-rejuv-atlas-2024]] — SINGULAR unifies six
  rejuvenation strategies (73 cell types) on a common network-biology
  footing.
- [[papers/shift-sb000-2025]] — generative clock-directed screen that
  produced SB000 (a real tier-1 therapy lead).
- [[papers/network-repurposing-aging]] — SHARP pipeline +
  transcription-based pAGE metric over 6,442 drugs.

## What I'd bet against (for now)

- **NAD+ precursors (NMN / NR) as standalones** — meta-analyses
  consistently show blood NAD+ rises, but clinical endpoints move
  little. [[papers/nr-longcovid-2025]] is the latest large RCT and
  results are mixed.
- **"Young plasma" transfusion products without an identified active
  factor** — the factor-level work
  ([[papers/pedf-parabiosis-2024]], [[papers/fgf17-young-csf-2022]]) is
  the credible path; undefined plasma products are the opposite.
- **Telomerase-as-pill hype** — the 2025 MCMV-TERT retraction is a
  warning flag.

## For a CS person specifically

If you want to **contribute to** rejuvenation rather than just measure
it, the most tractable project from this list is:

> **Extend [[papers/clockbase-agent-2025]]'s architecture with a better
> clock** (e.g., [[papers/pathwayage]] or an [[papers/epinflammage]]-style
> tabular DNN) and **rescore the 43k public perturbations**.

All inputs are public, all methods are described, any significant new
hit is publishable, and the infrastructure scales with your budget.

Secondary projects:

- **SHARP × ITP cross-validation** — do drugs that
  [[papers/network-repurposing-aging]] ranks close to aging hallmarks
  actually extend mouse lifespan in [[papers/itp-mpd-portal]]?
- **Rebuild [[papers/shift-sb000-2025]]-style single-gene screen** on
  Perturb-seq or LINCS L1000 data.
- **Submit to** [[papers/biomarkers-aging-challenge]] or form a team
  for [[papers/xprize-healthspan]] ($101M, biomarker-measured).

## Related topics
[[topics/reprogramming]], [[topics/senolytics]],
[[topics/parabiosis-blood-factors]], [[topics/clinical-trials]],
[[topics/machine-learning]], [[topics/drug-repurposing]]
