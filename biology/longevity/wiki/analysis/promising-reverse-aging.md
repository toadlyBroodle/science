---
id: promising-reverse-aging-2026
title: "Most promising reverse-aging technologies (April 2026)"
date: 2026-04-23
kind: analysis
---

# Most promising reverse-aging technologies, April 2026

Ranking by **evidence strength × translational readiness ×
computational tractability**. Every claim here is backed by a paper
page in this wiki; chase the wikilinks for primary sources.

**Tier-rubric note.** Throughout this page, "Tier 1 / 2 / 3" refers
to *this analysis page's frontier-readiness ranking*, not the
[[analysis/evidence-tiers]] T0–T7 evidence-maturity ladder used on
the per-paper pages. The two are distinct: a frontier-Tier-1 entry
here (partial reprogramming, IL-11) may be at evidence-T2 (single
mouse study) on the underlying paper pages. The frontier-readiness
ranking weights *expected* translation rate and computational
tractability; the evidence-tier ranking weights *current* trial
maturity. Forward-looking trial readouts that would change this
ranking are tracked in [[analysis/yet-to-publish]].

The big shift since the previous version of this page is that **2025
delivered four new mouse-lifespan wins above 20%**
([[papers/il11-inhibition-2024]], [[papers/klotho-skl-aav-2025]],
[[papers/trametinib-rapamycin-itp-2025]],
[[papers/aav-osk-lifespan-2024]] reconfirmed and the new chemistry
in [[papers/retro-precision-reprog-2025]]) plus the **first rigorous
human RCT showing a biological-age reduction**
([[papers/tpe-ivig-biological-age-rct-2025]]: -2.61 yr from TPE+IVIG).
At the same time, the **clock infrastructure went foundation-model**
([[papers/longevity-llm-2026]], [[papers/scageclock-2026]],
[[papers/spatial-aging-clocks-brain-2024]]) and the **CAR-T senolytic
class** generalised from metabolism into gut regeneration
([[papers/anti-upar-cart-intestinal-2025]]).

## Tier 1: strongest evidence, translation already underway

### 1. Partial reprogramming (OSK / OSKM)
The only intervention class with *both* clean single-cell rejuvenation
data ([[papers/mesenchymal-drift-cell-2025]]) *and* a lifespan-level
readout in old mice. [[papers/aav-osk-lifespan-2024]] reports **+109%
remaining lifespan** with AAV-OSK in 124-week-old mice.

Three converging lines now solve the historical pluripotency / cancer
risk:

- **Single-factor alternatives** to OSK(M):
  [[papers/shift-sb000-2025]] identified **SB000** via ML-clock-directed
  screens; matches OSK(M) potency without activating pluripotency
  pathways.
- **Cell-targeted delivery**: [[papers/retro-precision-reprog-2025]]
  (Retro Biosciences, 2025) drives OSK from a **Cdkn2a promoter**, so
  the factors only express in senescent / aged cells. Single AAV dose
  in aged mice extended lifespan, reduced inflammation, and improved
  wound healing without the off-target organ toxicity.
- **Strain-level safety map**:
  [[papers/mouse-strains-osk-induction-2025]] shows that OSKM leak in
  reprogrammable mouse strains concentrates in **liver, intestine, and
  kidney** (the same organs where chemical reprogramming causes
  toxicity, see [[papers/lipid-droplets-reprog-2025]]).
  [[papers/organ-dedifferentiation-review-2025]] surveys cyclic-induction
  protocols (e.g. 2 ON / 5 OFF) shown safe for >35 cycles.

Altos Labs, NewLimit, and Retro Biosciences are all converging on this
class. Reviews: [[papers/paine-partial-reprog-review-2024]],
[[papers/reprogramming-rejuv-review-ncomms-2024]].

### 2. Senolytics (small molecule, PROTAC, CAR-T)
The class has split into three modalities, all moving forward
simultaneously.

**Small-molecule senolytics in humans:**

- [[papers/senolytic-mci-ebiomed-2025]]: STAMINA pilot dosed D+Q
  against cognition + mobility in pre-Alzheimer's adults.
- [[papers/senolytic-methylation-2024]]: D+Q±fisetin vs. epigenetic
  clocks over 6 months.
- ML discovery pipeline ([[papers/smerbarreto2023-senolytics]],
  [[papers/senolytic-predictor-2025]]) is cheap to extend; already gave
  3 validated hits from <60 training positives.

**PROTAC senolytics (new chemistry that rescues the BH3-mimetic class):**

- [[papers/bcl-xl-protac-753b-senolytic-2025]] (*Nature Aging* 2025):
  **753b**, a dual BCL-xL/BCL-2 PROTAC recruiting the VHL E3 ligase,
  selectively eliminates **senescent hepatocytes** in aged and
  MASH-driven HCC mice. **Avoids the navitoclax/ABT-263 thrombocytopenia**
  that killed the BH3-mimetic class for senolytic use. Effective even
  after fibrosis is established. The "rescue" of a written-off class
  is the year's biggest senolytic-chemistry story.

**CAR-T senolytics (single-shot, year-long persistence):**

- [[papers/senolytic-cart-upar-2024]] (*Nature Aging* 2024): anti-uPAR
  CAR-T persists and expands for **>12 months** in mice; one infusion
  improves glucose tolerance and exercise capacity in aged mice;
  prophylactic dosing prevents HFD-induced metabolic dysfunction.
- [[papers/anti-upar-cart-intestinal-2025]] (*Nature Aging* 2025):
  Amor lab follow-up generalises to **gut aging**, restoring barrier
  function, microbiome composition, and stem-cell regenerative
  capacity. Also radioprotective.

The implication: senolysis is no longer just about which small
molecule kills which senescent subtype. The platform question now is
**which senescent compartment matters most for which disease**, then
pick the modality.

### 3. mTOR / MEK / IL-11: the cytokine-nutrient signalling axis
This started as "rapamycin extends lifespan." It is now a **convergent
target axis** with three independent lifespan-extending interventions
that all touch the **ERK / AMPK / mTORC1** circuitry, all in mice over
the last 24 months:

- [[papers/itp-nia]] + [[papers/pearl-rapamycin-2025]]: rapamycin is
  the most reproducible mouse-lifespan intervention (+20-25%) and
  now has a 1-year human RCT (PEARL). PEARL's primary efficacy
  endpoint (visceral adiposity) was not met; sex-stratified subgroup
  analyses of secondary endpoints showed lean-mass/pain benefit in
  women on 10 mg and bone-mineral-content benefit in men on 10 mg.
  Safety endpoint was met. Read it as a feasibility/safety result,
  not a longevity-efficacy RCT.
- [[papers/trametinib-rapamycin-itp-2025]] (*Nature Aging* 2025):
  combining rapamycin with the FDA-approved MEK inhibitor trametinib
  gives **~29% (F) / 27% (M) median lifespan extension, additive over
  rapamycin alone**. Both drugs are FDA-approved, so the human path is
  the shortest of any new lifespan story this cycle.
- [[papers/il11-inhibition-2024]] (*Nature* 2024): genetic deletion of
  the pro-inflammatory cytokine **IL-11** extends lifespan **+25%**;
  late-life anti-IL-11 antibody dosing extends lifespan **+22% (M),
  +25% (F)**. The antibody is already in human trials for fibrotic
  lung disease, putting it on the **shortest path from mouse-lifespan
  hit to human readout**.

Mechanistically these three converge: IL-11 signals through ERK to
AMPK to mTORC1; trametinib blocks the ERK arm; rapamycin blocks the
mTORC1 arm. The implication is that any single-axis intervention
(rapamycin alone, anti-IL-11 alone) leaves residual aging signal on
the table, and **combinations should compound**. The
trametinib + rapamycin additivity is the first published
demonstration. See [[topics/mtor]], [[topics/inflammation]],
[[topics/hallmarks-of-aging]].

## Tier 2: strong mechanism, promising data, scaling now

### 4. Klotho gene therapy
[[papers/klotho-skl-aav-2025]] (*Mol Therapy* 2025) extended **wild-
type mouse lifespan by ~20%** with a **single AAV9 injection** of the
secreted klotho isoform (s-KL). Cross-organ rejuvenation: muscle
fibrosis down, bone microstructure preserved, hippocampal
neurogenesis up, microglial phagocytosis up, brain transcriptome
showed reduced mitochondrial-dysfunction signature.

Why it sits high: it is a **clean, single-dose, single-target**
intervention with WT-mouse lifespan extension and a clinical pipeline
(Klotho Neurosciences). The s-KL isoform avoids the FGF23/mineral-
metabolism safety concerns of the membrane-bound p-KL. Translates
~30 years of Klotho-and-aging biology into a deliverable.

Caveats: male-only longevity readout (female cohort confounded by
unrelated dermatitis); requires combined ICV + IV injection in the
mouse protocol, so a BBB-crossing AAV serotype is needed for clinical
translation. See [[topics/gene-therapy]],
[[topics/brain-rejuvenation]].

### 5. Young blood / CSF factors and the TPE+IVIG split
After a decade of parabiosis work, the field has identified **three
specific single factors** that reproduce much of the young-blood
effect:

- [[papers/pedf-parabiosis-2024]]: **PEDF** extends fibroblast
  replicative lifespan and reverses age-related pathology in aged
  mice.
- [[papers/fgf17-young-csf-2022]]: **FGF17** in young CSF rejuvenates
  aged hippocampus via oligodendrocyte progenitor proliferation.
- [[papers/gpld1-tnap-brain-vasculature-2026]] (*Cell* 2026, Villeda
  lab follow-up): identifies **TNAP on brain endothelium** as the
  substrate of the liver exerkine **GPLD1**. TNAP inhibition alone
  rescues memory in aged and 5xFAD AD mice. This is the **first
  exercise-mimetic axis** to reach a single druggable target on the
  vasculature side, distinct from the FGF17 / PEDF parenchymal path.

Plus [[papers/hcpb-review-2024]] for the field overview.

**Plasma exchange has split into a positive and a negative trial:**

- [[papers/tpe-ivig-biological-age-rct-2025]] (*Aging Cell* 2025,
  Buck Institute): biweekly **TPE plus IVIG cuts biological age by
  2.61 years** in a single-blinded placebo-controlled RCT (n=42, age
  >50). All 15 epigenetic clocks moved in the rejuvenation direction;
  immunosenescence markers reversed. **First rigorous human RCT
  showing biological-age reduction by an actively delivered
  intervention.**
- [[papers/plasmapheresis-aging-trial-2025]] (*Sci Reports* 2025,
  with Steve Horvath co-authoring): plasmapheresis without IVIG in
  healthy donors produced **no rejuvenation** across multiple clocks;
  some clocks accelerated.

The two together imply that the active ingredient is in the **IVIG
replacement fluid**, not the removal of "old" plasma factors per se.
This is the most informative "matched positive vs negative trial" the
parabiosis field has ever had. See [[topics/parabiosis-blood-factors]].

### 6. iPSC-derived young immune cells
[[papers/ipsc-mononuclear-phagocyte-aging-brain-2025]] (*Adv Sci*
2025, Cedars-Sinai / Svendsen): IV infusion of **human iPSC-derived
mononuclear phagocytes (iMPs)** into aged and 5xFAD mice restored
hippocampal mossy cells, improved hippocampus-dependent cognition,
reduced age-elevated serum amyloids (SAA2, SAP), and **reduced
transcriptional age in 9 of 15 hippocampal cell types** by an ML
estimator. Cells stay peripheral; the effect is systemic and immune-
mediated.

This is a **new modality**: off-the-shelf, manufacturable, allogeneic
iPSC-derived "young immune cell" therapy. Distinct from
[[papers/adsc-exosomes-2022]] (cell-free EVs) and from
[[papers/senolytic-cart-upar-2024]] (engineered T cells targeting
senescent cells). See [[topics/stem-cells]],
[[topics/immune-rejuvenation]].

### 7. Mitophagy induction
[[papers/urolithin-a-immune-2025]] is a clean *Nature Aging* RCT
showing UA expands naive-like CD8+ T cells and increases CD8+
fatty-acid oxidation in middle-aged adults over 4 weeks. Gut-microbial
metabolite, shelf-stable pill, well-understood mechanism, replicable
endpoint. Part of the [[topics/nad-mitophagy]] axis.

### 8. Young stem-cell exosomes
[[papers/adsc-exosomes-2022]] (*Sci Adv*): ADSC-derived small EVs from
young mice improve healthspan, reduce frailty, and lower epigenetic
age in old mice. Cell-free, shelf-stable, avoids the regulatory
complexity of whole-cell stem-cell therapies. See
[[topics/exosomes-extracellular-vesicles]].

## Tier 3: longer horizon, high ceiling

### 9. Chemical reprogramming cocktails
[[papers/yang-chemical-cocktails-2023]] (Sinclair lab) demonstrated 6
cocktails reversing transcriptomic age without gene therapy;
[[papers/chemical-reprog-lifespan-2025]] extended to actual lifespan
in mice. **But** [[papers/lipid-droplets-reprog-2025]] is a serious
negative result: in-vivo cocktails cause lipid-droplet toxicity in
liver and kidney. Therapeutic window is narrow and unresolved. The
[[papers/mouse-strains-osk-induction-2025]] tissue-leak map predicts
which organs will be hit hardest for any non-targeted reprogramming
strategy.

### 10. Telomerase gene therapy
[[papers/tert-knockin-2025]] shows TERT knock-in extends lifespan
without tumorigenicity. Older AAV9-TERT work (~+13–24% lifespan) is
real. **Warning:** the 2022 PNAS MCMV-TERT paper (+41.4% lifespan)
was retracted in August 2025. Do not cite it. See [[topics/telomeres]].

## Mechanistic convergence: the picture is getting cleaner

Several apparently unrelated 2024-2026 results now point at **shared
mechanistic substrates**. This is the strongest sign that the field
is moving from "what extends lifespan?" to "why does it?":

1. **The cytokine to nutrient-sensing axis.**
   [[papers/il11-inhibition-2024]] (anti-IL-11 antibody) +
   [[papers/trametinib-rapamycin-itp-2025]] (MEK + mTOR inhibitors) +
   [[papers/pearl-rapamycin-2025]] (rapamycin alone) all converge on
   the **ERK to AMPK to mTORC1** circuit. Three different
   interventions, three different chemistries, one pathway, all giving
   ~22-29% mouse lifespan extension late in life. Combination space is
   wide open.

2. **Senescence is not a uniform cell state.**
   [[papers/anti-upar-cart-intestinal-2025]] (CAR-T in gut) +
   [[papers/bcl-xl-protac-753b-senolytic-2025]] (PROTAC in liver)
   succeed by being **tissue-specific**. The earlier
   [[papers/smerbarreto2023-senolytics]] ML pipeline is now best read
   as identifying senolytic chemistry per senescence subtype, not a
   universal "kills senescent cells" molecule. Senolytic discovery
   should now be tagged by tissue and disease, not just by
   "senolytic activity."

3. **Brain aging is the load-bearing healthspan axis.**
   [[papers/organ-proteomic-clocks-2025]] and
   [[papers/plasma-proteomics-brain-immune-2025]] (independent UKB
   analyses, n>43k each) **both** rank brain proteomic age as the
   strongest single mortality / healthspan predictor across 10-11
   organ clocks. [[papers/spatial-aging-clocks-brain-2024]] then shows
   the brain itself is **structured aging**: T-cell infiltration
   accelerates aging in nearby cells, NSC niches slow it. Brain
   rejuvenation interventions
   ([[papers/fgf17-young-csf-2022]],
   [[papers/gpld1-tnap-brain-vasculature-2026]],
   [[papers/klotho-skl-aav-2025]],
   [[papers/ipsc-mononuclear-phagocyte-aging-brain-2025]]) are
   load-bearing for healthspan, not just niche cognitive endpoints.

4. **Removal vs. infusion in plasma-based interventions.**
   [[papers/tpe-ivig-biological-age-rct-2025]] (positive, with IVIG)
   vs [[papers/plasmapheresis-aging-trial-2025]] (negative, without
   IVIG) is the clearest natural experiment the parabiosis field has
   produced. The active ingredient is **introduction of a defined
   replacement, not removal of "old" plasma**. Single-factor work
   ([[papers/pedf-parabiosis-2024]],
   [[papers/fgf17-young-csf-2022]],
   [[papers/gpld1-tnap-brain-vasculature-2026]]) is the consistent
   path.

## Computational frontier: from clocks to foundation models

The clock toolkit changed shape in 2025-2026. What used to be a list
of regression models on CpGs is now a **layered stack** of cell-type-
resolved, spatial, and foundation-model components:

**Tabular foundation models for aging biology:**

- [[papers/longevity-llm-2026]]: Qwen3-14B fine-tuned on multi-omic
  aging biodata; **beats Horvath multi-tissue clock on epigenetic-age
  MAE (4.34 yr)**; one model handles age prediction across modalities
  plus proteome generation.
- [[papers/longevity-bench-2026]]: companion benchmark suite. The
  evaluation gap (no shared eval to compare general LLMs to fine-tuned
  models) is now closed.

**Single-cell foundation clocks:**

- [[papers/scageclock-2026]]: gated multi-head attention NN, 16M
  cells, 40+ tissues, 400+ cell types; per-cell-type MAE ~2 yr;
  introduces an Aging Deviation Index (ADI) usable as an intervention
  readout per cell.
- [[papers/scimmuaging-immune-clocks-2025]]: per-cell-type immune
  aging clocks across 1,081 donors; **demonstrates clock reversibility
  in humans** post-COVID (monocyte rejuvenation in recovery) and
  post-BCG (CD8+ T-cell rejuvenation in inflamed subjects).
- [[papers/spatial-aging-clocks-brain-2024]]: first **spatial** aging
  clock; reveals T-cell pro-aging shadow vs. NSC pro-rejuvenation halo.
  Cell-cell interactions become the unit of intervention design.
- [[papers/scbayesage-2025]]: earlier Bayesian per-cell clock; the
  baseline scAgeClock improves on.

**Genome-scale perturbation substrate:**

- [[papers/x-atlas-orion-perturbseq-2025]]: Xaira's 8M-cell, all-
  protein-coding-gene Perturb-seq atlas. The **training-data
  bottleneck for "virtual cell" aging models is now broken**.
- [[papers/lifestyle-atlas-tirolgesund-2025]]: open multi-omic
  longitudinal lifestyle-intervention atlas (156 women, 6 months IF
  vs smoking cessation, 7 tissues). Counterpart on the **human cohort
  side**.

**Discovery agents on top:**

- [[papers/clockbase-agent-2025]]: autonomous agent over ~2M molecular
  profiles, ~43k perturbations, ~40 aging clocks.
- [[papers/agextend-2025]]: explainable AI, **1.1B compounds**
  screened.
- [[papers/shift-sb000-2025]]: generative clock-directed screen that
  produced a real Tier 1 therapy lead (SB000).
- [[papers/network-repurposing-aging]]: SHARP + transcription-based
  pAGE metric over 6,442 drugs.
- [[papers/singular-rejuv-atlas-2024]]: SINGULAR unifies six
  rejuvenation strategies on a common network footing.

**The composable picture:** [[papers/x-atlas-orion-perturbseq-2025]]
provides 8M-cell perturbation data; [[papers/scageclock-2026]] /
[[papers/scimmuaging-immune-clocks-2025]] provide per-cell-type aging
scoring; [[papers/longevity-llm-2026]] / [[papers/clockbase-agent-2025]]
orchestrate. Anyone building an intervention-discovery loop can now
assemble these as off-the-shelf modules, where 12 months ago you
would have had to build the substrate from scratch.

## Negative results and what they teach

The 2025-2026 negative results are unusually informative because they
are **paired with positive results from the same modality**:

- **Plasmapheresis without IVIG fails.**
  [[papers/plasmapheresis-aging-trial-2025]] paired with
  [[papers/tpe-ivig-biological-age-rct-2025]]: removal alone does not
  rejuvenate; replacement composition matters. Implication: any
  "young-blood" product without an identified active fraction is
  poorly bet.
- **Whole-body chemical reprogramming is liver-toxic.**
  [[papers/lipid-droplets-reprog-2025]] paired with
  [[papers/mouse-strains-osk-induction-2025]] paired with
  [[papers/retro-precision-reprog-2025]]: the same off-target tissues
  (liver, intestine, kidney) keep showing up. The lesson is **target
  the cells that need it** (Cdkn2a-restricted promoters) rather than
  fight the toxicity.
- **Telomerase-as-pill hype**: the 2025 MCMV-TERT retraction means
  the +41% lifespan claim should be treated as withdrawn.
  [[papers/tert-knockin-2025]] is the credible remaining anchor.
- **Standalone NAD+ precursors disappoint at endpoints.**
  [[papers/nr-longcovid-2025]] is the latest large RCT and the
  blood-NAD+-up / clinical-endpoint-flat pattern persists.

## What I'd bet against (for now)

- **Single-modality "young plasma" products without a defined active
  fraction.** Active-fraction work
  ([[papers/pedf-parabiosis-2024]], [[papers/fgf17-young-csf-2022]],
  [[papers/gpld1-tnap-brain-vasculature-2026]]) is the credible path;
  see the negative-result pair above.
- **NAD+ precursors (NMN, NR) as standalones.**
- **Whole-body OSKM without cell targeting** (see
  [[papers/lipid-droplets-reprog-2025]] +
  [[papers/mouse-strains-osk-induction-2025]]).
- **Clinical claims based on the retracted MCMV-TERT result.**
- **Single biological-age-clock readouts as the only trial endpoint.**
  The 14-clock comparison ([[papers/nc-2025-14clocks]]) showed clocks
  disagree by enough that 1-clock evidence is fragile;
  [[papers/longevity-bench-2026]] is the right way to evaluate
  going forward.

## Worth watching but not yet Tier 1

- **GLP-1 agonists as gerotherapy.**
  [[papers/semaglutide-glp1-epigenetic-age-rct-2025]] is the first
  RCT with epigenetic-age outcomes (-2.3 to -4.9 years across clocks,
  -9% DunedinPACE). Population-specific (HIV with lipohypertrophy);
  needs replication and a weight-loss-matched comparator before the
  effect can be separated from weight-loss confounding.
- **Lifestyle multi-omic atlases.**
  [[papers/lifestyle-atlas-tirolgesund-2025]] shows high-compliance
  intermittent fasting moves multi-omic age trajectories within 6
  months. Useful as a comparator for any human intervention trial.

## What to watch in 2026-2027

- **First human readouts** for the partial-reprogramming class.
  NewLimit and Retro Biosciences are public about clinic-readiness;
  the [[papers/retro-precision-reprog-2025]] Cdkn2a-targeted approach
  has the cleanest preclinical safety profile.
- **anti-IL-11 antibody healthspan-endpoint data** from the existing
  fibrotic-lung-disease trials. Repurposing the active asset is the
  shortest human path of any Tier 1 intervention here.
- **Trametinib + rapamycin combination trial in humans.** Both drugs
  FDA-approved; the combo is a natural Phase II design.
- **Klotho Neurosciences clinical entry** for AAV-s-KL or recombinant
  s-KL.
- **Foundation-model crossover wins.** First case of a model trained
  on [[papers/x-atlas-orion-perturbseq-2025]] producing a validated
  rejuvenation hit that beats prior ML-only screens.
- **Spatial aging clocks for human tissues** beyond mouse brain
  ([[papers/spatial-aging-clocks-brain-2024]] sets the template).

## Open mechanistic questions

- Is **mesenchymal drift** ([[papers/mesenchymal-drift-cell-2025]])
  the same underlying signal that **scAgeClock's
  ribosome / translation modules** ([[papers/scageclock-2026]]) and
  **EpInflammAge's joint epi-inflammatory signal**
  ([[papers/epinflammage]]) are picking up from different angles? A
  unified-signature paper is the natural next step.
- The cytokine-mTOR axis (IL-11 + MEK + mTORC1): does **triple
  blockade** beat double, or does feedback compensation cap the
  ceiling?
- For senolytics: how much of the in-vivo benefit comes from
  removing the **secretome** (SASP) vs. removing the cells? A clean
  separation experiment is missing.
- **What is the active fraction of IVIG** that mediates the
  [[papers/tpe-ivig-biological-age-rct-2025]] effect?
- **Why does brain proteomic age dominate** healthspan prediction
  ([[papers/organ-proteomic-clocks-2025]],
  [[papers/plasma-proteomics-brain-immune-2025]]) when most
  interventions act peripherally? Is there a peripheral-to-CNS
  signaling bottleneck that all roads pass through?

## For a CS person specifically

The single highest-leverage project is unchanged in spirit but the
substrate is much better than 12 months ago:

> **Build an intervention-discovery loop on
> [[papers/x-atlas-orion-perturbseq-2025]] with
> [[papers/scageclock-2026]] (or
> [[papers/scimmuaging-immune-clocks-2025]]) as the per-cell-type
> aging readout, and rank every protein-coding gene knock-down by
> rejuvenation potential.**

This is now possible end-to-end with public data and public model
weights. Validating the top-N hits in mouse (via
[[papers/itp-mpd-portal]]-adjacent labs) is the bridge to in-vivo
evidence.

Secondary projects that the new substrate makes tractable:

- **Score the [[papers/network-repurposing-aging]] SHARP top-N
  against [[papers/itp-mpd-portal]]** (does network proximity to
  aging hallmarks actually predict ITP lifespan extension on
  held-out compounds?).
- **Train a clock-class mixture-of-experts** on the
  [[papers/longevity-bench-2026]] suite; the Horvath / GrimAge /
  DunedinPACE / scAgeClock ensemble is plausibly stronger than any
  single model.
- **Spatial intervention scoring.** Adapt the
  [[papers/spatial-aging-clocks-brain-2024]] architecture to other
  tissues (gut, muscle, skin) using existing Visium / MERFISH
  atlases; the cell-proximity feature is what gives the lift.
- **Replicate [[papers/shift-sb000-2025]] on Perturb-seq.** Use
  X-Atlas/Orion as the screen substrate, scAgeClock as the
  objective; any single-gene hit is publishable.
- **Submit to** [[papers/biomarkers-aging-challenge]] or form a team
  for [[papers/xprize-healthspan]] (\$101M, biomarker-measured).

## Related topics
[[topics/reprogramming]], [[topics/senolytics]],
[[topics/parabiosis-blood-factors]], [[topics/clinical-trials]],
[[topics/machine-learning]], [[topics/drug-repurposing]],
[[topics/mtor]], [[topics/inflammation]], [[topics/gene-therapy]],
[[topics/brain-rejuvenation]], [[topics/single-cell]],
[[topics/aging-clocks]]
