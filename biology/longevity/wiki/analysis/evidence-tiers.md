---
id: evidence-tiers
title: "Evidence maturity tiers for longevity interventions"
kind: synthesis
covers: [rapamycin, vitamin-d, omega-3, creatine, blood-pressure, statins-apob, aerobic-exercise, resistance-training, sleep, gip-glp1, senolytics, partial-reprogramming, nad-mitophagy, biomarkers-of-aging, aging-clocks]
---

# Evidence maturity tiers

A common ladder for grading the evidence behind every intervention in
the wiki. Avoids the inconsistency where mouse-only data is used to
*include* one intervention (rapamycin, partial reprogramming) while
identical-tier data is used to *exclude* another (creatine,
resveratrol). The tier alone is not a verdict; quality and
reproducibility within tier matter at least as much.

## Tier ladder

| Tier | Description | Example |
|------|-------------|---------|
| **T0** | In vitro / cell culture only | Most senolytic screens before animal validation |
| **T1** | Invertebrate lifespan (C. elegans, Drosophila) | Many "longevity compounds" that never replicate in mammals |
| **T2** | Single mouse study, lifespan or healthspan endpoint | Initial reports before independent replication |
| **T3** | Replicated mouse studies; ITP-grade or independent labs | Rapamycin, 17α-estradiol, acarbose |
| **T4** | Non-human primate or large mammal | Caloric restriction in rhesus (NIA, Wisconsin) |
| **T5** | Small human trial (n < 100), surrogate endpoint | Many pilot RCTs of off-label longevity drugs |
| **T6** | Phase 2/3 human RCT, surrogate endpoint, **or** large prospective cohort (n>10k) with hard endpoint | PEARL (rapamycin), VITAL (vitamin D), Mandsager VO2max cohort |
| **T7** | Phase 3 RCT or meta-analysis with hard endpoint (MACE, all-cause mortality), **or** large multi-cohort meta-analysis with hard endpoint where randomization is infeasible | CTT statin meta-analysis, SPRINT BP trial, Cappuccio sleep-mortality meta-analysis, Wood alcohol pooled IPD |

## Within-tier quality dimension

Tier alone is necessary but not sufficient. For each entry, separately
assess:

- **Endpoint clarity.** Was the primary endpoint pre-specified? Did
  the trial meet it, or are the headline findings post-hoc /
  subgroup / secondary?
- **Effect size.** Absolute risk reduction, not just relative. RR 0.8
  on a 1% baseline event rate is a different result than RR 0.8 on a
  20% baseline.
- **Replication.** Single-lab finding vs. independent replication
  vs. meta-analysis of trials.
- **Population.** Healthy normative-aging adults, clinical patients,
  athletes, mice on standard chow vs. high-fat diet — generalisation
  matters.
- **Confounding / bias risks.** P-hacking, selection bias,
  industry-funded with no independent confirmation, retracted source
  papers in the chain. For observational data: residual confounding,
  reverse causation, healthy-user bias.
- **Design (RCT vs. observational).** RCT > observational at the
  same tier. For interventions where RCT is infeasible (lifelong
  smoking, sleep duration, occupational exercise), observational
  evidence with Mendelian-randomization triangulation is the
  realistic ceiling and should be flagged as such, not penalised.
- **Dose-response.** Was a dose-response curve shown? Was the
  effective dose identified, or is it educated guesswork?

A T7 trial with weak effect size and high heterogeneity can be a
weaker basis for action than a T3 result with consistent
cross-laboratory replication and a clean dose-response.

## Why we accept some T2-T3 evidence and not others

The wiki's editorial choice when a higher-tier result is unavailable:

| Accept the T2-T3 result | Reject the T2-T3 result |
|-------------------------|-------------------------|
| Replicated independently | Single lab, no replication |
| Plausible mechanism mapped to a known pathway | Mechanism vague or "anti-oxidant", "anti-inflammatory" handwave |
| Consistent dose-response | No dose-response or non-monotonic |
| Effect persists across strains, sexes, diets | Effect specific to one sex, strain, or diet |
| Chain of reasoning to a human-relevant endpoint | "More research needed" with no human path |

Two examples that score differently on the same tier:

- **Rapamycin (T3 in mice).** Replicated across multiple ITP cohorts,
  independent labs, plausible mTOR mechanism, dose-response shown,
  effect in both sexes. We retain it as "promising despite mouse-only
  lifespan evidence."
- **Resveratrol (T2-T3 mixed).** Initial mouse hits failed to
  replicate in subsequent ITP runs, mechanism contested, no
  dose-response convergence, no clean human surrogate-endpoint signal
  at achievable doses. We reject it despite roughly the same notional
  tier.

## Where each intervention currently sits (initial pass)

This list is incomplete and will grow. Each row gives the maximum
tier reached by *any* trial of that intervention, not an average.

### Tier 7 (mortality/hard-endpoint RCT or meta-analysis)
- **Smoking cessation.** [[papers/jha-2013-smoking-mortality]].
- **Blood pressure control to <120 SBP** in non-diabetic
  hypertensives. [[papers/sprint-2015-intensive-bp]].
- **Statin therapy** for LDL-C lowering. [[papers/ctt-2012-statins-low-risk]].
- **Resistance training** for lean mass / strength /
  all-cause mortality. [[papers/saeidifard-2019-resistance-mortality]],
  [[papers/garcia-hermoso-2018-strength-mortality]].
- **Creatine monohydrate as RT adjunct** for lean mass / strength
  in older adults. [[papers/chilibeck-2017-creatine-older-adults]]
  is T7 on the biomarker; mortality benefit is by analogy to the
  strength-mortality literature, not directly demonstrated.

### Tier 6 (Phase 2/3 RCT, surrogate endpoint)
- **GLP-1 agonists** for biological age / metabolic markers.
  [[papers/semaglutide-glp1-epigenetic-age-rct-2025]]. (Cardiovascular
  hard-endpoint RCTs separately push GLP-1s closer to T7 for the
  CV indication.)
- **Vitamin D supplementation** in non-deficient adults — null on
  cancer, CVD, mortality. [[papers/vital-2019-vitd-omega3]].
  Tier 6, **negative result.**
- **Rapamycin** (PEARL): primary endpoint not met; secondary
  subgroup signals only. [[papers/pearl-rapamycin-2025]]. Tier 6,
  **primary endpoint negative.** (Higher tier in mice: T3.)
- **Plasmapheresis without IVIG** for biological age:
  [[papers/plasmapheresis-aging-trial-2025]]. Tier 6, **negative on
  primary, some clocks accelerated.**

### Tier 5 (small human trial, surrogate endpoint)
- **Urolithin A** (Mitopure), immune endpoints, n=66.
  [[papers/urolithin-a-immune-2025]].
- **Senolytic D+Q** in MCI. [[papers/senolytic-mci-ebiomed-2025]],
  [[papers/senolytic-methylation-2024]]. Pilot-scale.
- **NR / NMN** in long-COVID and other surrogate-endpoint
  trials. [[papers/nr-longcovid-2025]]. Tier 5, **null on primary
  endpoints.**
- **TPE + IVIG** for biological age clocks.
  [[papers/tpe-ivig-biological-age-rct-2025]].

### Tier 3-4 (replicated mouse / non-human primate)
- **Rapamycin** (mouse lifespan): T3 ITP-grade.
  [[papers/itp-nia]].
- **Trametinib + rapamycin combination** (mouse lifespan).
  [[papers/trametinib-rapamycin-itp-2025]]. T3.
- **Anti-IL-11 antibody** (mouse lifespan).
  [[papers/il11-inhibition-2024]]. T3 (single high-quality study,
  awaiting replication).
- **Caloric restriction** (rhesus): T4 lifespan/healthspan.
- **17α-estradiol, acarbose** (mouse lifespan): T3 ITP-grade.
- **Klotho gene therapy** (mouse): T2-T3.
  [[papers/klotho-skl-aav-2025]].

### Tier 1-2 (invertebrate or single-mouse)
- Most of the supplement-aisle "longevity compounds" that have
  failed to reach T3, including many resveratrol-class polyphenols.

### Tier 0 (cell / in vitro only)
- Most partial-reprogramming assays at the in-vitro stage.
- Many senolytic screens before animal validation.

## How to use this in the wiki

- **Paper pages** should state the maximum tier reached by the
  primary paper plus the within-tier quality flags relevant to that
  finding.
- **Recommendation pages** should tag every intervention with
  `(Tier N)` and state the trial used to reach it.
- **Analysis pages** should call out tier-mismatches when comparing
  interventions (don't compare a T7 statin meta-analysis to a T3
  mouse rapamycin study without flagging the asymmetry).
- **Ingest pipeline** (see `biology/longevity/CLAUDE.md`): every new
  paper gets a tier assignment in its YAML or summary.

## Caveats on the framework

- A high tier is not a verdict of "useful." VITAL (T6) was negative.
- A low tier is not a verdict of "useless." Most clinical-stage
  longevity therapies started at T3 and the bar to advance beyond
  T5 is dominated by trial cost, not biology.
- Surrogate endpoints (biological-age clocks, ApoB, biomarker
  panels) have varying degrees of mortality validation. Treat
  T6-on-clocks as weaker than T6-on-MACE.
- For interventions where T7 trials are infeasible (e.g.
  multi-decade longevity primary endpoint in healthy humans), the
  ladder caps out earlier and we should say so explicitly rather
  than penalise the intervention for an impossible trial.

## Related
- [[analysis/promising-reverse-aging]] — applied tier reasoning across
  the current frontier.
- [[topics/clinical-trials]] — what's running now.
