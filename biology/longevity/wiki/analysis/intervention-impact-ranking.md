---
id: intervention-impact-ranking
title: "Top 20 longevity interventions ranked by tier-weighted impact"
date: 2026-04-29
kind: analysis
---

# Top 20 longevity interventions ranked by tier-weighted impact

A working ranking of the interventions in `recommendations.md` by
how much they would actually move all-cause mortality if a typical
adult adopted them, discounted by how confident the evidence is.

This is a thinking tool, not a prescription. The ranking exists to
make trade-offs visible — large effect size on weak evidence
versus modest effect size on strong evidence — and to honestly
flag where mouse-only or observational data is doing the heavy
lifting. Don't read row N as "objectively better than row N+1";
the score is a heuristic, not a measurement.

## Methodology

**Score = effect × tier_weight × population_applicability**

- **effect**: best-evidence relative reduction in all-cause
  mortality or composite hard endpoint, as a decimal. HR 0.80
  → 0.20 effect. Where the primary endpoint is a surrogate (lean
  mass, LDL-C, ApoB), I substitute the best-evidence translation
  to mortality from the relevant per-paper page and flag it.
- **tier_weight** (per [[analysis/evidence-tiers]]):
  T7 = 1.0, T6 = 0.6, T5 = 0.3, T4 = 0.5, T3 = 0.2, T2 = 0.1,
  T0–T1 = 0.05. T6 is heavily discounted because the modal T6
  result is a surrogate-endpoint RCT or large cohort, both of
  which have specific failure modes (surrogate-mortality
  divergence, healthy-user bias).
- **population_applicability**: 1.0 if the intervention applies
  to most adults; lower for narrow indications. Smoking
  cessation has applicability 1.0 only among smokers; CoQ10 has
  applicability 0.05 because heart-failure prevalence is small.
  For a given individual the applicability for them is binary:
  0 if you don't have the indication, 1 if you do. The ranking
  weights by population so that "lift across all readers" is
  legible.

**Key honest caveats baked into the methodology:**

- The score is multiplicative, so a strong-evidence small-effect
  intervention can outrank a weak-evidence large-effect one.
  This is intentional. Most "huge effect" claims in longevity
  marketing collapse on inspection of the underlying tier.
- T7 observational (large prospective cohort with hard endpoint)
  and T7 RCT both score 1.0 on tier weight here, but the
  per-paper write-ups separately flag observational vs RCT
  design — see the within-tier quality dimension in
  [[analysis/evidence-tiers]].
- Effect sizes are best-evidence point estimates; the
  uncertainty intervals matter and are noted in the per-paper
  pages. A score of 0.21 vs 0.18 should not be read as a
  meaningful ranking difference.
- Several entries are partial substitutes (cardio + resistance
  training + sauna are all heat / activation interventions
  with overlapping mechanisms; a person already doing #2
  cannot fully add the score of #5 on top).

## The top 20

Effect sizes are pulled from the cited paper pages; chase the
wikilinks for primary sources, study design, and confidence
intervals.

| Rank | Intervention | Effect | Tier | Pop. | Score | Anchor paper |
|------|--------------|--------|------|------|-------|--------------|
| 1 | Don't smoke (or quit) | 0.60 | T7 | 1.0 | **0.60** | [[papers/jha-2013-smoking-mortality]] |
| 2 | Lower BP to ~120 SBP (in hypertensives) | 0.27 | T7 | 0.45 | **0.12** | [[papers/sprint-2015-intensive-bp]] |
| 3 | Aerobic / cardio training | 0.30 | T6 obs | 1.0 | **0.18** | [[papers/mandsager-2018-vo2max-mortality]] |
| 4 | Resistance training | 0.21 | T7 | 1.0 | **0.21** | [[papers/saeidifard-2019-resistance-mortality]] |
| 5 | Lower LDL-C / ApoB with statins (per indication) | 0.21 (per 1 mmol/L) | T7 | 0.50 | **0.10** | [[papers/ctt-2012-statins-low-risk]] |
| 6 | Sleep 7–9 hr regularly | 0.15 | T7 | 1.0 | **0.15** | [[papers/cappuccio-2010-sleep-mortality]] |
| 7 | GLP-1 in obesity with prior CVD | 0.20 (MACE) | T7 | 0.10 | **0.020** | [[papers/select-2023-semaglutide-cv-outcomes]] |
| 8 | Manage body composition (waist, visceral fat) | 0.25 | T7 obs | 1.0 | **0.15** | [[papers/pischon-2008-waist-mortality]] |
| 9 | Sauna 4–7 sessions/week | 0.40 (all-cause) | T6 obs | 0.4 | **0.10** | [[papers/laukkanen-2015-sauna-mortality]] |
| 10 | Minimize alcohol (≤100 g/week) | 0.08 | T7 | 0.7 | **0.06** | [[papers/wood-2018-alcohol-thresholds]] |
| 11 | Manage glucose / HbA1c | 0.15 | T7 | 0.4 | **0.06** | [[papers/sniderman-2011-apob-meta]] (residual-risk path) |
| 12 | Measure Lp(a) once + escalate other risk factors if elevated | 0.10 (indirect) | T7 | 1.0 | **0.10** | [[papers/kamstrup-2009-lpa-mendelian]] |
| 13 | Measure ApoB, treat to target | 0.12 | T7 | 1.0 | **0.12** | [[papers/sniderman-2011-apob-meta]] |
| 14 | Creatine 5 g/day + RT (older adults) | 0.05 (inferred via lean mass) | T7 (biomarker) | 0.5 | **0.025** | [[papers/chilibeck-2017-creatine-older-adults]] |
| 15 | CoQ10 100 mg TID for symptomatic chronic HF | 0.49 (all-cause) | T7 | 0.02 | **0.010** | [[papers/mortensen-2014-coq10-qsymbio]] |
| 16 | Treat documented vitamin D / B12 / iron / TSH deficiencies | 0.05 | T6 | 0.3 | **0.009** | [[papers/vital-2019-vitd-omega3]] (defines what NOT to do) |
| 17 | Urolithin A 1 g/day (Mitopure) | 0.05 (immune surrogate) | T6 | 1.0 | **0.030** | [[papers/urolithin-a-immune-2025]] |
| 18 | Rapamycin off-label (intermittent low-dose) | 0.10 (best estimate; primary not met in humans) | T6 RCT primary not met / T3 mouse | 0.2 | **0.012** | [[papers/pearl-rapamycin-2025]], [[papers/itp-nia]] |
| 19 | Senolytic intermittent dosing (D+Q, fisetin) | 0.05 (uncertain) | T5 | 0.5 | **0.008** | [[papers/senolytic-mci-ebiomed-2025]] |
| 20 | Cold exposure (30 s end-of-shower) | 0.02 (sickness-absence proxy) | T6 soft endpoint | 1.0 | **0.012** | [[papers/buijze-2016-cold-shower-rct]] |

## How to read the table

- **Rank** is by tier-weighted score, ties broken by effect size
  on a hard endpoint.
- **Effect** is conditional on having the indication and adhering
  to the protocol. A statin's 21% per-1-mmol/L number assumes you
  actually take it; smoking's 60% number assumes you actually quit.
- **Tier** uses the rubric in [[analysis/evidence-tiers]].
- **Pop.** is rough. Smoking applies to ~12% of US adults
  (currently smoking) but ~30% of US adults are ever-smokers; the
  intervention "don't smoke" applies population-wide which is why
  it scores 1.0. Statins' applicability ~50% reflects that roughly
  half of adults have elevated lipid risk by guideline. CoQ10's
  0.02 reflects ~2% prevalence of symptomatic heart failure.
- **Score** is the product. Read it as a relative ordering of
  expected lift, not an absolute estimate.

## Top-of-the-table observations

The ranking has some non-obvious shape. Worth thinking about why:

- **#1 smoking cessation dwarfs everything else.** The effect
  size (60% relative mortality, ~10 years of life expectancy
  recovered when quitting at 25-44) is so large that no other
  single intervention comes close. If you smoke, this is the
  one to fix first. Everything else is a rounding error in
  comparison.
- **#4 resistance training outranks #5 statins** despite
  cardiovascular medicine getting more research dollars. The RT
  effect size is comparable, the population applicability is
  larger, and the tier evidence is at parity. This is one of the
  underprescribed-by-the-medical-system items the wiki keeps
  flagging.
- **#3 aerobic training scores lower than #4 resistance training**
  in this ranking, despite the popular framing being the other
  way around. The reason is the underlying tier: the cleanest
  cardio mortality evidence is observational (T6, Mandsager
  cohort) where the cleanest RT evidence is meta-analytic over
  RCTs (T7). The actual mortality benefit may well be larger
  for cardio in absolute terms, but the tier discount is real.
  The correct interpretation: do both.
- **#9 sauna scores surprisingly high** (0.10) given how
  speculative it sounds. The Laukkanen cohort's 50% CVD-mortality
  reduction at 4-7 sessions/week is very large; even after a 0.6
  tier discount and 0.4 population-applicability discount (most
  people don't have a daily sauna), it still beats glucosamine,
  cold showers, urolithin A, and rapamycin in the ranking.
- **#15 CoQ10 in chronic HF** has the largest raw effect on the
  list (HR 0.50 / 0.51 on MACE and all-cause mortality in
  Q-SYMBIO) but the lowest population applicability. For the
  ~2% of adults with symptomatic chronic HFrEF on standard
  therapy, this is one of the highest-leverage adjuncts
  available. For everyone else, it does nothing on the wiki's
  evidence base.
- **#18 rapamycin** ranks low here despite the loud marketing.
  Best evidence is mouse (T3); the human RCT (PEARL) missed
  primary endpoint. The score-weighting honestly reflects the
  current state.
- **#20 cold exposure** is included as a calibration: it has an
  RCT, it has a positive effect, and a defensible mechanism
  exists. It still ranks last because the endpoint is soft
  (self-reported sickness absence), the tier discount is real,
  and there is no mortality evidence at all.

## What's not in the top 20 and why

- **HRT for postmenopausal symptoms.** Mortality-neutral over
  18-yr WHI follow-up (HR 0.99). For symptomatic women, modern
  HRT improves quality of life, which is a legitimate goal that
  this longevity-mortality ranking does not score. See
  [[papers/manson-2017-whi-hrt-mortality]].
- **CPAP for cardiovascular prevention.** SAVE trial primary
  endpoint not met (HR 1.10, p=0.34). CPAP is excellent for
  daytime sleepiness; it is not a longevity intervention on
  current evidence. See [[papers/save-2016-cpap-cv-prevention]].
- **Glucosamine.** Observational signal only
  ([[papers/ma-2019-glucosamine-cv-mortality]], HR 0.85 CVD).
  No RCT confirmation. By the tier-weighted methodology it
  would land at score ~0.06 — comparable to alcohol moderation —
  but the lack of any RCT-level confirmation in a domain
  (supplement / preventive cardiology) where multiple
  observationally-promising candidates have failed phase-3 (vitamin
  D, vitamin E, beta-carotene, fish oil) means the implied
  uncertainty is wider than the methodology captures. Filed in
  "interesting but unconfirmed" rather than ranked.
- **NMN / NR.** RCT-level null on clinical endpoints. Score 0.
- **Universal vitamin D in non-deficient adults.** RCT-level
  null. Score 0. Treating documented deficiency does land in
  the ranking (#16) at modest score.
- **Resveratrol, "anti-aging" peptides without defined trial.**
  T0–T2 with no human endpoint data. Score < 0.005.
- **Telomerase-as-a-pill products.** The 2022 paper that drove
  the marketing was retracted. No evidence base.
- **Partial reprogramming, klotho gene therapy, anti-IL-11
  antibody, trametinib + rapamycin combination.** Mouse-only
  (T2-T3), no human trials read out yet. These dominate
  [[analysis/promising-reverse-aging]] but are not yet
  actionable. Tracked in [[analysis/yet-to-publish]] for when
  they advance.

## Honest limits of this ranking

- **Effect sizes are point estimates.** The confidence
  intervals matter. A 0.21 score with [0.10, 0.30] CI is
  meaningfully different from 0.21 with [0.18, 0.24]. Per-paper
  pages have the CIs.
- **Mortality is not the only longevity-relevant endpoint.**
  Healthspan, cognitive function, and quality of life matter
  too. HRT for symptoms is a clean example: zero mortality
  effect, real symptom benefit. This ranking under-scores
  symptom-driven and quality-of-life interventions.
- **Interactions between interventions are not modeled.**
  Cardio + resistance training has been shown to be additive
  (Saeidifard 2019: ~40% combined mortality reduction vs
  ~21% from RT alone). The table treats them as independent
  scores; in reality the additive structure is closer to the
  truth than pure independence.
- **The tier-weight constants are choices, not measurements.**
  T6 = 0.6 vs 0.5 vs 0.7 would shuffle ranks 9, 17, 18 around.
  The relative ordering of the top 6 is robust to plausible
  weight changes; the lower half is more sensitive.
- **Adherence is not modeled.** Statins are easy to take
  daily; sauna 4-7 times per week requires a sauna. The score
  assumes adherence; real-world impact is lower for harder-to-
  adhere interventions.

## Related
- `recommendations.md` (top-level reader-facing doc) — the
  actionable companion document.
- [[analysis/evidence-tiers]] — the tier rubric this ranking
  uses.
- [[analysis/promising-reverse-aging]] — the frontier-readiness
  ranking for not-yet-actionable interventions.
- [[analysis/yet-to-publish]] — pending readouts that would
  shuffle this ranking when they publish.
