# Follow-up: ranking the longevity interventions by tier-weighted impact

Quick update on the [open-source longevity wiki](https://www.reddit.com/r/Biohackers/comments/1sv0cyu/analyzed_75_longevity_papers_most_of_your_stack/), 698 score, 252 comments) I posted a few days ago. Several commenters asked the obvious next question: of all the interventions in the wiki, which ones actually move the needle the most? I worked it up as an analysis page and the rank ordering is non-obvious enough to be worth sharing.

Wiki page with the full table and per-row sourcing: [intervention-impact-ranking.md](https://github.com/toadlyBroodle/science/blob/main/biology/longevity/wiki/analysis/intervention-impact-ranking.md).

## Methodology

For each intervention I had a paper page for, I computed:

**score = effect × tier_weight × population_applicability**

- **effect**: best-evidence relative reduction in all-cause mortality or composite hard endpoint. HR 0.80 → 0.20.
- **tier_weight**: T7 hard-endpoint RCT/meta = 1.0, T6 surrogate-endpoint RCT or large cohort = 0.6, T5 = 0.3, T3 mouse = 0.2, T0–T1 = 0.05. Per the wiki's existing T0–T7 evidence tier rubric.
- **population_applicability**: 1.0 if the intervention applies to most adults, lower for narrow indications. CoQ10's applicability is 0.05 because symptomatic chronic HF is a 2% prevalence indication.

Effect sizes are point estimates. Confidence intervals matter and are noted on the per-paper pages. A score of 0.21 vs 0.18 should not be read as a meaningful difference.

## Top 10

| Rank | Intervention | Effect | Tier | Pop. | Score |
|------|--------------|--------|------|------|-------|
| 1 | Don't smoke (or quit) | 0.60 | T7 | 1.0 | **0.60** |
| 2 | Resistance training | 0.21 | T7 | 1.0 | **0.21** |
| 3 | Aerobic / cardio training | 0.30 | T6 obs | 1.0 | **0.18** |
| 4 | Sleep 7–9 hr regularly | 0.15 | T7 | 1.0 | **0.15** |
| 5 | Manage body composition / waist | 0.25 | T7 obs | 1.0 | **0.15** |
| 6 | Lower BP to ~120 SBP (in hypertensives) | 0.27 | T7 | 0.45 | **0.12** |
| 7 | Measure ApoB, treat to target | 0.12 | T7 | 1.0 | **0.12** |
| 8 | Lower LDL-C with statins (per indication) | 0.21 / mmol/L | T7 | 0.50 | **0.10** |
| 9 | Sauna 4–7 sessions/week | 0.40 | T6 obs | 0.4 | **0.10** |
| 10 | Lp(a) once + escalate other risk factors if elevated | 0.10 | T7 | 1.0 | **0.10** |

Ranks 11–20 (glucose management, alcohol moderation, creatine, CoQ10 in HF, deficiency correction, urolithin A, rapamycin off-label, senolytics, cold exposure) are in the wiki page.

## What jumps out

**Smoking cessation dwarfs everything else.** 0.60 vs the next closest at 0.21. ~10 years of life expectancy recovered when quitting at 25–44. If you smoke, this is the one to fix first; everything else is a rounding error.

**Resistance training outranks statins.** Comparable effect size, larger population applicability, evidence tier at parity. RT is one of the most underprescribed-by-the-medical-system items in the whole wiki.

**Aerobic training scores lower than RT** in this ranking, despite the popular framing being the other way around. The reason is the underlying tier: cleanest cardio mortality evidence is observational (Mandsager cohort, T6), cleanest RT evidence is meta-analytic over RCTs (Saeidifard, T7). The actual mortality benefit may well be larger for cardio in absolute terms, but the tier discount is real. The correct interpretation: do both. They're additive (~40% combined in Saeidifard 2019).

**Sauna scores surprisingly high** (0.10) despite sounding speculative. Laukkanen 2015 KIHD cohort: 4–7 sauna sessions/week vs 1/week, HR 0.50 fatal CVD, HR 0.60 all-cause mortality. Even after a 0.6 tier discount and a 0.4 population discount, it beats glucosamine, cold showers, urolithin A, and rapamycin in the ranking.

**CoQ10 in chronic HF has the largest raw effect on the list** (HR 0.51 all-cause in Q-SYMBIO) but the lowest population applicability (~2%). For symptomatic HFrEF on standard therapy this is one of the highest-leverage adjuncts available. For everyone else it does nothing on the wiki's evidence base.

**Rapamycin ranks low** despite the loud marketing. Best evidence is mouse (T3); the human PEARL trial missed primary endpoint (visceral adiposity, p=0.379). The score-weighting honestly reflects current state.

## What didn't make the top 20

- **HRT for postmenopausal symptoms.** WHI 18-yr follow-up: all-cause HR 0.99. Mortality-neutral. Real symptom benefit not scored by a mortality ranking.
- **CPAP for CV prevention.** SAVE 2016 missed primary endpoint (HR 1.10, p=0.34). CPAP works for daytime sleepiness, not for CV prevention on current evidence.
- **Glucosamine.** Observational signal only (Ma 2019, HR 0.85 CVD). No RCT confirmation in a domain where multiple observationally-promising candidates have failed phase-3.
- **NMN / NR.** Null on clinical endpoints in RCT. Score 0.
- **Universal vitamin D in non-deficient adults.** VITAL 2019 null. Treating documented deficiency lands at modest score; megadosing healthy adults does not.
- **Resveratrol, "anti-aging" peptides without defined trial.** No human endpoint data.

## Honest limits

- Effect sizes are point estimates; CIs matter and are on the per-paper pages.
- Mortality is not the only endpoint that matters. Healthspan, cognition, quality of life are real and this ranking under-scores symptom-driven interventions like HRT.
- Interactions are not modeled. Cardio + RT is closer to additive than the table treats them.
- The tier weights (T6 = 0.6 etc.) are choices, not measurements. Top 6 ordering is robust; lower half is more sensitive to the constants.
- Adherence is not modeled. Sauna 4–7x/week requires owning a sauna.

[GitHub: open-source longevity research wiki](https://github.com/toadlyBroodle/science/tree/main/biology/longevity)

90+ papers, every effect size traces to a primary source, every intervention tier-tagged. If you find an error, open an issue or PR.

As with the previous post: anti-AI comments will be ignored (if you don't understand the power of AI, I don't have time to explain it to you). Specific factual corrections, missing primary sources, and methodology critiques are welcome. Constructive suggestions from the previous post's comments have been incorporated into the latest version.
