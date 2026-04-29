# Wiki Improvement TODO — Reddit Feedback Synthesis

Source: comments on r/Biohackers post (695 score, 243 comments) and r/immortalists post (26 score, 9 comments), Nov 2025.

Useful feedback distilled into actionable wiki improvements, ordered roughly by impact.

---

## Implementation status as of 2026-04-29

Almost all items below have been implemented. Summary of what shipped, with commit refs (see git log for full diffs):

**Centerpiece (mjjtiffany):**
- Evidence maturity model: `wiki/analysis/evidence-tiers.md` defines T0-T7 ladder + within-tier quality dimensions. Every paper page tier-tagged via YAML `evidence_tier` and `endpoint` fields. Commit `5be63fe` (rubric), commit `42ee421` (bulk-tag remaining 61 papers).
- Biomarker / intervention split: `recommendations.md` now has Part 1 (interventions, with tier tags) and Part 2 (biomarkers, with target ranges + the intervention that moves each + a marker-vs-cause caveat per metric). Commit `5be63fe`.

**Verified factual fixes** (commit `e2c8e0f`):
- Helgerud 4x4 VO2max corrected from 13% to 7.2% across 5 wiki files. Population caveat added (untrained 10-15%, elite <5%).
- PEARL rapamycin reframed: primary endpoint (visceral adiposity, p=0.379) not met; significant findings are sex-stratified subgroup analyses of secondary endpoints. Updated in 4 places.
- ApoB section updated with 2026 ACC/AHA dyslipidemia guideline citation and explicit "metric not drug target" distinction.
- Statin framing softened with NNT, absolute-vs-relative, and side-effect nuance.
- Cappuccio "typo" item dropped; was not a real error in the wiki.

**Coverage gaps ingested** (commits `fd97eb1`, `c646bf9`, `295d914`, `95b4c5f`, `47e6829`):
- Creatine: Chilibeck 2017 (T7 meta-analysis, +1.37 kg lean mass on top of RT in older adults).
- Sauna: Laukkanen 2015 (T6, KIHD cohort, HR 0.50 fatal CVD).
- GLP-1 hard CV endpoint: SELECT / Lincoff 2023 (T7, HR 0.80 MACE).
- Glucosamine: Ma 2019 (T6 observational, HR 0.85 CVD events; explicitly noted weaker evidence than vitamin D's negative RCT).
- HRT: Manson 2017 WHI 18-yr follow-up (T7, mortality-neutral; corrects the 2002 fearmongering).
- CoQ10: Q-SYMBIO (T7 in HF, HR 0.50 MACE; explicitly framed as HF-only, not generalizable).
- Cold exposure: Buijze 2016 (T6 RCT, soft endpoint).
- Built environment: Ulrich 1984 + Marucha 1998 + new `topics/built-environment.md`. Cited as context, not primary lever.
- Mitochondrial health: `topics/nad-mitophagy.md` rewritten as comprehensive hub covering urolithin A, NR/NMN, CoQ10, exercise.
- Lp(a): Kamstrup 2009 Mendelian randomization (T7 observational, ~2x MI risk independent of LDL-C/ApoB).
- MASLD: Younossi 2023 (T6, 30% global prevalence, CVD dominates mortality).
- Sleep apnea: SAVE 2016 (T7 primary not met; CPAP useful for symptoms, not RCT-supported for CV prevention).

**Onboarding** (commit `56e3d14`):
- README updated with reading path for Reddit-link visitors and "Contributing via AI agent" section that promotes the contribution prompts (previously only in a Reddit comment) into the repo.
- Source counts updated: now 87 sources, 146 wiki pages.

**Pending publications tracked** (commit `47e6829`):
- New `wiki/analysis/yet-to-publish.md` lists Lp(a)-lowering Phase 3 (HORIZON, OCEAN(a), ACCLAIM-Lp(a)), tirzepatide CV (SURMOUNT-MMO), trametinib+rapamycin combo, anti-IL-11 human, klotho s-KL, partial-reprogramming FIH, PEARL long-term, resmetirom outcomes, CKM-syndrome, SGLT2 in non-DM, K2/CAC, Phase 3 senolytics, XPRIZE Healthspan, EV therapies. Khavinson and BPC-157-class peptides explicitly noted as unlikely to publish at the RCT scale.

**Deep integration** (commit `1c4cdb5`):
- 11 older paper pages cross-linked to newly-ingested ones in their Related sections (sniderman/ctt/sprint <-> Lp(a) + SELECT; saeidifard/garcia-hermoso/leong <-> creatine; cappuccio sleep <-> SAVE CPAP; pischon waist <-> MASLD + SELECT; urolithin-A and NR-longcovid <-> Q-SYMBIO; VITAL <-> glucosamine; semaglutide-epi-age <-> SELECT).
- `wiki/index.md` surfaces all three analysis pages and the new built-environment topic in the orientation reading path.

**Reddit post artifact** (commit `???`):
- `posts/biohackers.md` rewritten to fix the propagated factual errors (Helgerud, PEARL, ApoB), reduce AI cadence, integrate the new evidence (creatine, sauna, SELECT, Lp(a), HRT, CoQ10, MASLD, SAVE), and update counts (75->87 sources, 131->146 pages).

**Items NOT acted on (intentional):**
- PR #1 from `seeforschauer` was kept closed. The author is the same person as Reddit's `tiohlongm`. Their built-environment wiki has reasonable underlying primary sources (Ulrich, Marucha) but a parallel domain with parallel conventions in the repo wasn't worth the maintenance cost. The two strongest primary sources were ingested independently into the longevity wiki framework.
- Khavinson / epithalon peptides: Russian rodent literature, mostly unreplicated in Western labs, no human RCT pipeline. Implicitly covered by the wiki's existing "anti-aging peptides without defined trial: no RCT support" framing.

The original feedback TODO content is preserved below for the historical record.

---

## Centerpiece feedback: mjjtiffany's comment

This is the most actionable single piece of feedback in the entire thread. Two distinct structural fixes, both worth implementing in full.

> Thanks for doing this work! It looks like you're using an AI to assist you, which is great because it can read through lots of papers quickly. However, it could use some help from you to improve its reasoning. Here are my recommendations:
>
> 1. Make an "evidence maturity model" to evaluate supplements and other interventions along their journey from in vitro results to mouse study results all the way to well-run human trials. Many people treat RCTs as a "gold standard" (as if there is truly only one methodological Right Way; you can decide if you agree, or if you want your model to consider forms of evidence as highly mature). It would also be great to guide your AI model to critically evaluate the quality of studies, not just their type, to identify flaws like p-hacking.
>
> Given your large collection of interventions, as a reader it would be great to see where, on your maturity model, an individual intervention gets "stuck", for instance promising mouse model results don't translate to non-human primate results.
>
> Why this is my top recommendation: your short list rejects many interventions that showed promising results in mice, on the grounds that they don't hold up under RCT, while several of your "what's actually exciting" bullets come from mouse studies. Depending on how you define the milestones of your maturity model, "promising results in mice" is probably milestone 7 while "phase 3 equivalent RCT" is milestone 15.
>
> 2. Spell out the biomarkers you're using to judge efficacy. Your recommendation of rapamycin cites "healthspan markers", not simply all-cause mortality. I agree you should look at more markers. But what did you use to judge rapamycin a keeper, but not creatine? Creatine is extraordinarily well supported in clinical literature, so we readers can't tell if this is an oversight (not enough breadth of papers) or a conscious decision on your part to pay attention to the biomarkers that rapamycin moves and not the ones that creatine moves.
>
> This will also help your model to stop confusing interventions with measurements. It's confusing to see sleep targets (an intervention), limits to smoking and drinking (an intervention), statins (an intervention), with grip strength (a measurement, ie a biomarker), VO2 Max (another biomarker), apoB (a biomarker), etc. as recommendations. It would be better to make two sets of recommendations: biomarkers to measure and then key interventions that either move those biomarkers or move all-cause mortality.

---

## 1. Build an evidence maturity model (mjjtiffany #1)

The wiki currently treats interventions inconsistently: some mouse-only results land in "promising / exciting" while other mouse-only results are rejected as "no human evidence." A explicit ladder fixes this.

Also echoed by thelostdutchman68 ("cherry-picked endpoints, secondary results dressed up as primary") and nhouseholder ("conclusions are overly conservative based on the no long-term human evidence angle, but human lifespans are 70+ years and these compounds have only existed 5 years").

**Proposed ladder** (refine when implementing):

| Tier | Description |
|------|-------------|
| T0 | In vitro / cell culture only |
| T1 | Invertebrate (C. elegans, Drosophila) lifespan effect |
| T2 | Single mouse study, lifespan or healthspan endpoint |
| T3 | Replicated mouse studies, ITP-grade or independent labs |
| T4 | Non-human primate or large-mammal data |
| T5 | Small human trial (n < 100), surrogate endpoint |
| T6 | Phase-2/3 human RCT, surrogate endpoint (ApoB, BP, etc.) |
| T7 | Phase-3 RCT or meta-analysis, hard endpoint (MACE, all-cause mortality) |

**Action:**
- [ ] Define the tier rubric in a new methodology page (e.g. `wiki/analysis/evidence-tiers.md`). Pin the table above; refine wording.
- [ ] Tag every intervention in `recommendations.md` and `wiki/analysis/promising-reverse-aging.md` with its current maximum tier.
- [ ] Where an intervention is "stuck" between tiers, say so explicitly. Example: "Rapamycin: T3 (replicated mouse), stalled at T5 (PEARL trial negative on primary)." Per Anesketin's comment.
- [ ] Within-tier quality assessment: per mjjtiffany, audit study quality (p-hacking, sample size, primary vs secondary endpoint, absolute vs relative effect size), not just study type. Add a per-paper quality flag in `sources.json` or in the paper page frontmatter.
- [ ] FAQ on the methodology page: "Why we accept T3 evidence for partial reprogramming but reject T3 evidence for resveratrol" — make the asymmetry defensible or remove it.
- [ ] Update `biology/longevity/CLAUDE.md` to make tier-tagging part of the ingest pipeline so new papers land with a tier and a quality flag.

## 2. Split biomarkers from interventions (mjjtiffany #2)

The current `recommendations.md` mixes "do this" with "measure this." mjjtiffany's example: rapamycin is judged on healthspan markers, creatine is omitted entirely despite strong clinical literature — readers can't tell if creatine was reviewed and rejected or just missed. The biomarker/intervention conflation also drives the entire grip-strength backlash (~15 separate commenters: FrewdWoad, Background_Net7441, jazzmugz, Nicholasjh, MarkHardman99, double-thonk, cizmainbascula, Sir-Olimus, username_1839, oompa_loomper, igniteyourbones579, ktyzmr, TranquilConfusion, MiscBrahBert, Free-Competition-241).

Why these commenters are right: training grip strength does not extend life. Grip is a marker that loads heavily on overall health, lean mass, growth-hormone status, and "ability to exercise at all." Same for VO2 max — TranquilConfusion's framing is sharp: VO2 max is a fraction (oxygen burn / bodyweight), so it's strength + endurance + leanness + the absence of disease that prevents training, all rolled into one number.

**Action:**
- [ ] Restructure `recommendations.md` into two top-level sections:
  - **Biomarkers to measure** (and target ranges): grip strength, VO2 max, ApoB, waist-to-hip, CAC, BP, sleep duration, body composition. For each: what it measures, what it predicts, and explicitly *what intervention moves it* (so the reader can act on the marker).
  - **Interventions to do** (with tier per §1): resistance training, zone 2 + HIIT, sleep hygiene, statins where indicated, GLP-1s where indicated, etc. For each: target biomarker(s) it moves, tier of evidence, dose/protocol, downsides.
- [ ] For every biomarker, add a "marker vs cause" caveat: how much of the mortality association is likely confounded by survivorship / healthy-person bias. Cite TranquilConfusion's framing for VO2 max and ktyzmr's cardiologist comment for grip.
- [ ] Audit the "rejected" interventions list against the same biomarkers used to keep the "exciting" interventions. Mjjtiffany's specific challenge: justify rapamycin-in / creatine-out using the same biomarker criteria, or change the verdict.
- [ ] Add creatine as a topic page and run it through the new biomarker + tier evaluation honestly.

---

## 3. Reinforcing critique: the marker/cause confusion is the most-cited issue

Standalone comments hammering the same point as mjjtiffany #2, in case the centerpiece comment is ever lost:

- FrewdWoad (138 score): "Sicker people would have weaker grip strength as a result of their poor health, not as a cause."
- Background_Net7441 (63 score): "Stronger healthier people have higher grip strength. Increasing grip strength specifically for longevity would be a misinterpretation."
- double-thonk (18 score): "Training grip strength for longevity is like polishing your car to stop it breaking down, because shinier cars tend to be more reliable."
- cizmainbascula: "Being in a good functioning condition right now means you are less likely to die right now. Which is quite irrelevant for long term longevity."
- ktyzmr: "Cardiologists test for [grip strength]. But training grip strength doesn't make your heart healthier, just hides the symptoms."
- MarkHardman99: explicit "healthy person bias" framing — physical activity makes you healthy, but health also enables physical activity, so the development of grip strength itself is a downstream marker of health.

These all collapse into the §2 fix, but the volume itself is signal: if a casual reader walks away with one wrong takeaway, it will be "I should buy a Captains of Crush." The wiki should make the marker/cause distinction impossible to miss.

## 4. Fix specific factual errors raised in comments

Each item below has been independently fact-checked against primary sources before action. Status flags: ✅ verified error in wiki, ⚠️ partially correct, ❌ commenter wrong / not actionable.

- [ ] ✅ **Helgerud 2007 / Norwegian 4×4 VO2 max claim — confirmed wiki error, propagated in 5+ files.** divinentity is correct. Primary paper (PMID 17414804) abstract states "5.5 and 7.2%" for 15/15 and 4×4 groups respectively. Wiki currently says "+13%" in `recommendations.md:28`, `sources.json:1230`, `wiki/papers/helgerud-2007-4x4-vo2max.md` (title + body), `wiki/papers/mandsager-2018-vo2max-mortality.md:45`, `wiki/topics/cardiorespiratory-fitness.md:20`. Fix every occurrence to 7.2% and add the population caveat (untrained populations may see 10–15%, elite <5%).
- [ ] ✅ **Rapamycin / PEARL trial — confirmed.** Anesketin's claim verified against PMID 40188830 / PMC12074816. Primary endpoint (visceral adiposity) p=0.942, ηp²=0.001. Secondary endpoints significant only for lean tissue mass and self-reported pain, only in women, only on 10 mg dose. Audit how rapamycin is framed in recommendations and analysis docs; add the PEARL null result with PMID and a one-line summary of which subgroup got which secondary effect.
- [ ] ✅ **ApoB clinical guidance — confirmed.** TheWatch83's quote is verbatim from the 2026 ACC/AHA/Multisociety Dyslipidemia Guideline (CIR.0000000000001423). YoGundam's cardiologist is out of date. Update the ApoB section in the wiki to cite the 2026 guideline directly and clearly distinguish "ApoB as a risk metric (newly elevated in 2026 guidelines)" from "drugs that directly target ApoB (no approved class yet, several in trials)."
- [ ] ⚠️ **Statin framing — partially actionable.** No commenter cited a specific factual error; the objection is one of tone ("underprescribed") and missing nuance. Soften the framing to acknowledge: absolute vs relative risk reduction, NNT in primary vs secondary prevention, side-effect profile (myalgia, possible cognitive effects, lower testosterone — though most of these have weaker evidence than commenters imply). Do not cite the Amazon book links 7h4tguy posted; find the underlying primary literature.
- [x] ❌ **Cappuccio name — dropped, not a real error.** Wiki consistently spells "Cappuccio" correctly. The Reddit jokes ("Cuppuccio", "Cappuccino") were riffing on the author's surname, not flagging a typo in the post. No action needed.

## 5. Address conspicuous coverage gaps

Commenters specifically asked about these and they belong in the wiki.

- [ ] **GLP-1 agonists (semaglutide, tirzepatide).** Asked by v0idl0gic, Full-Possibility-190. Major missing topic given current obesity/metabolic data. Add a topic page.
- [ ] **Creatine.** scientia_analytica: well-supported clinical literature, omitted entirely. mjjtiffany: explicit ask "why is rapamycin in but not creatine?"
- [ ] **CoQ10.** scientia_analytica.
- [ ] **Mitochondrial health as a section.** nada8, Warm-Ad-3185.
- [ ] **HRT (hormone replacement).** v0idl0gic.
- [ ] **Fish oil / omega-3.** v0idl0gic.
- [ ] **Citrus bergamot.** v0idl0gic.
- [ ] **Glucosamine.** mast4pimp: claims data quality exceeds vitamin D.
- [ ] **Sauna and cold exposure.** pasadenapasadena: Finnish sauna literature is large and well-studied; cold exposure too.
- [ ] **Klotho.** Shive55, Nugget834: active interest, ask for current state of human work and the Jay Campbell company line.
- [ ] **Khavinson / Russian peptide research (epithalon, etc.).** Jack-o-Roses, phido3000: even if the human RCT bar isn't met, summarize the rodent data honestly and note why it has not progressed to human trials. phido3000 wrote a thoughtful long comment on this — worth using as a template for how to present "promising rodent-only" interventions without overclaiming.
- [ ] **Built environment / allostatic load.** tiohlongm cited Ulrich 1984, Marucha 1998, hospital noise, circadian lighting. (Note: tiohlongm linked their own competing wiki — do not link or copy from it; cite the same primary sources independently.)

## 6. Methodology / writing quality fixes

The "AI slop" critique is the second biggest theme after the biomarker confusion. Even commenters who liked the content disliked the format.

- Quick_Adhesiveness89: AI mimics reasoning rather than reasoning. Vitamin D summary cited as example: "supplementation in non-deficient adults" with no dose, no definition of deficient, no dosing schedule.
- thelostdutchman68: cherry-picked endpoints, relative risk presented without absolute, secondary results dressed up as primary findings, generalizing from selected cohorts.
- Quick_Adhesiveness89, happyfridays_: prompt LLMs to state load-bearing assumptions and observations before conclusions; ask explicitly for "LLM-typical distortions."
- Many commenters (Nugget834, gtwooh, Better_Leather_2214, Micslar, le_pedal, henlochimken, Familiar_Text_6913) dismissed the post on tone alone — bullet-point AI cadence, decorative formatting, marketing-style "nothing-burger" framing.

**Action:**
- [ ] Add a section to `CLAUDE.md` (or `biology/longevity/CLAUDE.md`) requiring: absolute risk reduction alongside relative, primary-endpoint distinction from secondary, dose and population for every intervention claim.
- [ ] Add a checklist for paper ingest: "primary endpoint? met? what was the absolute effect size? what population? what dose?"
- [ ] Strip AI-cadence formatting from `posts/biohackers.md` and the analysis docs: fewer headers-as-decoration, fewer bold-everything sentences, fewer "TL;DR / Bottom line" framings.
- [ ] Add a human-review pass to the ingest pipeline: every new paper gets a final read by a human contributor before it lands in `recommendations.md`.

## 7. Wiki onboarding / navigation

- BelgianGinger80: "How to use the github... to look into the other documents?" — first-time visitors land in a directory tree they can't navigate. The reply with prompt examples is good content but lives in a comment, not in the repo.
- General: the post sent traffic to the repo but the README does not orient a casual reader.

**Action:**
- [ ] Promote the contribution prompts (currently posted as a Reddit comment) into `biology/longevity/README.md` under a "Contributing via AI" section.
- [ ] Add a top-of-README reading path for a non-contributor: start at `recommendations.md`, then `wiki/analysis/promising-reverse-aging.md`, then `wiki/topics/` for depth.
- [ ] Add a "What this wiki is / is not" disclaimer: peer-reviewed primary papers, AI-assisted synthesis, not medical advice, marker-vs-intervention caveats.

## 8. Lower-priority / nice-to-have

- [ ] Add a "lifespan vs healthspan" framing note (MuscaMurum). Most readers care about healthspan; the wiki should make clear which endpoint each intervention targets.
- [ ] Address the "healthy person bias" / survivorship issue head-on in a methodology page (cizmainbascula, MarkHardman99, double-thonk).
- [ ] Consider a "user-reported experiences" section flagged clearly as anecdote, not evidence — because commenters keep volunteering it (NMN, MOTS-C, NAD+, rapamycin side effects). Either curate or explicitly reject; current wiki silence on lived experience pushes readers to anecdote-heavy subreddits anyway.
- [ ] Address rapamycin side effects directly: chronic infections (CapriKitzinger), bladder infections, immunosuppression (johnolivers_hamster). Currently a "promising" pick without honest downside accounting.

---

## What we are NOT acting on

- Generic "this is AI slop" dismissals with no specific critique. The format objection is captured under §5; bare dismissals are not.
- Pure shill accusations (statin shill, Pfizer shill, Rapamycin rep). Not actionable.
- The competing-wiki link from tiohlongm. Cite the underlying primary sources independently if added.
- Anti-vaccine / "clot shot" / conspiracy framings.
