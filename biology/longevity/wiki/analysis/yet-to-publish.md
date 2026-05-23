---
id: yet-to-publish
title: "Pending publications and trial readouts to ingest"
kind: analysis
---

# Yet-to-publish: pending readouts to track and ingest

A running list of trials and readouts the wiki should ingest *when*
they publish, with a one-line answer to "what question would the
result answer?" Listed roughly by expected impact and proximity.

This file is the working list for the next ingest pass; it is not
itself evidence. When a paper publishes, move it from this list into
`sources.json` + a paper page, and remove it here.

## High impact, near-term

### Lp(a)-targeted lowering trials (cardiovascular hard endpoints)
The genetic evidence for Lp(a) causing CVD
([[papers/kamstrup-2009-lpa-mendelian]]) is decades old, but no
approved drug specifically lowers Lp(a). The current generation of
antisense and siRNA Lp(a)-lowering therapies are testing whether
pharmacologically lowering Lp(a) reduces MACE.

- **HORIZON** (pelacarsen, Novartis, NCT04023552). Phase 3 in
  patients with established CVD and elevated Lp(a). Primary
  endpoint MACE; readout expected 2025.
- **OCEAN(a)-Outcomes** (olpasiran, Amgen, NCT05581303). Phase 3
  in established ASCVD with Lp(a) ≥200 nmol/L. Primary endpoint
  MACE; readout expected 2026–2027.
- **ACCLAIM-Lp(a)** (lepodisiran, Lilly, NCT06292013). Phase 3
  cardiovascular outcomes for the Lilly siRNA candidate.

If any of these reads out positive, Lp(a) lowering becomes the
fourth modifiable lipid lever (after LDL-C, HDL-C management,
triglycerides). If they read out null, the niacin / cholesteryl-
ester transfer protein pattern repeats and Lp(a) joins the list of
plausibly-causal-but-not-pharmacologically-actionable risk factors.

### GLP-1 / GIP cardiovascular outcomes (tirzepatide and beyond)
- **SURMOUNT-MMO** (tirzepatide, Lilly, NCT05556512). MMO
  cardiovascular outcomes in obesity; will tell us if tirzepatide
  matches or exceeds semaglutide's SELECT result
  ([[papers/select-2023-semaglutide-cv-outcomes]]).
- **SELECT-2** or other phase-3 follow-ups in primary-prevention
  obese populations (no prior CVD).

### Trametinib + rapamycin Phase II combination trial
[[papers/trametinib-rapamycin-itp-2025]] showed +27–29% mouse
lifespan extension additive over rapamycin alone. Both drugs are
FDA-approved separately; the human combination trial is the
shortest mouse-to-human path of any 2024-2026 longevity story.

### Anti-IL-11 antibody human readouts
[[papers/il11-inhibition-2024]] showed +25% mouse lifespan. The
anti-IL-11 antibody (e.g. Boehringer's BI 765423) is in human
trials for fibrotic lung disease. Even a fibrosis readout would
provide indirect evidence on the mechanism in humans; a longevity
readout is years away.

### Klotho s-KL Phase 1
Klotho Neurosciences' s-KL gene therapy moved toward IND in 2024;
first-in-human readout expected 2026–2027. The mouse evidence
([[papers/klotho-skl-aav-2025]]) is +20% lifespan in males.

### Partial reprogramming first-in-human entries
NewLimit and Retro Biosciences
([[papers/retro-precision-reprog-2025]]) are the closest to FIH.
Any IND or Phase 1 readout would be the first human partial-
reprogramming data and would substantially restructure the
[[analysis/promising-reverse-aging]] tier ranking.

## Medium impact

### Long-term (5-year+) PEARL rapamycin follow-up
PEARL Phase 1 ([[papers/pearl-rapamycin-2025]]) was 48 weeks and
missed primary endpoint. A longer-duration follow-up or repeat
trial with prespecified endpoints (lean tissue mass in women on 10
mg, the secondary signal that did emerge) would convert PEARL's
subgroup finding into either a confirmed or rejected effect.

### Resmetirom long-term outcomes (MAESTRO follow-up)
Resmetirom was approved in 2024 for MASH with significant
fibrosis. Long-term outcome data (cardiovascular events, all-cause
mortality, hepatic decompensation) is the question; the approval
trial used surrogate histological endpoints.

### CKM-syndrome-targeted trials
The 2026 ACC/AHA dyslipidemia guideline framing of cardiovascular-
kidney-metabolic (CKM) syndrome is new. Trials specifically
designed around CKM endpoints (vs. classical cardiology endpoints)
are coming.

### SGLT2 inhibitors in non-diabetic, non-HF populations
EMPACT-MI tested empagliflozin post-MI in non-diabetic patients.
Broader primary-prevention SGLT2 trials in non-diabetic adults are
ongoing.

### Vitamin K2 / aortic calcification RCTs
Multiple small RCTs are running on K2 (MK-7) and arterial
calcification scores (CAC progression, pulse-wave velocity). No
single trial is yet large enough to land in the wiki.

## Speculative / longer horizon

### Senolytic Phase 3 trials with disease endpoints
The current senolytic landscape ([[papers/senolytic-mci-ebiomed-2025]]
and others) is pilot-scale. A phase-3 senolytic with a hard endpoint
(MACE, dementia incidence, mortality) does not yet exist. Watch for
fisetin and D+Q follow-ups.

### Yamanaka-factor delivery in humans
Beyond Retro / NewLimit, several academic groups are working on
inducible OSK delivery systems. IND timelines are unclear.

### XPRIZE Healthspan winners
[[papers/xprize-healthspan]]: $101M competition. Winning teams will
publish their evidence packages; whatever wins will reshape the
wiki's "promising" tier.

### Whole-body exosome / EV therapies
[[papers/adsc-exosomes-2022]] is the closest existing paper page in
the wiki. Emerging area; watch for phase-1 readouts.

## Reddit-asked but unlikely to publish soon

The Reddit thread asked about several topics where the
"yet-to-publish" pipeline is thin:

- **Khavinson / epithalon peptides.** The Russian rodent literature
  is decades old and largely unreplicated outside Russia. Western
  human RCTs are not currently planned at scale.
- **Specific peptide stacks (BPC-157, TB-500, AOD-9604, etc.).**
  No phase-3 RCT pipeline. Most of this evidence will remain at
  the case-series / small-RCT level.

## How to use this list

When ingesting a new paper from this list:
1. Move the entry into `sources.json` with a stable id and the
   best open-access URL.
2. Add a `LICENSE_MAP` entry in `scripts/licenses.py`.
3. Write the paper page with `evidence_tier` and `endpoint`
   fields.
4. Link from the relevant topic and analysis pages.
5. **Remove the entry from this file.** This is a working list, not
   archive.
6. Append to `log.md`.

## Related
- [[analysis/promising-reverse-aging]] — current state of the
  research frontier.
- [[analysis/evidence-tiers]] — tier rubric these readouts will be
  graded against.
