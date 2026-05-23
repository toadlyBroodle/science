# sst-wiki-curator evolution TODO (handoff doc, driven from ~/Dev/science/)

> Cross-cycle state for the sst-wiki-curator skill evolution. Every cycle reads this + `SPEC.md` end-to-end before picking work; updates both in the same commit as the working-copy `SKILL.md` edit at `~/Dev/science/.claude/skills/ssp-wiki-curator/SKILL.md`. Canonical transferable at `~/Dev/skill-set/skills/research/sst-wiki-curator/SKILL.md` is untouched until Phase 14 promote.

## In flight

<!--
  One line per currently-running skill:
  - [<skill-name> @ <utc-iso>] <one-line>
  Empty when nothing is in progress.
-->

## Just shipped (last cycle)

<!--
  Newest first. Format:
  - <one-line summary> — by <skill> at <utc-iso>
  Correlate to commits via: git log --oneline --grep '<keyword>'
  Trim to last 10.
-->

- Fix 4.3: change benchmark_maturity "adds" to "could add" (prospective illustration, not an existing field) — by sst-dev-cycle at 2026-05-23T21:30:00Z
- Add §Aggregating by domain field (4.1) + aggregation snippet to index.md scaffold template (4.2) — by sst-dev-cycle at 2026-05-23T20:00:00Z
- Close 3.4 (rename Step 5 label to synthesis-page aggregation) + 3.5 (re-tier 6.1/7.1/13.1 hard→medium) — by sst-dev-cycle at 2026-05-23T19:00:00Z
- Close Phase 3 (3.1 + 3.2 + 3.3) — add §"Extending the schema for your domain" with longevity evidence_tier worked example and `domain-fields:` schema-spec template block — by sst-dev-cycle at 2026-05-23T18:00:00Z
- Append lint-spectrum update reminder to SPEC 6.2; close 11.3 — by sst-dev-cycle at 2026-05-23T16:00:00Z
- Close batch-sizing advisory (1.5), add contradiction-handling skip softener (11.2), add lint output spectrum section (6.3) — by sst-dev-cycle at 2026-05-23T14:00:00Z
- Add synthesis page kind (Phase 1 1.1-1.4), fix maintain-pass criteria (10.2), close batch-sizing advisory (10.3) — by sst-dev-cycle at 2026-05-23T12:00:00Z
- Add drafts/ optional layer, reading paths pattern, lint-output reconciliation, source-papers table to ssp-wiki-curator (SPEC 2.1-2.4, 5.1-5.2, 9.1-9.2, 10.1) — by sst-dev-cycle at 2026-05-23T00:00:00Z
- Bootstrap: SPEC + TODO moved to ~/Dev/science/docs/; proprietary working copy ssp-wiki-curator created at .claude/skills/ssp-wiki-curator/SKILL.md (body identical to canonical sst-wiki-curator v1.0.1); Phase 14 (promote-back) appended to SPEC — by claude (manual) at 2026-05-23
- SPEC + TODO scaffolded for the 13-phase sst-wiki-curator evolution plan — by claude (manual) at 2026-05-23

## Next up (queued for next cycle)

<!--
  Top item is the next cycle's work unless the user redirects.
  Format: - [<difficulty>] <one-line>. Reason: <spec id, supervisor verdict, user request>
  Ordered by priority (highest-impact-low-effort first), not by SPEC phase number.
-->

- [easy] [should-fix] 4.4 `.claude/skills/ssp-wiki-curator/SKILL.md:342` — access_tier claimed as existing in ai-empowerment but no page uses it; annotate as prospective like 4.3's fix — review of bb93819 (group with example3-field-accuracy)
- [easy] [should-fix] 4.5 `.claude/skills/ssp-wiki-curator/SKILL.md:342` — cost_tier enum free|freemium|paid mismatches wiki's freeform values; update enum or note it's a template suggestion — review of bb93819 (group with example3-field-accuracy)
- [medium] Write the middle-variant `lint.py` template (~100 LoC, stdlib only) + wire into Mode A.6.5. Reason: SPEC 6.1 + 6.2 — fills the biggest middle-variant gap; re-tiered hard→medium (3.5 advisory: prose-only edits cannot meet hard-tier 200k threshold).
- [medium] Spec Mode D `umbrella <parent-dir>` + template + argument-hint update. Reason: SPEC 7.1 + 7.2 + 7.3 — useful at exactly 3+ sibling wikis (science/ already there); re-tiered hard→medium (3.5 advisory).
- [medium] Add variant-boundary assertion to lint (both LLM-judgment and scripted) + mirror in scripted `lint.py`. Reason: SPEC 8.1 + 8.2 — surfaces ambiguous variant claims observed in comsci wikis.
- [medium] Embed a real contradiction-resolution worked example from longevity in §Contradiction handling. Reason: SPEC 11.1 — grounds aspirational guidance.
- [medium] Add §"Adjacent patterns, not wikis" + one-question gate at top of Mode A. Reason: SPEC 12.1 + 12.2 — prevents wiki-ifying bible/, astronomy/, moon-explore/-shaped artifacts.
- [medium] Spec the `profile:` axis (personal vs publishable) orthogonal to `variant:`; extend §"The three variants" table; add to Mode A. Reason: SPEC 13.1 + 13.2 + 13.3 — biggest mental-model change; do last so other phases inform the profile defaults; re-tiered hard→medium (3.5 advisory).
