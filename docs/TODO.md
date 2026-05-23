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

- Close 7.9+8.1+8.2+11.1+12.1+12.2: variant-boundary lint check, NAD+ contradiction worked example, adjacent-patterns section + Mode A gate — by sst-dev-cycle at 2026-05-24T02:00:00Z
- Fix 7.7+7.8: scope check_orphans to kind==topic only (eliminates paper double-reporting) + annotate SPEC 7.6 batch-pick omission — by sst-dev-cycle at 2026-05-24T01:00:00Z
- Fix 7.4+7.5: correct Mode D variant inference (sources.json required for scripted) + add check_unlinked_papers to middle lint.py (closes items 1-5 coverage gap) — by sst-dev-cycle at 2026-05-24T00:30:00Z
- Close 6.1+6.2 (middle-variant lint.py template + A.6.5 step) and 7.1+7.2+7.3 (Mode D umbrella spec + template + argument-hint) — by sst-dev-cycle at 2026-05-23T23:00:00Z
- Fix 4.4+4.5: annotate access_tier as prospective, fix cost_tier to reflect freeform wiki values (not a categorical enum) — by sst-dev-cycle at 2026-05-23T22:00:00Z
- Fix 4.3: change benchmark_maturity "adds" to "could add" (prospective illustration, not an existing field) — by sst-dev-cycle at 2026-05-23T21:30:00Z
- Add §Aggregating by domain field (4.1) + aggregation snippet to index.md scaffold template (4.2) — by sst-dev-cycle at 2026-05-23T20:00:00Z
- Close 3.4 (rename Step 5 label to synthesis-page aggregation) + 3.5 (re-tier 6.1/7.1/13.1 hard→medium) — by sst-dev-cycle at 2026-05-23T19:00:00Z
- Close Phase 3 (3.1 + 3.2 + 3.3) — add §"Extending the schema for your domain" with longevity evidence_tier worked example and `domain-fields:` schema-spec template block — by sst-dev-cycle at 2026-05-23T18:00:00Z
- Append lint-spectrum update reminder to SPEC 6.2; close 11.3 — by sst-dev-cycle at 2026-05-23T16:00:00Z

## Next up (queued for next cycle)

<!--
  Top item is the next cycle's work unless the user redirects.
  Format: - [<difficulty>] <one-line>. Reason: <spec id, supervisor verdict, user request>
  Ordered by priority (highest-impact-low-effort first), not by SPEC phase number.
-->

- [medium] Spec the `profile:` axis (personal vs publishable) orthogonal to `variant:`; extend §"The three variants" table; add to Mode A. Reason: SPEC 13.1 + 13.2 + 13.3 — biggest mental-model change; do last so other phases inform the profile defaults; re-tiered hard→medium (3.5 advisory).
