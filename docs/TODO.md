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

- Add drafts/ optional layer, reading paths pattern, lint-output reconciliation, source-papers table to ssp-wiki-curator (SPEC 2.1-2.4, 5.1-5.2, 9.1-9.2, 10.1) — by sst-dev-cycle at 2026-05-23T00:00:00Z
- Bootstrap: SPEC + TODO moved to ~/Dev/science/docs/; proprietary working copy ssp-wiki-curator created at .claude/skills/ssp-wiki-curator/SKILL.md (body identical to canonical sst-wiki-curator v1.0.1); Phase 14 (promote-back) appended to SPEC — by claude (manual) at 2026-05-23
- SPEC + TODO scaffolded for the 13-phase sst-wiki-curator evolution plan — by claude (manual) at 2026-05-23

## Next up (queued for next cycle)

<!--
  Top item is the next cycle's work unless the user redirects.
  Format: - [<difficulty>] <one-line>. Reason: <spec id, supervisor verdict, user request>
  Ordered by priority (highest-impact-low-effort first), not by SPEC phase number.
-->

- [medium] Add `synthesis` page kind to the `kind:` enum + describe promotion path (topic → synthesis → root). Reason: SPEC 1.1 + 1.2 — names the most-read page class in mature wikis (longevity recommendations.md, aliens grokipedia ref).
- [easy] Add synthesis-page template to Mode A scaffold output + cite the three worked examples. Reason: SPEC 1.3 + 1.4 — completes Phase 1 after 1.1/1.2 land.
- [hard] Write §"Extending the schema for your domain" with longevity's evidence_tier as worked example; add `domain-fields:` block to schema-spec template. Reason: SPEC 3.1 + 3.2 + 3.3 — the most leveraged change; every future scripted/middle wiki benefits.
- [medium] Spec the domain-field → navigation-axis → reading-path loop with three reference examples. Reason: SPEC 4.1 — depends on Phase 3 landing first; pairs naturally as a follow-up.
- [hard] Write the middle-variant `lint.py` template (~100 LoC, stdlib only) + wire into Mode A.6.5 + document the lint-output spectrum. Reason: SPEC 6.1 + 6.2 + 6.3 — fills the biggest middle-variant gap.
- [hard] Spec Mode D `umbrella <parent-dir>` + template + argument-hint update. Reason: SPEC 7.1 + 7.2 + 7.3 — useful at exactly 3+ sibling wikis (science/ already there).
- [medium] Add variant-boundary assertion to lint (both LLM-judgment and scripted) + mirror in scripted `lint.py`. Reason: SPEC 8.1 + 8.2 — surfaces ambiguous variant claims observed in comsci wikis.
- [medium] Embed a real contradiction-resolution worked example from longevity in §Contradiction handling + add skip-if-not-applicable softener. Reason: SPEC 11.1 + 11.2 — grounds aspirational guidance.
- [medium] Add §"Adjacent patterns, not wikis" + one-question gate at top of Mode A. Reason: SPEC 12.1 + 12.2 — prevents wiki-ifying bible/, astronomy/, moon-explore/-shaped artifacts.
- [hard] Spec the `profile:` axis (personal vs publishable) orthogonal to `variant:`; extend §"The three variants" table; add to Mode A. Reason: SPEC 13.1 + 13.2 + 13.3 — biggest mental-model change; do last so other phases inform the profile defaults.
