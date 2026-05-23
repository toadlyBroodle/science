# sst-wiki-curator evolution SPEC (driven from ~/Dev/science/)

> Project-scoped spec governing the evolution of the sst-wiki-curator skill. This repo (`~/Dev/science/`) is the validation testbed: each phase ships as one (or a few) targeted edits to the working skill copy, validated against the wikis in this repo (`aliens/`, `bpu/`, `biology/longevity/`, `comsci/ai-empowerment/`, `comsci/edge-llm/`). When all phases are closed, the working skill is promoted back to `~/Dev/skill-set/skills/research/sst-wiki-curator/SKILL.md` via `sst-sanitize-transferable` + `/sst-promote-skill-proposal`. Read this file and `TODO.md` end-to-end before any edit; update both in the same commit as the change.

## Goal

Make the wiki-curator skill produce wikis that are more useful, more powerful, and better organized at scale, by absorbing patterns observed in five real-world wikis (this repo) but not yet codified in the canonical spec. Specifically: surface synthesis docs above the per-subdir taxonomy, formalize the iteration workspace, codify domain-schema extension, give middle-variant wikis automated lint, and stop being aspirational about contradiction handling and variant boundaries.

## Architecture / stack (one-liner each)

- Working copy: `~/Dev/science/.claude/skills/ssp-wiki-curator/SKILL.md` — proprietary copy edited each cycle. Frontmatter declares `transferable: sst-wiki-curator`. Created at cycle bootstrap from the current canonical.
- Canonical home: `~/Dev/skill-set/skills/research/sst-wiki-curator/SKILL.md` — untouched until promote time (sanitize + promote-skill-proposal flow).
- Validation: every change illustrated by, or compatible with, at least one of the five reference wikis in this repo.
- Variant model: stays at three (minimal / middle / scripted); phases extend the model without adding a fourth.
- Modes: stays at three (scaffold / ingest / maintain) except Phase 7 which adds a fourth `umbrella` mode.
- Commit shape: each cycle ships `docs/SPEC.md` + `docs/TODO.md` + the working SKILL.md edit (+ any worked-example updates inside the wiki dirs) in one science/ commit. No edits to `~/Dev/skill-set/` until the final promote step. Commit messages follow this repo's `CLAUDE.md` voice (no AI attribution, no em dashes, imperative subject).

## Phases

### Phase 1: synthesis page kind + root-level docs

`longevity/wiki/recommendations.md`, `longevity/wiki/analysis/evidence-tiers.md`, and `aliens/grokipedia_ufo_alien_reference.md` are the most-read pages in their wikis but live above or outside the per-subdir taxonomy. The spec currently has no name for them. Introduce a `synthesis` page kind and document the promotion path (topic → synthesis → root-level link).

- [x] 1.1 [medium] Add `synthesis` to the `kind:` enum in §File conventions and describe what goes in it (high-level orientation, ranked recommendations, rubric pages, master corpus indexes).
- [x] 1.2 [medium] Document the promotion criteria: when a topic page is read more than the catalog OR an analysis page becomes the answer to "where do I start," promote to `synthesis` and link from `wiki/index.md`'s top.
- [x] 1.3 [easy] Add a synthesis-page template (front matter + sections: TL;DR, ranked list / rubric, sources) to Mode A scaffold output.
- [x] 1.4 [easy] Cite `recommendations.md`, `evidence-tiers.md`, and `grokipedia_ufo_alien_reference.md` as worked examples in the spec.

**Phase 1 completed 2026-05-23.** Added `synthesis` as a first-class `kind:` value (§File conventions YAML block), introduced the §Synthesis page kind subsection describing the three page shapes (ranked recommendations, evidence rubric, master corpus index) and three promotion criteria, added synthesis front-matter field `covers:`, added the A.5b synthesis-page template to Mode A with the TL;DR / ranked-list / sources structure, and cited `biology/longevity/recommendations.md`, `biology/longevity/wiki/analysis/evidence-tiers.md`, and `aliens/grokipedia_ufo_alien_reference.md` as worked examples throughout.

- Changes: `.claude/skills/ssp-wiki-curator/SKILL.md` (§File conventions, §Synthesis page kind, §Mode A A.5b), `docs/SPEC.md`, `docs/TODO.md`.
- Test delta: n/a (prose skill, no automated tests).

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [ ] 1.5 [easy] [should-fix] [batch-sizing] `docs/TODO.md:## Next up` — dev-skill input ~56k tokens (medium undersize threshold 100k); window-target stated ~200k but actual filled ~28%; queue held 5+ compatible medium/hard items (SPEC 4.1, 8.1, 11.1, 12.1, plus 3.1/6.1/7.1/13.1). Prior 10.3 fix (pick ≥2 medium items) was applied but did not close the gap — this prose skill's pages are short enough that picking additional items alone will not reach threshold. Proposed fix: next cycle, bundle enough medium/hard items to push context above 100k, or acknowledge that a pure-prose skill's natural fill is structurally below medium threshold and adjust the difficulty tag to [easy] for future cycles.

### Phase 2: drafts/ working layer

`aliens/drafts/reddit-r-ufos-post.md` exists with no schema coverage; the LLM has nowhere clean to iterate on long-form synthesis before committing to a wiki page. Formalize `drafts/` as an optional directory between `raw/` and `wiki/`.

- [x] 2.1 [medium] Spec a `drafts/` directory in §Three-layer architecture as an optional fourth layer: LLM scratchpad for long-form synthesis, never cross-referenced from `wiki/`, never an authoritative claim source.
- [x] 2.2 [medium] Define promotion criteria (draft → wiki page: when stable, sourced, and cross-referenced) and prune criteria (drafts unedited for N maintain-passes get flagged for removal in `LINT-REPORT.md`).
- [x] 2.3 [easy] Add `drafts/` to Mode A scaffold output as an empty directory with a one-line README explaining its purpose.
- [x] 2.4 [easy] Add `drafts/` to the `.gitignore` template (commented out by default; user opts in to gitignore vs commit).

### Phase 3: codified domain-schema extension pattern

`longevity` added `evidence_tier: T0-T7` + `endpoint: primary_met|...`, `edge-llm` adds VRAM-with-context-length, `ai-empowerment` adds maturity/cost/access tiers. Each was invented ad-hoc. Codify the pattern so the next domain doesn't reinvent it.

- [ ] 3.1 [hard] Write a new §"Extending the schema for your domain" section: pick 1-3 categorical YAML fields, write a rubric page (sibling to `index.md`, kind `synthesis`), add corresponding lint check, document the field in the schema spec.
- [ ] 3.2 [medium] Use longevity's `evidence_tier` as the canonical worked example: include the actual T0-T7 rubric snippet, show how it appears in paper front matter, show how it appears in topic-page aggregation.
- [ ] 3.3 [medium] Add a `domain-fields:` block to the schema-spec template so any wiki declares its extensions in one place (visible to the LLM on every read).

### Phase 4: categorical ranking → navigation primitive

Once a domain field exists in front matter, `index.md` and synthesis pages can aggregate by it (tier-weighted reading paths, maturity-sorted lists, cost-friction-filtered tool tables). Document the feedback loop so the pattern is reused.

- [ ] 4.1 [medium] Spec the loop in a new §"Aggregating by domain field": domain field → navigation axis → reading path. Three reference examples (longevity evidence_tier, edge-llm benchmark maturity, ai-empowerment cost/access).
- [ ] 4.2 [easy] Add an "aggregation by domain field" snippet to the scaffolded `index.md` template (commented out by default; user uncomments after adding a domain field in Phase 3).

### Phase 5: reading paths / guided tours

`longevity/wiki/index.md` opens with a 6-step "Grok the field" path. The spec currently prescribes only an alphabetized catalog. Make guided tours a documented pattern.

- [x] 5.1 [easy] Add §"Reading paths" to the schema-spec template documenting the optional ordered tour pattern at the top of `index.md` (3-7 numbered steps with one-line rationale each).
- [x] 5.2 [easy] Update the scaffolded `index.md` skeleton to include `## If you're new here, read in this order:` stub above the per-subdir catalog.

### Phase 6: middle-variant lint script template

The middle variant has no automated lint; every maintain pass re-derives the same checks via LLM judgment. Ship a ~100 LoC `lint.py` so middle-variant wikis get fast, deterministic lint.

- [ ] 6.1 [hard] Write a middle-variant `lint.py` template (~100 LoC, no dependencies beyond stdlib): checks broken relative links, missing index entries, orphan files, optional YAML front-matter requirements, empty pages.
- [ ] 6.2 [medium] Add Mode A step A.6.5 that drops the template into middle-variant scaffolds (skipped for minimal; the scripted variant uses its own fuller `lint.py`).
- [ ] 6.3 [easy] Add §"Lint output spectrum" documenting the three lint paths (LLM judgment for minimal, script + LINT-REPORT.md for middle, script + log-only for scripted).

### Phase 7: umbrella / super-wiki mode

`~/Dev/science/` holds 5 wikis with no master catalog. Add a fourth mode `umbrella` that walks sibling wikis and writes a parent-level index.

- [ ] 7.1 [hard] Spec Mode D `umbrella <parent-dir>`: walks child directories, identifies which are wikis (by presence of `index.md` + schema spec), reads each one's variant + page count + last-ingest date, writes/refreshes `<parent-dir>/index.md` with one row per child wiki.
- [ ] 7.2 [medium] Define the umbrella-index template (`# <Parent> wikis`, table: name | variant | pages | last-ingest | one-line description).
- [ ] 7.3 [easy] Update `argument-hint:` frontmatter to include the new umbrella invocation form.

### Phase 8: variant-boundary assertion in lint

`comsci/ai-empowerment/` and `comsci/edge-llm/` call themselves "middle variant" in their schema specs but have no raw dumps yet; their actual implementation is closer to minimal-scaffold. Make variant claims testable.

- [ ] 8.1 [medium] Add a variant-check assertion to lint (both LLM-judgment and scripted): scripted requires `sources.json`; middle requires `raw/` with per-source subdirs; minimal has flat `raw/`. Mismatch is a `[review]` finding, not auto-fixed.
- [ ] 8.2 [easy] Mirror the assertion in the scripted-variant `lint.py` (longevity-style) so the check stays in sync across both lint paths.

### Phase 9: reconcile LINT-REPORT.md vs script exit code

Spec §C.3 says every maintain pass writes `LINT-REPORT.md`. Longevity skips this in favor of script exit code + log.md entry. Two workflows exist; document both.

- [x] 9.1 [easy] Update §C.3 to make `LINT-REPORT.md` conditional: minimal and middle write it; scripted records lint result in `log.md` only (the script's exit code is the report).
- [x] 9.2 [easy] Add explicit "don't write both — pick one path per wiki and stick with it" guidance with a one-line rationale.

### Phase 10: optional source-papers table at index foot

`bpu/wiki/index.md` ends with a venue/year/authors table for every source paper. Cheap, useful, surfaces provenance at a glance. Document as an optional pattern.

- [x] 10.1 [easy] Add §"Optional: source-papers table" to the spec's index.md documentation, with the bpu table layout as a worked example.

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [x] 10.2 [easy] [should-fix] `SKILL.md:676` — maintain-pass completion criteria lists `LINT-REPORT.md` unconditionally; §C.3 (added this cycle as 9.1) explicitly exempts scripted wikis. An LLM running a scripted maintain pass sees the criterion as unmet and writes `LINT-REPORT.md` despite the §C.3 prohibition, defeating the 9.1/9.2 fix. Proposed fix: append "(minimal/middle only)" to the `LINT-REPORT.md` criterion, or split into two variant-specific lists.
- [x] 10.3 [easy] [should-fix] [batch-sizing] Dev cycle input ~66k tokens (medium-difficulty undersize threshold: 100k); window-target stated ~220k but actual filled ~30% of threshold. Queue had 6+ unclaimed medium-compatible items. Proposed fix: at next `/sst-dev-cycle`, claim at least 2 more `[medium]` items from `## Next up` to fill the medium window.

### Phase 11: contradiction handling with a worked example

Spec §Contradiction handling is aspirational — only longevity actually uses it. Either ground it in a real disagreement or downgrade.

- [ ] 11.1 [medium] Pull one real contradiction from the longevity corpus (or another wiki) and embed it as a worked example in §Contradiction handling, showing both source citations and the resolution prose.
- [ ] 11.2 [easy] Add a "skip if your domain doesn't have contested claims" softener so wikis that don't need this section don't pretend to.

### Phase 12: adjacent-patterns filter ("is this even a wiki?")

`bible/` (4 flat comparison files), `astronomy/` (research project with notebook), `moon-explore/` (project tracker) all sit under `~/Dev/science/` but aren't wikis. The spec should help disambiguate before the user (or agent) tries to wiki-ify the wrong artifact.

- [ ] 12.1 [medium] Add §"Adjacent patterns, not wikis": comparative-prose folders, research-output folders with notebooks, project trackers, single-document deep dives. For each, name the pattern, give a one-line "use X instead" pointer, and reference a concrete example from `~/Dev/science/`.
- [ ] 12.2 [easy] Add a one-question gate at the top of Mode A: "Is this prose knowledge accumulated across many sources?" — if no, redirect per the §Adjacent patterns guidance instead of scaffolding.

### Phase 13: personal vs publishable profiles

`longevity` is publish-quality (full front matter, wikilinks, scripted lint, LICENSES.md); `bpu`/`aliens` are personal-grade (no front matter, relative links, LLM lint). The spec treats all wikis identically. Add a `profile:` axis orthogonal to `variant:`.

- [ ] 13.1 [hard] Spec a `profile:` axis: `personal` (defaults: relative links, no front matter, LLM lint, no license tracking) vs `publishable` (defaults: wikilinks, full front matter, scripted lint, license tracking).
- [ ] 13.2 [medium] Extend the §"The three variants" table with a profile column and document how profile interacts with variant (e.g. minimal+publishable is unusual; scripted+personal is normal for personal research notes).
- [ ] 13.3 [medium] Add a `profile:` question to Mode A alongside the variant question; default to `personal` for minimal/middle, `publishable` for scripted (changeable on request).

### Phase 14: promote back to skill-set

After all of Phases 1-13 are closed, the working SKILL.md in `~/Dev/science/.claude/skills/ssp-wiki-curator/` has accumulated 13 phases of refinements. Promote it back into the canonical transferable at `~/Dev/skill-set/skills/research/sst-wiki-curator/SKILL.md`.

- [ ] 14.1 [medium] Run `sst-sanitize-transferable` over the working SKILL.md; resolve any `must-fix` findings (proprietary leakage, banned terms). `nit` findings are reviewed and either accepted or rewritten.
- [ ] 14.2 [easy] Invoke `/sst-promote-skill-proposal` to copy the sanitized SKILL.md over the canonical at `~/Dev/skill-set/skills/research/sst-wiki-curator/SKILL.md`. Bump the `version:` frontmatter field (1.0.1 → 1.1.0 since the schema gained new page kinds, profile axis, umbrella mode).
- [ ] 14.3 [easy] Commit the promotion in `~/Dev/skill-set/` with an imperative-mood subject; verify `bin/validate-frontmatter.py` passes; run `bin/install-skills.sh -y --force` so `~/.claude/skills/sst-wiki-curator/` picks up the new version.
- [ ] 14.4 [easy] Final science-side commit closing 14.1-14.3 in `docs/SPEC.md` + `docs/TODO.md`; archive this spec as `docs/SPEC-archive-wiki-curator-evolution.md` if you want to keep the testbed working directory clean for the next skill-evolution project.

## Deferred / out of scope

- **External-feedback ingestion workflow** (brainstorm item #8). Longevity's recent log entries show responding to Reddit critique with schema patches; this is a recurring real-world workflow but lower-impact than the codified-extension and synthesis-page changes. Revisit once Phases 1, 3, 4 land — those reshape the schema-extension and synthesis surface that any feedback-loop would update.
- **Cross-wiki shared scripts** (sibling-skill of Phase 7). Sharing `lint.py` across wikis under a parent dir is a natural extension of umbrella mode but adds path-resolution complexity. Revisit if umbrella mode sees real use.

## Glossary (project-specific terms)

- **variant**: minimal / middle / scripted; the automation tier of a wiki (Phase 6, 8).
- **profile**: personal / publishable; the polish tier of a wiki (Phase 13). Orthogonal to variant.
- **page kind**: paper / topic / analysis / entity / event / concept / **synthesis** (new in Phase 1).
- **domain field**: a categorical YAML front-matter field specific to one domain (longevity's `evidence_tier`, edge-llm's `benchmark_context`). Codified in Phase 3.
- **navigation axis**: a domain field used to aggregate or order pages (Phase 4).
- **reading path**: an ordered sequence of pages at the top of `index.md` for newcomers (Phase 5).
- **umbrella mode**: Mode D, walks sibling wikis under a parent dir and writes a master index (Phase 7).

---

### How this file evolves

Same contract as `~/Dev/skill-set/docs/SPEC.md`:

- A skill closes an item by flipping `- [ ]` → `- [x]` in the same commit as the `SKILL.md` edit.
- When all items in a phase are checked, append a one-paragraph "completed" block with file-citation bullets to that phase. Don't delete the checklist.
- New work surfaced mid-cycle goes to `TODO.md`'s "Next up", not directly here.
- Sub-item IDs (`<phase>.<n>`) are stable and never renumbered; gaps from removed items are intentional. Inserts use letter suffixes.
- Difficulty labels (`[easy]` / `[medium]` / `[hard]`) route model + effort tier per the skill-set framework's `max(item_tier, skill_floor)` rule.
