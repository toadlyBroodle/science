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
- [x] 1.5 [easy] [should-fix] [batch-sizing] `docs/TODO.md:## Next up` — dev-skill input ~56k tokens (medium undersize threshold 100k); window-target stated ~200k but actual filled ~28%; queue held 5+ compatible medium/hard items (SPEC 4.1, 8.1, 11.1, 12.1, plus 3.1/6.1/7.1/13.1). Prior 10.3 fix (pick ≥2 medium items) was applied but did not close the gap — this prose skill's pages are short enough that picking additional items alone will not reach threshold. Proposed fix: next cycle, bundle enough medium/hard items to push context above 100k, or acknowledge that a pure-prose skill's natural fill is structurally below medium threshold and adjust the difficulty tag to [easy] for future cycles. **Resolved:** closed by bundling 1.5 + 11.2 + 6.3 in one easy cycle; structural fill-rate limitation acknowledged — prose-only skill edits should batch 3+ items to approach window targets.

### Phase 2: drafts/ working layer

`aliens/drafts/reddit-r-ufos-post.md` exists with no schema coverage; the LLM has nowhere clean to iterate on long-form synthesis before committing to a wiki page. Formalize `drafts/` as an optional directory between `raw/` and `wiki/`.

- [x] 2.1 [medium] Spec a `drafts/` directory in §Three-layer architecture as an optional fourth layer: LLM scratchpad for long-form synthesis, never cross-referenced from `wiki/`, never an authoritative claim source.
- [x] 2.2 [medium] Define promotion criteria (draft → wiki page: when stable, sourced, and cross-referenced) and prune criteria (drafts unedited for N maintain-passes get flagged for removal in `LINT-REPORT.md`).
- [x] 2.3 [easy] Add `drafts/` to Mode A scaffold output as an empty directory with a one-line README explaining its purpose.
- [x] 2.4 [easy] Add `drafts/` to the `.gitignore` template (commented out by default; user opts in to gitignore vs commit).

### Phase 3: codified domain-schema extension pattern

`longevity` added `evidence_tier: T0-T7` + `endpoint: primary_met|...`, `edge-llm` adds VRAM-with-context-length, `ai-empowerment` adds maturity/cost/access tiers. Each was invented ad-hoc. Codify the pattern so the next domain doesn't reinvent it.

- [x] 3.1 [hard] Write a new §"Extending the schema for your domain" section: pick 1-3 categorical YAML fields, write a rubric page (sibling to `index.md`, kind `synthesis`), add corresponding lint check, document the field in the schema spec.
- [x] 3.2 [medium] Use longevity's `evidence_tier` as the canonical worked example: include the actual T0-T7 rubric snippet, show how it appears in paper front matter, show how it appears in topic-page aggregation.
- [x] 3.3 [medium] Add a `domain-fields:` block to the schema-spec template so any wiki declares its extensions in one place (visible to the LLM on every read).

**Phase 3 completed 2026-05-23.** Added §"Extending the schema for your domain" between §File conventions and §Mode A. The new section codifies the five-step pattern (pick 1-3 categorical fields → enumerate values → write rubric synthesis page → document in schema spec → add lint check), gives anatomy criteria for a good domain field (stable, source-assignable, orthogonal, within-tier quality stays outside), and walks the longevity `evidence_tier` worked example end-to-end: T0-T7 ladder snippet, paper front-matter shape (PEARL rapamycin example), topic-page aggregation snippet, recommendations-page integration, and lint-check requirement. The §Mode A.3 schema-spec template grew an optional `Domain fields` section with a `domain-fields:` YAML block format and a real longevity example covering both `evidence_tier` and `endpoint`.

- Changes: `.claude/skills/ssp-wiki-curator/SKILL.md` (new §Extending the schema for your domain section + §Mode A.3 schema-spec template extension), `docs/SPEC.md`, `docs/TODO.md`.
- Test delta: n/a (prose skill, no automated tests).

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [x] 3.4 [easy] [should-fix] `.claude/skills/ssp-wiki-curator/SKILL.md:264` — Step 5 of the worked example is labeled "in topic-page aggregation" but demonstrates aggregation inside `wiki/analysis/evidence-tiers.md`, which is the same synthesis rubric page introduced in Step 3 (not a topic page; topic pages live in `wiki/topics/`). A reader extending a new domain follows the label literally and puts tier aggregations in `wiki/topics/*.md` instead of in the rubric synthesis page, fragmenting the navigation primitive. Proposed fix: rename Step 5 to "in synthesis-page aggregation" and add one sentence noting that the rubric and the cross-corpus aggregation usually share one synthesis page (as evidence-tiers.md does with §"Tier ladder" + §"Where each intervention currently sits").
- [x] 3.5 [easy] [should-fix] [batch-sizing] Dev skill input ~73k tokens (hard-difficulty undersize threshold: 200k; band 400-500k); window-target stated ~450k, actual filled ~16% of lower edge. Third consecutive batch-sizing undersize for this prose skill (prior: 10.3, 1.5). Queue held SPEC 4.1 [medium] explicitly tagged as Phase-3-dependent natural follow-on; not bundled. Structural pattern: prose-only skill edits cannot meet hard-tier thresholds even with multi-item Phase bundles. Proposed fix: at next cycle, bundle SPEC 4.1+4.2 with any residual prose items, AND consider re-tiering remaining hard items in this spec (6.1, 7.1, 13.1) to [medium] since they are also prose-only edits. **Resolved:** re-tiered 6.1, 7.1, 13.1 from [hard] to [medium] in SPEC and TODO; next cycle should bundle 4.1+4.2 together.

### Phase 4: categorical ranking → navigation primitive

Once a domain field exists in front matter, `index.md` and synthesis pages can aggregate by it (tier-weighted reading paths, maturity-sorted lists, cost-friction-filtered tool tables). Document the feedback loop so the pattern is reused.

- [x] 4.1 [medium] Spec the loop in a new §"Aggregating by domain field": domain field → navigation axis → reading path. Three reference examples (longevity evidence_tier, edge-llm benchmark maturity, ai-empowerment cost/access).
- [x] 4.2 [easy] Add an "aggregation by domain field" snippet to the scaffolded `index.md` template (commented out by default; user uncomments after adding a domain field in Phase 3).

**Phase 4 completed 2026-05-23.** Added §"Aggregating by domain field" between §Declaring fields in the schema spec and §Mode A. The section codifies the domain-field → navigation-axis → reading-path feedback loop, explains when each promotion step fires (aggregation added at first sort-by-field query; reading path added when aggregation becomes the newcomer entry point), and walks three reference examples end-to-end: longevity `evidence_tier` (tier-ordered listing with reading-path promotion), edge-llm `benchmark_maturity` (maturity-sorted model table without reading path), and ai-empowerment `cost_tier + access_tier` (cross-filtered free/friction tool table with reading-path promotion). Mode A.5's `wiki/index.md` skeleton grew a commented-out `<!-- domain-field-aggregation -->` block with a two-value template for users to uncomment after adding their first domain field.

- Changes: `.claude/skills/ssp-wiki-curator/SKILL.md` (new §Aggregating by domain field + Mode A.5 index.md skeleton aggregation snippet), `docs/SPEC.md`, `docs/TODO.md`.
- Test delta: n/a (prose skill, no automated tests).

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [x] 4.3 [easy] [should-fix] `.claude/skills/ssp-wiki-curator/SKILL.md:325` — Example 2 in §Aggregating by domain field asserts "`comsci/edge-llm/` adds a `benchmark_maturity` field" in present tense, but no such field exists in any edge-llm page or its `CLAUDE.md` schema spec. An LLM maintaining the edge-llm wiki reads this as a declared domain field and may attempt to aggregate by a non-existent field. Proposed fix: change "adds" to "could add" (or annotate explicitly that this is a prospective illustration, not a field that currently exists in the wiki).
- [x] 4.4 [easy] [should-fix] `.claude/skills/ssp-wiki-curator/SKILL.md:342` — Example 3 claims `access_tier` (enum: `no-signup | email | account | waitlist`) exists in `comsci/ai-empowerment/` but no page in that wiki uses it. Same false-present-tense pattern as 4.3. An LLM maintaining ai-empowerment reads this as a declared domain field and attempts to aggregate by a non-existent key. Proposed fix: annotate `access_tier` as prospective with the same pattern as 4.3 ("could add" + parenthetical).
- [x] 4.5 [easy] [should-fix] `.claude/skills/ssp-wiki-curator/SKILL.md:342` — Example 3 states `cost_tier` enum as `free | freemium | paid` but ai-empowerment pages use freeform cost strings (e.g. "low ($0-20/mo)", "usage-based ($0.10-0.40/sec)"). An LLM adding a new tool page would apply the wrong enum values. Proposed fix: update the enum to reflect actual wiki values, or add a note that this is a template suggestion for new wikis rather than a declared enum for ai-empowerment.

### Phase 5: reading paths / guided tours

`longevity/wiki/index.md` opens with a 6-step "Grok the field" path. The spec currently prescribes only an alphabetized catalog. Make guided tours a documented pattern.

- [x] 5.1 [easy] Add §"Reading paths" to the schema-spec template documenting the optional ordered tour pattern at the top of `index.md` (3-7 numbered steps with one-line rationale each).
- [x] 5.2 [easy] Update the scaffolded `index.md` skeleton to include `## If you're new here, read in this order:` stub above the per-subdir catalog.

### Phase 6: middle-variant lint script template

The middle variant has no automated lint; every maintain pass re-derives the same checks via LLM judgment. Ship a ~100 LoC `lint.py` so middle-variant wikis get fast, deterministic lint.

- [x] 6.1 [medium] Write a middle-variant `lint.py` template (~100 LoC, no dependencies beyond stdlib): checks broken relative links, missing index entries, orphan files, optional YAML front-matter requirements, empty pages.
- [x] 6.2 [medium] Add Mode A step A.6.5 that drops the template into middle-variant scaffolds (skipped for minimal; the scripted variant uses its own fuller `lint.py`). Also update the §Lint output spectrum table's Middle row (SKILL.md §Lint output spectrum) to replace "LLM judgment" with the new script path once lint.py exists.
- [x] 6.3 [easy] Add §"Lint output spectrum" documenting the three lint paths (LLM judgment for minimal, script + LINT-REPORT.md for middle, script + log-only for scripted).

**Phase 6 completed 2026-05-23.** Added middle-variant `lint.py` template (~80 executable LoC, stdlib only) as §A.6.5 in Mode A. The template checks: broken relative links, missing index entries, orphan pages (no inbound links), required front-matter fields (`id`, `title`, `kind`), bad `kind` values, and empty pages (body < 50 chars). Updated §Lint output spectrum Middle row from "LLM judgment" to "scripts/lint.py (stdlib) + LLM judgment (stale claims, contradictions, gaps)". Updated Mode C.2 to separate minimal/middle into distinct lint paths: middle variant runs `scripts/lint.py` first for deterministic checks (items 1-5) then LLM judgment for fuzzy checks (items 6-8).

- Changes: `.claude/skills/ssp-wiki-curator/SKILL.md` (§Lint output spectrum table, §Mode A.6.5, §Mode C.2), `docs/SPEC.md`, `docs/TODO.md`.
- Test delta: n/a (prose skill, no automated tests).

### Phase 7: umbrella / super-wiki mode

`~/Dev/science/` holds 5 wikis with no master catalog. Add a fourth mode `umbrella` that walks sibling wikis and writes a parent-level index.

- [x] 7.1 [medium] Spec Mode D `umbrella <parent-dir>`: walks child directories, identifies which are wikis (by presence of `index.md` + schema spec), reads each one's variant + page count + last-ingest date, writes/refreshes `<parent-dir>/index.md` with one row per child wiki.
- [x] 7.2 [medium] Define the umbrella-index template (`# <Parent> wikis`, table: name | variant | pages | last-ingest | one-line description).
- [x] 7.3 [easy] Update `argument-hint:` frontmatter to include the new umbrella invocation form.

**Phase 7 completed 2026-05-23.** Added §Mode D: umbrella as a new top-level mode section after Mode C. Mode D walks `<parent-dir>` up to two levels deep, identifies wikis by the presence of `wiki/index.md` + schema spec, infers variant from directory structure, collects name/variant/page-count/last-ingest/description per wiki, and writes/refreshes `<parent-dir>/index.md` using the umbrella-index template. The template uses a `<!-- umbrella-index: auto-generated -->` sentinel so the table block can be overwritten on re-runs without touching human-written prose above it. Updated `argument-hint:` frontmatter to include `umbrella <parent-dir>`. Worked example cites `~/Dev/science/` with its five wikis.

- Changes: `.claude/skills/ssp-wiki-curator/SKILL.md` (new §Mode D + frontmatter `argument-hint:`), `docs/SPEC.md`, `docs/TODO.md`.
- Test delta: n/a (prose skill, no automated tests).

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [x] 7.4 [easy] [should-fix] `SKILL.md:1013` — Mode D variant inference `scripts/lint.py present → scripted` conflicts with A.6.5, which drops `scripts/lint.py` into middle-variant wikis. Any middle wiki that receives the A.6.5 lint.py template will be misclassified as scripted by umbrella mode. Proposed fix: update D.1 to `scripts/lint.py` + `sources.json` → scripted; `scripts/lint.py` alone → middle; `raw/` with subdirectories → middle; otherwise → minimal.
- [x] 7.5 [easy] [should-fix] `SKILL.md:962` — Mode C.2 claims `scripts/lint.py` covers "items 1-5 above" but the lint.py template has no check for item 3 ("Papers nothing links to"); the script covers items 1, 2, 4, 5 plus empty-page. Item 3 silently drops from middle-variant lint checks. Proposed fix: either add `check_unlinked_papers` to the lint.py template, or narrow the C.2 claim to "items 1, 2, 4, 5" and include item 3 in LLM-judgment alongside items 6-8.
- [x] 7.6 [easy] [should-fix] [batch-sizing] `docs/TODO.md:## Next up` — dev-skill input ~62k tokens (medium undersize threshold: 100k; band 200-300k); window-target ~250k, actual ~25% of lower edge. Queue held 4+ compatible medium items (SPEC 8.1+8.2, 11.1, 12.1+12.2). Fourth consecutive batch-sizing undersize for this prose skill. Proposed fix: bundle 3-4 medium items from Next up at the start of the next cycle; all are prose-only SKILL.md edits with similar per-item token cost to this batch.

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [ ] 7.7 [medium] [should-fix] `SKILL.md:729` — `check_orphans()` flags all wiki pages with no inbound links, not just topic pages. After 7.5 added `check_unlinked_papers()`, a paper page with no inbound links from anyone now appears twice in lint output: as `[orphan]` from `check_orphans()` and as `[unlinked-paper]` from `check_unlinked_papers()`. Spec item 2 is "Orphan topic pages" — the function should filter for `kind == "topic"` before reporting. Proposed fix: in `check_orphans()`, skip pages where `parse_front_matter(absp.read_text(...)).get("kind") != "topic"` (or equivalently, only include pages where kind is "topic").
- [ ] 7.8 [easy] [should-fix] [batch-coherence] `docs/SPEC.md:116` — batch-pick declared "2 items @ easy" (7.4 + 7.5) but the diff contains 3 `[x]` flips (7.4, 7.5, 7.6). Item 7.6 was closed in the commit without appearing in the batch-pick declaration, violating rule (b) of the batch-coherence protocol. Proposed fix: when closing meta/advisory items alongside declared items, include them in the batch-pick block; or annotate the batch-pick rationale with "plus closing meta-item 7.6 (no implementation work, spec `[x]` flip only)."

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
- [x] 11.2 [easy] Add a "skip if your domain doesn't have contested claims" softener so wikis that don't need this section don't pretend to.

**Review follow-ups (open — schedule as the next `/sst-dev-cycle` cycle):**
- [x] 11.3 [easy] [should-fix] `SKILL.md:113` & `docs/SPEC.md:6.2` — §Lint output spectrum table says "LLM judgment" for middle (correct today, since lint.py doesn't exist yet), but SPEC 6.2 ("Add Mode A step A.6.5") has no reminder to also update this row after lint.py lands. A cycle closing 6.1+6.2 will leave middle's row stale, contradicting the newly-scaffolded script. Proposed fix: append "also update §Lint output spectrum middle row (SKILL.md §Lint output spectrum) to reflect scripts/lint.py" to SPEC 6.2's description.

### Phase 12: adjacent-patterns filter ("is this even a wiki?")

`bible/` (4 flat comparison files), `astronomy/` (research project with notebook), `moon-explore/` (project tracker) all sit under `~/Dev/science/` but aren't wikis. The spec should help disambiguate before the user (or agent) tries to wiki-ify the wrong artifact.

- [ ] 12.1 [medium] Add §"Adjacent patterns, not wikis": comparative-prose folders, research-output folders with notebooks, project trackers, single-document deep dives. For each, name the pattern, give a one-line "use X instead" pointer, and reference a concrete example from `~/Dev/science/`.
- [ ] 12.2 [easy] Add a one-question gate at the top of Mode A: "Is this prose knowledge accumulated across many sources?" — if no, redirect per the §Adjacent patterns guidance instead of scaffolding.

### Phase 13: personal vs publishable profiles

`longevity` is publish-quality (full front matter, wikilinks, scripted lint, LICENSES.md); `bpu`/`aliens` are personal-grade (no front matter, relative links, LLM lint). The spec treats all wikis identically. Add a `profile:` axis orthogonal to `variant:`.

- [ ] 13.1 [medium] Spec a `profile:` axis: `personal` (defaults: relative links, no front matter, LLM lint, no license tracking) vs `publishable` (defaults: wikilinks, full front matter, scripted lint, license tracking).
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
