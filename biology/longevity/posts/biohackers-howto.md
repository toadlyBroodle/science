# Contributing to the wiki (the AI runs the pipeline)

Open `claude` (or cursor, codex, whatever) in a directory you don't mind it touching, then prompt:

**Clone:**
> Clone https://github.com/toadlyBroodle/science.git here. Then read biology/longevity/CLAUDE.md so you know how the wiki works.

**Add a paper:**
> Ingest this paper into the longevity wiki: <PMC or DOI URL>. Follow biology/longevity/CLAUDE.md.

**Add several papers / topic sweep:**
> Research and ingest the strongest recent papers on <topic, e.g. "GLP-1 agonists and biological aging"> into the longevity wiki. Follow biology/longevity/CLAUDE.md.

**Update the analysis after new ingests:**
> Update biology/longevity/wiki/analysis/promising-reverse-aging.md to integrate the new papers. Lint clean.

**Update the reader-facing recommendations doc:**
> Update biology/longevity/recommendations.md to cite any new supporting evidence.

The AI handles sources.json, licenses, download/convert, paper pages, topic links, lint, log.

**PR it:**

> Commit (no AI attribution per biology/longevity/CLAUDE.md) and prep a PR branch.

Then push and open the PR on GitHub yourself (most agent setups don't have push scope).
