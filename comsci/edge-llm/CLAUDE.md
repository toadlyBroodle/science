# Edge-LLM Knowledge Base

Compiles cutting-edge research on small/edge LLMs with one operational target: **run a useful, productive agentic coding harness on a laptop GPU with 4 GB VRAM, getting as close as possible to Claude Haiku 4.5 (or Sonnet 4.6) inside Claude Code**.

End-user is a single solo developer with a small budget. The wiki must answer:

1. What is the SOTA across small models, quantization, alt architectures, agentic training, runtimes, and harnesses?
2. Where can a solo dev with small compute plausibly contribute?
3. What is the best path forward?

## Structure

```
raw/                # Source papers, model cards, blog posts, READMEs (PDF + markdown). Never edit.
wiki/               # LLM-compiled knowledge base. All files here are LLM-authored.
  index.md          # Master catalog: every wiki page with one-line summary, by category.
  models/           # One page per model family (Qwen3, Phi-4, Gemma 3, DeepSeek-R1-Distill, LFM2, ...).
  techniques/       # Quantization, distillation, speculative decoding, KV-cache quant,
                    #   tool-call format conformance, prompt/prefix caching, offload.
  architectures/    # SSM/Mamba-2, Jamba, Zamba2, Hymba, MoE, Mixture-of-Depths, linear attention.
  runtimes/         # llama.cpp, ExLlamaV2, MLX, vLLM, TensorRT-LLM, KTransformers, Ollama, LM Studio.
  training/         # Agentic SFT (xLAM, Hammer, ToolACE), R1-style RL, distillation,
                    #   synthetic-data pipelines.
  harnesses/        # Claude Code, aider, Cline, Continue, OpenDevin, Roo, Goose.
  benchmarks/       # SWE-bench Verified (+ Lite), LiveCodeBench, BFCL v3, Aider polyglot,
                    #   ToolBench, Terminal-Bench, HumanEval+.
  analysis/         # Syntheses, gap maps, contribution roadmap, 4 GB budget math.
  entities/         # OPTIONAL. Spin up only when a name appears in 3+ pages.
log.md              # Append-only operation log (INIT, INGEST, QUERY, LINT entries).
```

## Wiki Article Format

Every article follows this template:

```markdown
# Article Title

> **Summary:** One-paragraph overview.

**Sources:** [[raw/source_filename.md]], ...

---

## Section Heading

Body with cross-references using relative markdown links:
[Qwen3-Coder](../models/qwen3-coder.md), [AWQ](../techniques/awq.md), etc.

## See Also

- [Related Article](../category/related-article.md)
```

## Naming Conventions

- Filenames: kebab-case, `.md` extension (e.g., `deepseek-r1-distill.md`, `kv-cache-quantization.md`).
- One topic per file. Prefer focused articles over mega-pages, except when consolidating prevents stub fan-out.
- Subdirectories: `models/`, `techniques/`, `architectures/`, `runtimes/`, `training/`, `harnesses/`, `benchmarks/`, `analysis/`, optional `entities/`.

## Workflows

### Ingest

1. Download paper / model card / blog post / README to `raw/<slug>.md` (and `.pdf` if applicable).
2. Read the converted markdown.
3. Extract: key claims, methods, benchmark numbers, model/quant/runtime constraints, dates.
4. Create or update wiki articles in the appropriate subdirectory.
5. Add cross-references between new and existing articles.
6. Update `wiki/index.md` with any new or changed pages.
7. Append to `log.md`: `YYYY-MM-DD INGEST: <one-line description>`.

### Query

1. Search relevant wiki pages first (grep + index.md). Do not re-derive from `raw/` if a wiki page already covers it.
2. Synthesize a response citing specific wiki articles.
3. If the analysis is novel or load-bearing, file it as an `analysis/` page.
4. Append to `log.md`: `YYYY-MM-DD QUERY: <question summary>`.

### Lint

1. Scan all wiki articles for:
   - Contradictions between articles (small-model benchmark numbers diverge often; flag and reconcile).
   - Missing cross-references (topics mentioned without a link).
   - Orphaned pages (no inbound links).
   - Stale claims (model versions, leaderboard positions, library APIs change quickly in this domain).
   - Gaps: topics referenced in `raw/` without a dedicated wiki page.
   - Broken relative links.
2. Produce a lint report.
3. Append to `log.md`: `YYYY-MM-DD LINT: <summary of findings>`.

## Domain-Specific Conventions

- **Always record numbers with their full context.** A "70% on SWE-bench" claim is meaningless without the variant (Verified vs Lite vs full), the model's quant, the runtime, the harness, and the date. Capture all five or note them as unknown.
- **Always record VRAM footprint with context length.** Weights-only VRAM is misleading; KV cache often dominates beyond 32k tokens. Where possible, cite measured (model, quant, ctx) → VRAM tuples.
- **Distinguish model from harness performance.** Many "small model fails at coding" claims confound model capability with tool-call format issues. Tag the failure mode.
- **Date everything.** This domain moves quarterly. Every wiki page should record the source's publication date and the date of last verification.

## Writing Style

- NEVER use em dashes. Use colons, semicolons, commas, or restructure.
- Concise, technically confident. No fluff or hedging.
- No "I believe", "perhaps", "it seems", "in order to", "utilize".
- Cross-reference liberally; isolated articles are less useful.
- Every claim must trace back to a file in `raw/`.
- Raw sources are immutable; all curation happens in `wiki/`.
- Prefer factual, evidence-weighted language; note where sources disagree.
- The wiki is a living document; expect quarterly rewrites of model and benchmark pages.

## Out of Scope

- Pretraining experiments.
- Novel architecture design from scratch.
- Multi-modal small models (vision/audio), unless they bear on coding-agent capability.
- BitNet 1.58b deep dive: one stub only until a competitive coder-tuned 1.58b checkpoint at 7B exists.
- Long-context (>64k) tricks: half-priority. Agentic coding needs 32-64k effective; the practical lever is KV-cache quant + sliding window, covered in `techniques/`.
