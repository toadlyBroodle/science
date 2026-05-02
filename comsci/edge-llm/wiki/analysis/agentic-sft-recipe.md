# Agentic SFT Recipe (Concrete Plan for Path 1)

> **Summary:** Step-by-step recipe for replaying Claude Code (or aider+Claude) sessions on real OSS issues, distilling the traces into a 4 GB-class open-weight base via Unsloth + LoRA. Target: a model that emits Cline / aider format with ~99% conformance on first call. Direct application of [Path 1 from the contribution roadmap](contribution-roadmap.md).

**Sources:** [training/slm-agentic-tool-calling.md](../training/slm-agentic-tool-calling.md), [training/xlam-2.md](../training/xlam-2.md), [training/toolace.md](../training/toolace.md), [harnesses/cline-continue-goose.md](../harnesses/cline-continue-goose.md), [harnesses/aider.md](../harnesses/aider.md).

---

## Pipeline overview

```
1. Source OSS issues          → 1k-5k issues from active repos
2. Run Claude Sonnet 4.6      → solve each via Claude Code or aider
3. Filter for successful runs → only sessions where tests pass
4. Re-format to target harness → Cline XML / aider SEARCH-REPLACE / Goose MCP
5. SFT via Unsloth + LoRA      → on a 1-3B base
6. Evaluate                    → BFCL v3 / Aider polyglot / SWE-Bench Lite
7. Iterate                     → focus on format errors first, then quality
```

## Step 1: Source issues

Target active OSS repos with rich issue trackers. Suggested sources:
- SWE-Bench instances (already curated, with reference solutions and tests).
- OSS issues tagged `good-first-issue` / `help-wanted` on actively-maintained Python and JS repos.
- Aider polyglot exercises (already in 6 languages with tests).

Aim for 1,000-5,000 issues. Diverse languages, diverse repo sizes, diverse difficulty.

## Step 2: Run frontier model

Use Claude Code (or aider with `--model claude-sonnet-4-6`) to solve each issue end-to-end. Capture the full trace: prompt, every tool call, every tool result, final outcome.

**Cost estimate:** ~ $0.05-0.15 per session × 5,000 sessions = $250-750. Expensive but cap-able.

**Mitigation:** Replay on cheaper models for half the corpus (Haiku 4.5, GPT-4o-mini), keep frontier-quality only for hard cases.

## Step 3: Filter

Pass criterion: the final patch makes the issue's tests pass. Only retain successful sessions.

Expect a 30-60% pass rate at the frontier; you keep ~ 1,500-3,000 sessions.

## Step 4: Re-format

The captured trace is in Claude's tool-call JSON format. Convert to the target harness's format. Three target formats supported:

- **Cline XML:** `<tool>...</tool>` tags around tool calls.
- **aider SEARCH/REPLACE:** Markdown-fenced code blocks with `<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`.
- **Goose MCP:** JSON-RPC over stdio (closest to Claude's native).

Each session expands into a multi-turn dialog. Roles: system, user, assistant, tool. Same shape as standard chat SFT data, just with the harness-format constraint enforced.

## Step 5: SFT via Unsloth + LoRA

Recommended setup:

- **Base model:** Qwen3-Coder-3B-Instruct (when released) or Phi-4-mini-instruct (current default).
- **Framework:** Unsloth (fastest small-model SFT pipeline, 4-bit quant during training).
- **Method:** LoRA (rank 32-64, alpha 32-64).
- **Hyperparameters:** lr=2e-4 (Unsloth default), batch=8 with gradient accumulation, 1 epoch (per the [350M paper](../training/slm-agentic-tool-calling.md)).

**Cost estimate:** 1 H100 hour = $2-3. A 3B-parameter LoRA on 3,000 sessions × ~ 8K tokens average = 24M tokens; fits in 1-2 H100 hours.

Total: < $10 for the SFT step. Cheap.

## Step 6: Evaluate

Required scorecard:
- **[BFCL v3](../benchmarks/bfcl.md)** multi-turn; diagnostic for tool-call correctness.
- **[Aider polyglot](../benchmarks/aider-polyglot.md)** edit-correctness column; diagnostic for format conformance.
- **[SWE-Bench Lite](../benchmarks/swe-bench.md)**; diagnostic for end-to-end agentic capability.
- **[LiveCodeBench](../benchmarks/livecodebench.md)** time-segmented; diagnostic for raw coding (sanity check that SFT didn't break the base).

Compare four data points:
1. Original base (e.g., Phi-4-mini-instruct).
2. SFT'd model FP16.
3. SFT'd model Q4_K_M.
4. SFT'd model Q4_K_M + EAGLE-3 draft head.

The interesting metric is the gap between (2) and (3); that's the [SLMQuant](../techniques/slmquant.md) penalty in practice on agentic metrics. Path 3 (quant-aware fine-tune) tries to close that gap.

## Step 7: Iterate

First-iteration failures cluster predictably:
- **Format errors (most common):** Add more examples of the failure-mode-shaped tool calls.
- **Tool-result misinterpretation:** Add traces showing successful recovery from each tool-result type.
- **Multi-turn coherence loss:** Increase the proportion of long (8+ turn) successful sessions.
- **Specific language gaps:** Up-sample under-represented languages.

Each iteration costs < $50 in re-training compute.

## Realistic outcome estimate

Following this recipe on Phi-4-mini-instruct base:

- Pre-SFT BFCL v3: 0.4-0.5 (typical for general-purpose 3-4B models).
- Post-SFT BFCL v3: 0.65-0.75 (matching the published [ToolACE 8B](../training/toolace.md) tier).
- Post-SFT Aider polyglot edit column: 95-99% format conformance (vs ~ 60-80% for non-fine-tuned).
- Post-SFT SWE-Bench Lite: 8-15% (vs Claude Sonnet 4.6's 60%+, but useful for narrow tasks).

Not Claude Sonnet 4.6. Definitely useful at $0/turn.

## Deliverables

For a solo-dev publication:

1. Hugging Face dataset card (the SFT data).
2. Hugging Face model card (the LoRA adapter and merged weights).
3. arXiv preprint or blog post documenting the recipe and results.
4. GitHub repo with reproducible training and eval scripts.

## See Also

- [Contribution roadmap](contribution-roadmap.md)
- [Harness eval suite design](harness-eval-suite-design.md)
- [Quant-vs-capability frontier](quant-vs-capability-frontier.md)
- [SLMQuant](../techniques/slmquant.md)
