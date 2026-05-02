# SWE-Bench (Original / Lite / Verified / Multimodal / Pro)

> **Summary:** Jimenez et al., ICLR 2024 (arXiv:2310.06770). 2,294 software-engineering problems from real GitHub issues + corresponding pull requests across 12 popular Python repos. Given a codebase and an issue, the model must generate a patch; pass = patch resolves the issue and the repo's tests pass. **The closest proxy benchmark for agentic coding capability.** SWE-Bench Verified (500 human-filtered instances) is the practical comparator.

**Sources:** [raw/swe-bench.md](../../raw/swe-bench.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md), [raw/terminal-bench.md](../../raw/terminal-bench.md)

---

## Variants

| Variant | Size | Notes |
|---|---|---|
| SWE-Bench (full) | 2,294 | Original, 12 Python repos |
| SWE-Bench Lite | 300 | Easier subset, faster eval |
| SWE-Bench Verified | 500 | Human-filtered (OpenAI collab); **the practical comparator** |
| SWE-Bench Multimodal | varies | UI / visual issues |
| SWE-Bench Pro | hardened | 2026, harder than Verified |

## Evaluation protocols

- **Direct LM:** mini-SWE-agent (minimal ReAct loop, no scaffolding); probes raw model capability.
- **With scaffolding:** SWE-Agent or arbitrary harness (Cursor, Claude Code, Codex); probes model+harness joint capability.

These two numbers are not comparable to each other. Always record which protocol produced a number.

## Relevance to 4 GB VRAM target

SWE-Bench numbers are *only* meaningful with full context: **(model, quant, runtime, harness, SWE-Bench variant, scaffolding mode, evaluation date)**. A "70% on SWE-Bench" claim missing any of those tags is not citable.

For the 4 GB target, the practical question is: which (model × quant × harness) tuple maximizes SWE-Bench Verified (or Verified Pro) under the 4 GB budget? This is the central question of [`analysis/four-gb-budget-math.md`](../analysis/four-gb-budget-math.md) and [`analysis/harness-comparison.md`](../analysis/harness-comparison.md) (both pending Phase 3).

## See Also

- [Aider polyglot](aider-polyglot.md)
- [Terminal-Bench](terminal-bench.md)
- [LiveCodeBench](livecodebench.md)
- [BFCL](bfcl.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (claims SWE-Bench competitiveness at 3B-active)
