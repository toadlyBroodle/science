# Aider Polyglot Benchmark

> **Summary:** 225 challenging Exercism coding exercises across C++, Go, Java, JavaScript, Python, Rust. Two attempts per problem; on second attempt the model sees the failed test output. Tests both problem-solving and structured-edit-format adherence. **2026 leaders: Claude Opus 4.5 at 89.4%, GPT-5 at 88.0%, DeepSeek V3.2-Exp at 74.2% ($1.30/run, 22x cheaper than GPT-5).**

**Sources:** [raw/aider-polyglot.md](../../raw/aider-polyglot.md)

---

## What it measures

Aider polyglot tests two skills jointly:

1. **Generate correct code.** Solve the problem.
2. **Emit edits in a structured format the harness can apply.** Aider expects a specific edit-block format; mistakes here block the apply step regardless of code correctness.

Skill #2 is exactly the small-model failure mode of interest. A small model can generate correct code in its head while consistently emitting malformed edit blocks the harness rejects. Aider polyglot's edit-correctness column surfaces this.

## Two-attempt protocol

Models get two tries. On the second attempt, the failing test's output is fed back. This rewards models that can self-correct from a clear error signal; a directly relevant skill for agentic coding loops.

## 2026 leaders

| Model | Score | Notes |
|---|---|---|
| Claude Opus 4.5 | 89.4% | Anthropic-reported |
| GPT-5 (high) | 88.0% | |
| DeepSeek V3.2-Exp | 74.2% | $1.30/run, 22x cheaper than GPT-5 |

## Relevance to 4 GB VRAM target

For 4 GB-class models, the structured-edit metric is more telling than raw correctness. A small model fine-tuned to emit Aider's exact edit format (an [agentic SFT recipe](../training/) that includes harness-format conformance) likely jumps in performance more than from a larger base.

This is one of the clearest pieces of evidence for the wiki's [contribution roadmap](../analysis/contribution-roadmap.md) thesis: harness-format conformance is solo-dev-tractable and high-leverage.

## See Also

- [SWE-Bench](swe-bench.md)
- [BFCL](bfcl.md)
- [Aider harness](../harnesses/aider.md)
- [Claude Code harness](../harnesses/claude-code.md)
