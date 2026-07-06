# SWE-Chain

> **Summary:** May 2026 benchmark (arXiv:2605.14415, Lam et al.) testing coding agents on chained release-level package upgrades: 12 chains across 9 Python packages, 155 version transitions, 1,660 grounded requirements, each transition building on the agent's previous edits. Industry average 44.8% resolving; best result Claude-Opus-4.7 + Claude Code at 60.8%. Errors compound across a chain, which measures exactly the long-horizon state maintenance that single-issue [SWE-bench](swe-bench.md) misses.

**Sources:** [raw/swe-chain.md](../../raw/swe-chain.md)

---

## Design

Upgrade specifications are synthesized by aligning release notes with code diffs (divide-and-conquer pipeline), so requirements are grounded in what the release actually changed. The agent inherits its own modified codebase at every transition; there is no reset between tasks. Metrics: resolving rate, precision, F1, reported under a Build+Fix regime.

## Results (paper-reported, May 2026, frontier models)

| System | Resolving | Precision | F1 |
|---|---|---|---|
| Claude-Opus-4.7 + Claude Code | 60.8% | 80.6% | 68.5% |
| Industry average | 44.8% | 65.4% | 50.2% |

No small-model results published. Model/quant/runtime context per wiki convention: frontier API models, unquantized, Claude Code and comparable harnesses.

## Why it matters for the 4 GB target

Chained upgrades stress the two things small models are weakest at: carrying task state across long horizons and not regressing earlier work. Expect the frontier-to-4B gap here to be wider than on single-issue SWE-bench; a small-model SWE-Chain run is an open, cheap, publishable eval ([missing evals](../analysis/missing-evals.md)). Compaction and filesystem-state strategies from [long context via filesystem](../analysis/long-context-via-filesystem.md) are the plausible mitigations.

Adjacent June 2026 work (raw notes only, no wiki pages yet): SWE-Explore (arXiv:2606.07297) on repository exploration behavior, and the position paper "Coding Benchmarks Are Misaligned with Agentic Software Engineering" (arXiv:2606.17799), which argues the field's benchmark suite under-measures maintenance-style work; SWE-Chain is the concrete instance of that critique.

## See Also

- [SWE-Bench](swe-bench.md)
- [SWE-rebench, Multi-SWE-bench](swe-rebench-multi-swe.md)
- [Terminal-Bench](terminal-bench.md)
- [LongCodeBench](longcodebench.md)
- [Long context via filesystem](../analysis/long-context-via-filesystem.md)
