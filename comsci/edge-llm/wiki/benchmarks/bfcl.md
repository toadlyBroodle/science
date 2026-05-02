# Berkeley Function Calling Leaderboard (BFCL)

> **Summary:** Patil et al., ICML 2025. Standardized benchmark for LLM function-calling capability. v3 (2024) introduced multi-turn / multi-step interactions; v4 (2025) added holistic agentic evaluation. 1,000 test cases across vehicle control, trading, travel, file system. State-based evaluation: verifies system state changes AND execution-path correctness. **GLM-4.5 leads BFCL v3 at 0.778; average across 18 evaluated models is 0.699.**

**Sources:** [raw/bfcl.md](../../raw/bfcl.md), [raw/toolace.md](../../raw/toolace.md), [raw/slm-agentic-tool-calling.md](../../raw/slm-agentic-tool-calling.md)

---

## Why BFCL is the most diagnostic small-model benchmark

For 4 GB-class models, the dominant failure mode is *tool-call format conformance*, not raw IQ. A small model can produce semantically correct intentions but emit malformed JSON / XML / function-call syntax that the harness rejects. BFCL is designed to catch exactly this.

[SWE-Bench](swe-bench.md) measures whether the model can fix code; BFCL measures whether the model can call tools correctly. The wiki's contribution roadmap argues BFCL is the more sensitive metric in the 1-4B parameter range.

## v3 specifics

- 1,000 test cases.
- Domains: vehicle control, trading bots, travel booking, file system.
- State-based evaluation (correctness of side effects).
- Multi-turn: model must retain state across turns and chain multiple internal function calls.

## v4

Adds holistic agentic evaluation extending v3's tool-call focus to include planning, error recovery, and longer-horizon tasks.

## Current leaderboard observations (2025-2026)

- **GLM-4.5 (Zhipu AI):** 0.778; leads.
- 18 models evaluated, mean 0.699.
- 8B models trained via [ToolACE](../training/toolace.md) reach near-SOTA.

## Relevance for the SLM contribution path

The [350M tool-calling paper](../training/slm-agentic-tool-calling.md) showed that targeted SFT on a 350M base beats much larger generic models on ToolBench (BFCL's predecessor). Replicating this at 1-4B with BFCL v4 as the evaluation target is the most direct solo-dev contribution: a published (model card + dataset) demonstrating that the same recipe scales.

## See Also

- [SWE-Bench](swe-bench.md)
- [Aider polyglot](aider-polyglot.md)
- [ToolACE](../training/toolace.md)
- [SLM agentic tool calling](../training/slm-agentic-tool-calling.md)
