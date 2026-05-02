# Berkeley Function Calling Leaderboard (BFCL v3 / v4)

**Source:** ICML 2025 (Patil et al.); leaderboard at gorilla.cs.berkeley.edu/leaderboard.html
**Fetched:** 2026-05-02 via WebSearch

## Summary

Standardized benchmark for LLM function-calling capability. v3 introduced multi-turn / multi-step interactions: extended conversational exchanges where the model retains contextual information across turns and executes multiple internal function calls. v4 adds holistic agentic evaluation.

## v3 details

- 1,000 test cases.
- Domains: vehicle control, trading bots, travel booking, file system management.
- State-based evaluation: verifies system-state changes AND execution-path correctness.

## Current v3 leaders (2025-2026)

- GLM-4.5 (Zhipu AI): 0.778 (top).
- Average across 18 evaluated models: 0.699.
- 8B models trained via [ToolACE](toolace.md) reach SOTA-rivalling scores.

## Position

BFCL is the function-calling-specific complement to SWE-Bench. SWE-Bench measures whether the model can *fix code*; BFCL measures whether the model can *call tools correctly*. For the 4 GB-VRAM target, BFCL is arguably the more diagnostic metric: tool-call format failures are the dominant small-model agentic failure mode, not raw IQ.
