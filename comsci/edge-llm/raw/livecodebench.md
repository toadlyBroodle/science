# LiveCodeBench

**Source:** arXiv:2403.07974; livecodebench.github.io
**Fetched:** 2026-05-02 via WebSearch

## Summary

Holistic, contamination-free coding benchmark. Continuously collects new problems from LeetCode, AtCoder, CodeForces. Problems annotated with release dates, allowing time-segmented evaluation: only score a model on problems released *after* its training cutoff.

## Scope

- 600+ coding problems (May 2023 - Aug 2024 at the time of paper; ongoing collection).
- Tests: code generation, self-repair, code execution, test-output prediction.
- 50+ LLMs evaluated.

## Why it matters

Most coding benchmarks (HumanEval, MBPP, even early SWE-Bench) leak into training corpora. Time-segmented LiveCodeBench is the cleanest "is this model actually generalizing?" measurement. Successful at detecting contamination across GPT-4o, Claude, DeepSeek, Codestral.

## Position

For the 4 GB-VRAM target, LiveCodeBench provides:
- A fair comparison surface across the rapidly-changing small-model landscape.
- Baselines on subskills (self-repair, execution) that interact directly with agentic harness loops.
