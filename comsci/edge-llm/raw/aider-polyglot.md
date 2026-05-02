# Aider Polyglot Benchmark

**Source:** https://aider.chat/docs/leaderboards/ ; github.com/Aider-AI/polyglot-benchmark
**Fetched:** 2026-05-02 via WebSearch

## Summary

225 challenging Exercism coding exercises across C++, Go, Java, JavaScript, Python, Rust. Two attempts per problem; on second attempt the model sees the failed test output. Tests both problem-solving and structured-edit-format adherence.

## 2026 leaders

- Claude Opus 4.5: 89.4% (Anthropic-reported).
- GPT-5 (high): 88.0%.
- DeepSeek V3.2-Exp: 74.2% at $1.30/run (22x cheaper than GPT-5).

## Position

Aider polyglot tests two skills jointly: (1) generate correct code, (2) emit edits in a structured format the harness can apply. (2) is exactly the small-model failure mode of interest: a small model can produce correct code while emitting malformed edit blocks the harness rejects.

## Relevance to 4 GB target

For 4 GB-class models the structured-edit metric is more telling than the raw-correctness metric. A small model fine-tuned on Aider's expected edit format (the [agentic SFT recipe](../wiki/training/) thread) likely jumps in performance more than from a larger base.
