# SWE-bench and SWE-bench Verified

**Source:** arXiv:2310.06770 (Jimenez et al., ICLR 2024); SWE-bench Verified — collaboration with OpenAI, 2024.
**Fetched:** 2026-05-02 via WebSearch
**Web:** https://www.swebench.com/

## Summary

SWE-bench: 2,294 software-engineering problems from real GitHub issues + corresponding pull requests across 12 popular Python repositories. Given a codebase and an issue, the model generates a patch; pass = patch resolves the issue and the repo's tests pass.

SWE-bench Verified: human-filtered subset of 500 instances. Annotators verified clear problem descriptions, correct test patches, solvable tasks given available info.

## Variants

- SWE-bench (2,294 instances, full)
- SWE-bench Lite (300 instances, easier subset)
- SWE-bench Verified (500 instances, quality-filtered)
- SWE-bench Multimodal (visual UI issues)
- SWE-bench Pro (2026 hardening)

## Position

The closest proxy benchmark for "can the model actually do agentic coding." Cited as the central capability metric in essentially every 2025-2026 coding-model paper. Mini-SWE-agent (a minimal ReAct agent) is the standard scaffold for direct LM evaluation.

## Relevance to 4 GB VRAM target

Numbers without context are meaningless: the wiki must always record (model, quant, runtime, harness, SWE-bench variant). Per the [Aider polyglot leaderboard](aider-polyglot.md) and [BFCL](bfcl.md), 2026 frontier models hit ~89% on Aider polyglot but SWE-Bench Verified Pro remains <65% even for frontier; the 4 GB target is correspondingly lower.
