# Terminal-Bench (1.0 / 2.0)

> **Summary:** Stanford + Laude Institute (arXiv:2601.11868, January 2026, 84 co-authors led by Mike A. Merrill). Hard, realistic command-line tasks for agent evaluation. **89 carefully curated tasks**, each with unique environment, human reference solution, and verification tests. Frontier models score <65%. Released alongside Harbor (containerized agent-testing framework).

**Sources:** [raw/terminal-bench.md](../../raw/terminal-bench.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md)

---

## What's in the 89 tasks

Tasks span:
- Compiling code.
- Training models.
- Configuring servers.
- Playing games.
- Debugging systems.

Each task: unique env, expert solution, verification tests, several hours of manual + LLM-assisted validation.

## v1.0 vs v2.0

- **v1.0 (May 2025):** broader, easier set. Quickly became default agent benchmark.
- **v2.0 (Jan 2026):** narrowed to 89 hard tasks, better verification, process-level milestone rewards.

## Frontier-model scores

Below 65% on v2.0. The benchmark is designed to be unsaturated for current frontier systems, leaving headroom for years.

## Position vs SWE-Bench

[SWE-Bench](swe-bench.md) tests Python-repo issue resolution. Terminal-Bench tests cross-language CLI agent skills (compile chains, server config, infrastructure debugging). For the 4 GB-VRAM target running an agentic coder, **Terminal-Bench is the more representative benchmark of real workflow**; it captures the multi-tool, multi-step loops that small-model harness performance needs to support.

## Relevance

[Qwen3-Coder-Next](../models/qwen3-coder-next.md) (Feb 2026) cites both SWE-Bench and Terminal-Bench. Any future evaluation of small models for agentic coding should report both numbers.

## See Also

- [SWE-Bench](swe-bench.md)
- [LiveCodeBench](livecodebench.md)
- [Aider polyglot](aider-polyglot.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
