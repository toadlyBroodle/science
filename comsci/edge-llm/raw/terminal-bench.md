# Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces

**Source:** arXiv:2601.11868 (https://arxiv.org/abs/2601.11868)
**Fetched:** 2026-05-02 via WebFetch
**Lead author:** Mike A. Merrill, 84 co-authors (Stanford + Laude Institute)
**Submitted:** 2026-01-17
**Web:** https://www.tbench.ai/

## Abstract / extracted content

Terminal-Bench 2.0: 89 carefully curated tasks in terminal environments, each with unique setup, human-written reference solution, and comprehensive verification tests. Targets the gap between benchmarks that lack real-world relevance and those without enough difficulty for advanced models. Frontier-model performance is below 65%.

## Key facts

- 89 tasks (down from larger v1; intentionally hard subset).
- Each task: unique env, human reference solution, verification tests.
- Tasks include compiling code, training models, configuring servers, playing games, debugging.
- Released alongside Harbor (containerized agent-testing framework).
- Frontier models score <65%.

## Position

Terminal-Bench 1.0 (May 2025) became a default agent benchmark; v2.0 (Jan 2026) raises difficulty. Cited by [Qwen3-Coder-Next](../wiki/models/qwen3-coder-next.md) and other 2026 coding-agent papers.

## Relevance

For the 4 GB-VRAM target, Terminal-Bench is the realistic external eval. SWE-Bench is closely-coupled to Python; Terminal-Bench tests genuine multi-language CLI agent skills.
