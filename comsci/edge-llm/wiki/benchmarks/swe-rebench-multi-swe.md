# SWE-rebench, Multi-SWE-bench, SWE-Universe

> **Summary:** Three 2025-2026 benchmarks that supersede or extend [SWE-Bench Verified](swe-bench.md). **Multi-SWE-bench** (April 2025): 1,632 issues across 7 non-Python languages. **SWE-rebench** (May 2025; v2 February 2026): 32K decontaminated tasks across 20 languages. **SWE-Universe** (February 2026): scaling to millions of verifiable training instances. For honest 4 GB-class numbers, prefer SWE-rebench over SWE-Bench Verified; contamination inflates small-model scores more than large ones.

**Sources:** [raw/swe-bench-evolution-2026.md](../../raw/swe-bench-evolution-2026.md)

---

## Why this page exists

SWE-Bench Verified has three problems that compound at the 4 GB-class:

1. **Python-only.** Small models trained on Python-heavy corpora overfit to Python idioms; their actual generalization is masked.
2. **Static.** The 500 instances are public since late 2024; any small model trained after that may be (deliberately or incidentally) contaminated.
3. **Limited training scale.** Useful for evaluation, useless for the kind of large-scale RL training described in [agentic-rl-coding.md](../training/agentic-rl-coding.md).

The 2025-2026 work fills these gaps.

## Multi-SWE-bench

**arXiv:2504.02605**, April 2025. 1,632 issues across **Java, TypeScript, JavaScript, Go, Rust, C, C++**. Each language stresses different model capabilities:

| Language | Stresses |
|---|---|
| Java | Verbosity, type-system precision |
| TypeScript | Type inference under structural typing |
| JavaScript | Loose typing, runtime semantics |
| Go | Error handling idioms, concurrency |
| Rust | Lifetime/borrow reasoning |
| C / C++ | Memory management, undefined behavior |

Small models trained primarily on Python often degrade sharply on Rust and C. Multi-SWE-bench exposes this.

## SWE-rebench

**v1: arXiv:2505.20411** (May 2025). **v2: arXiv:2602.23866** (February 2026). Leaderboard: https://swe-rebench.com/.

| Version | Tasks | Languages | Years |
|---|---|---|---|
| v1 | 21K+ | Python | various |
| v2 | 32K | 20 languages | 2014-2025 |

**Method**: automated continuous extraction of fresh GitHub tasks. Decontaminated by construction, because new tasks land monthly.

**Why it matters at 4 GB**: contamination lifts small-model SWE-Bench-Verified scores disproportionately. A 3B model that scores 25% on Verified might score 12% on SWE-rebench v2. The latter is the honest number.

### Leaderboard scores are not comparable across models by default

The property that makes SWE-rebench honest (fresh tasks monthly) is also a trap. Older models are never re-run on newer batches, so each model's score is scoped to whatever batch existed when it was submitted; the site exposes this as a per-model `taskRangeTimestamp`. Its range buckets **clamp** to the intersection of the requested range and the tasks that model actually ran, so a wide range returns a number that looks global but is not.

Batch difficulty varies a lot. Across the 10 models evaluated on both the Feb 2026 and Mar-May 2026 batches, the later batch scored **9.4 points lower on average**, and weaker open models fell hardest:

| Model | Feb 2026 | Mar-May 2026 | Delta |
|---|---|---|---|
| Claude Opus 4.6-high | 65.3 | 47.8 | -17.4 |
| GLM-4.7 | 58.7 | 38.2 | -20.5 |
| Gemma 4 31B | 41.6 | 16.5 | -25.0 |
| Claude Code (harness) | 58.4 | 59.6 | +1.2 |

Worked example: **Qwen3.5-27B scores 58.9 and Qwen3.6-27B scores 36.5**, which reads as a 22-point regression. They share no tasks (Feb 2026, n=57 vs Mar-May 2026, n=55). Applying GLM-4.7's batch offset puts 3.6-27B near 57, level with its predecessor; on [SWE-Bench Verified](swe-bench.md) the ordering flips (77.2 vs 72.4). Two further cautions: at n≈56 the per-model 95% CI is roughly ±13 points, and Qwen3.6-27B used 28% fewer tokens per instance (1.88M vs 2.62M), which points at early agent-loop termination, a harness failure mode, not weaker reasoning.

**Rule**: only compare models that report a score on the *same* task window. The [dashboard](../../dashboard/index.html) enforces this with a window selector.

## SWE-Universe

**arXiv:2602.02361**, February 2026. Scaling verifiable SWE training environments to millions of instances. This is the training-data-scale companion to SWE-rebench's eval-quality story.

For solo-dev RL training pipelines (Path 1 in [contribution-roadmap.md](../analysis/contribution-roadmap.md)), SWE-Universe is the realistic source of millions of trajectories without per-instance manual curation.

## SWE-Bench Pro (mentioned for completeness)

Scale AI public dataset variant. Harder tasks, exposes the gap small models have to close. Public leaderboard at https://labs.scale.com/leaderboard/swe_bench_pro_public.

## What to use when

| Use case | Benchmark |
|---|---|
| Short reproducible eval, well-known numbers | SWE-Bench Verified (with contamination caveat) |
| Honest 4 GB-class generalization number | **SWE-rebench v2** |
| Multilingual coverage check | **Multi-SWE-bench** |
| RL-training task source | **SWE-Universe** + SWE-rebench |
| Hardest current eval | SWE-Bench Pro |

## See Also

- [SWE-Bench Verified](swe-bench.md)
- [Aider Polyglot](aider-polyglot.md)
- [LiveCodeBench](livecodebench.md)
- [LongCodeBench](longcodebench.md)
- [Agentic RL Coding](../training/agentic-rl-coding.md)
