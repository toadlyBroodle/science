# LiveCodeBench

> **Summary:** Jain et al. (arXiv:2403.07974). Holistic, contamination-free coding benchmark. Continuously collects new problems from LeetCode, AtCoder, CodeForces. Problems annotated with release dates: only score a model on problems released *after* its training cutoff. Tests code generation, self-repair, code execution, test-output prediction. **The cleanest "is this model actually generalizing?" measurement.**

**Sources:** [raw/livecodebench.md](../../raw/livecodebench.md)

---

## The contamination problem

Most coding benchmarks (HumanEval, MBPP, even early SWE-Bench) leak into training corpora. A model can memorize solutions during pretraining without learning the underlying skill. Time-segmented LiveCodeBench evaluation closes this gap: filter the problem set to those released *after* the model's training-data cutoff and you have a clean generalization measurement.

The paper validates the methodology by detecting contamination across GPT-4o, Claude, DeepSeek, Codestral.

## Subskills measured

- Code generation
- Self-repair (model fixes its own buggy code given test failures)
- Code execution (predict program output)
- Test-output prediction

The latter three are directly relevant for agentic coding loops where the model must reason about runtime behavior, not just emit text.

## Scope

- 600+ problems (May 2023 - Aug 2024 at paper time; ongoing collection).
- 50+ LLMs evaluated.

## Relevance to 4 GB VRAM target

For the rapidly-changing small-model landscape, LiveCodeBench provides:
- A fair comparison surface that doesn't reward memorization.
- Subskill baselines (self-repair, execution) that interact directly with agentic harness loops.

A small model with low LiveCodeBench raw-generation but high self-repair could still be a strong agentic-harness candidate; the harness provides the tool-execution feedback the model needs.

## See Also

- [SWE-Bench](swe-bench.md)
- [Aider polyglot](aider-polyglot.md)
- [BFCL](bfcl.md)
