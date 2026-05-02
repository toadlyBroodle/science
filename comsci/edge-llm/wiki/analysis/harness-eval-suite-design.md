# 4 GB-Envelope Eval Suite (Concrete Plan for Path 2)

> **Summary:** Design for a Docker-packaged, reproducible (model × quant × runtime × harness × benchmark) evaluation matrix at the 4 GB VRAM ceiling. The infrastructure piece that makes every other contribution path measurable. Target: 15 (model × quant) tuples × 5 benchmarks = 75 cells, on llama.cpp + KTransformers + vLLM.

**Sources:** [analysis/runtime-comparison.md](runtime-comparison.md), [analysis/harness-comparison.md](harness-comparison.md), [analysis/four-gb-budget-math.md](four-gb-budget-math.md), benchmark pages.

---

## What's missing in the world today

Many leaderboards exist (Artificial Analysis, llm-stats.com, Aider's own page, BFCL's gorilla-llm leaderboard). None of them:
- Constrain to the 4 GB VRAM envelope.
- Run consistent quant flavors across models.
- Run consistent runtimes (so the runtime variance is controlled).
- Run agentic-specific harness wrappers.
- Publish reproducible Docker recipes.

The result: nobody can answer "what should I run on my 4 GB laptop?" with cited numbers.

## Architecture

```
edge-llm-eval/
├── docker-compose.yml
├── runner/
│   ├── llama-cpp/         (Dockerfile + entrypoint)
│   ├── ktransformers/
│   ├── vllm/
│   └── exllamav2/
├── harness/
│   ├── aider-adapter/      (model + benchmark glue)
│   ├── cline-adapter/
│   └── direct-api/         (no harness; reference)
├── benchmark/
│   ├── bfcl-v3/            (driver + cases)
│   ├── aider-polyglot/
│   ├── swe-bench-lite/
│   ├── livecodebench/
│   └── terminal-bench/
├── matrix.yaml             (the experiment spec)
├── results/
│   ├── runs/
│   │   └── <date>-<model>-<quant>-<runtime>-<harness>/
│   └── leaderboard.json   (consolidated)
└── analyze/                (notebooks for charts and discussion)
```

## The matrix.yaml spec

```yaml
models:
  - id: phi-4-mini-instruct
    quants: [fp16, q5_k_m, q4_k_m, q3_k_m]
  - id: gemma-3-4b-it
    quants: [fp16, q4_k_m, q3_k_m]
  - id: qwen3-coder-3b
    quants: [fp16, q4_k_m, q3_k_m]
  - id: lfm2-2.6b
    quants: [fp16, q4_k_m]
  - id: zamba2-1.2b
    quants: [fp16, q4_k_m]
  - id: ds-r1-distill-qwen-1.5b
    quants: [fp16, q4_k_m]
  - id: xlam-2-3b-r
    quants: [fp16, q4_k_m]

runtimes:
  - llama-cpp
  - vllm
  - ktransformers     # MoE only
  - exllamav2          # NVIDIA only

harnesses:
  - aider
  - cline
  - direct-api

benchmarks:
  - bfcl-v3
  - aider-polyglot
  - swe-bench-lite
  - livecodebench-segment-2025-10-onward
  - terminal-bench-1.0  # 2.0 too expensive at first

vram_budget_gb: 4.0

speculative_decoding:
  - none
  - eagle-3-generic
  - eagle-3-coder-matched   # if Path 4 ships
```

## Output: standardized scorecard

Per (model, quant, runtime, harness) tuple:

```json
{
  "config": {
    "model": "phi-4-mini-instruct",
    "quant": "q4_k_m",
    "runtime": "llama-cpp@b3500",
    "harness": "aider@v0.65",
    "spec_decoding": "eagle-3-generic",
    "context_length": 32768,
    "vram_used_gb": 3.1
  },
  "benchmarks": {
    "bfcl-v3": {"overall": 0.62, "multi-turn": 0.51},
    "aider-polyglot": {"pass-1": 0.32, "pass-2": 0.41, "edit-correctness": 0.97},
    "swe-bench-lite": {"resolved": 0.085},
    "livecodebench-2025-10-onward": {"pass-1": 0.28},
    "terminal-bench-1.0": {"resolved": 0.18}
  },
  "throughput": {
    "tokens-per-sec": 47,
    "time-to-first-token-ms": 320
  },
  "date": "2026-05-15"
}
```

These scorecards are the citation unit.

## Phasing

**v1 (weeks 1-4):** llama.cpp runtime + aider harness + 5 models × 3 quants × 3 benchmarks = 45 cells. Establish reproducibility.

**v2 (weeks 5-8):** Add vLLM + Cline + 2 more benchmarks. Scale to ~ 100 cells.

**v3 (weeks 9-12):** Add KTransformers + MoE coverage (Qwen3-Coder-Next via offload) + Terminal-Bench. ~ 150 cells.

**Maintenance:** CI runs on new model releases; adds rows to the leaderboard automatically.

## Cost estimate

Per cell:
- BFCL v3: ~ $0.50 cloud compute (1k cases × ~ 2k tokens × small model speed).
- Aider polyglot: ~ $1-2 (225 problems × 2 attempts × 5-10k tokens).
- SWE-Bench Lite: ~ $5-15 (300 instances, longer agentic loops).
- LiveCodeBench: ~ $1-3.
- Terminal-Bench: ~ $5-10 (89 hard tasks, multi-step).

Total per (model × quant × runtime × harness): ~ $15-35.

For v1 (45 cells): $700-1,500.
For full matrix at v3 (150+ cells): $2,500-5,000 plus ongoing.

This is the dominant cost. Strategies to reduce:
- Run on a personal 4 GB laptop GPU (free!) for benchmarks that fit in time-budget.
- Cache benchmark inputs aggressively.
- Skip cells provably not viable (e.g., 7B at FP16 obviously over budget).

## Why this is the right shape of contribution

- **Reproducible.** Docker recipes, pinned versions, fixed seeds.
- **Citable.** Each scorecard is a permanent artifact.
- **Multiplier.** Every other contribution path needs this to demonstrate its win.
- **Ongoing relevance.** Domain moves quarterly; the suite re-runs on new releases.

## Caveats / things this won't do

- Won't measure subjective code quality or readability.
- Won't catch security flaws in generated code.
- Won't test conversational UX.
- Won't evaluate on private-corpus tasks.

These are explicit non-goals. The suite is for the agentic-coding-capability dimension only.

## See Also

- [Contribution roadmap](contribution-roadmap.md)
- [Agentic SFT recipe](agentic-sft-recipe.md)
- [Missing evals](missing-evals.md)
