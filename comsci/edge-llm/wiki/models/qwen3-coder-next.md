# Qwen3-Coder-Next

> **Summary:** 80B-total / 3B-active MoE specialized for coding agents (Qwen Team, Feb 2026). Trained via large-scale synthesis of verifiable coding tasks paired with executable environments, with mid-training and RL on environment feedback. The current canonical reference for "agentic-coder MoE" at small active-parameter footprint.

**Sources:** [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md), [raw/gemma-phi-qwen-tradeoffs.md](../../raw/gemma-phi-qwen-tradeoffs.md)

---

## Architecture

- **Total parameters:** 80 B (MoE).
- **Active per token:** 3 B.
- **Position:** Open-weight (base + instruct), released Feb 2026 on arXiv:2603.00729.

The active-3B footprint puts forward-pass compute in the 3B-dense range while drawing on 80B of stored expertise per layer. Routing details and expert count not extracted from the abstract; see paper Section X for the architecture diagram.

## Training

- Large-scale synthesis of *verifiable* coding tasks (compiler / test runner provides ground truth) paired with executable environments.
- Mid-training stage and RL stage both consume environment-feedback signal directly, rather than imitating teacher traces.
- Result: the model learns to act in coding environments, not just to predict next tokens.

## Benchmarks

The paper references SWE-Bench and Terminal-Bench. Specific numbers were not extracted from the arXiv abstract page; see the full report for tables. Independent comparison ([Gemma 4 / Phi-4 / Qwen3 tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md)) included Qwen3-30B-A3B (the smaller MoE sibling) but not Qwen3-Coder-Next directly.

## Relevance to 4 GB VRAM target

80B at FP16 = ~160 GB. At Q4 = ~40 GB. Pure-GPU inference on 4 GB VRAM is infeasible. The realistic path is hybrid offload using [DALI](../runtimes/dali-moe.md) (workload-aware CPU+GPU expert assignment) or [FlashMoE](../runtimes/flashmoe.md) (SSD-tier offload for the cold-expert tail). Active-expert footprint at 3 B fits comfortably in 4 GB; the question is keeping the *right* 3 B in VRAM at any given step. This is the canonical use case the 2026 MoE-offload research thread targets.

For a pure-GPU 4 GB candidate, smaller dense Qwen3-Coder variants (when released) and [Phi-4-mini](phi-4-mini.md) / [Gemma 3-4B](gemma-3.md) remain the practical choices.

## See Also

- [DALI: workload-aware MoE offloading](../runtimes/dali-moe.md)
- [FlashMoE: SSD-based MoE caching](../runtimes/flashmoe.md)
- [MoE-Spec: spec decoding under expert budgets](../techniques/moe-spec.md)
- [Gemma 4 / Phi-4 / Qwen3 reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md)
- [Phi-4-Mini](phi-4-mini.md)
- [Gemma 3](gemma-3.md)
