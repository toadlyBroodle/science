# MoE Active-Parameter Architectures

> **Summary:** Mixture-of-Experts models that activate only a subset of total parameters per token. Modern lineup includes OLMoE (open MoE, 2024), Qwen3-MoE-A3B (3B-active), Gemma-4-26B-A4B (4B-active, 128 experts, top-2 routing), GRIN-MoE, and the agentic-coder-specific [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (80B-total / 3B-active). The 2026 thread is *not* "MoE always wins"; see [Manik & Wang's tradeoffs paper](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md); but "MoE wins when training and serving stack are both designed for it."

**Sources:** [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md), [raw/gemma-phi-qwen-tradeoffs.md](../../raw/gemma-phi-qwen-tradeoffs.md), [raw/dali-moe.md](../../raw/dali-moe.md), [raw/ktransformers.md](../../raw/ktransformers.md)

---

## Active vs total parameters

In an MoE, each token activates *active* parameters out of *total* parameters. Compute scales with active; memory scales with total. Examples:

| Model | Total | Active | Notes |
|---|---|---|---|
| Qwen3-30B-A3B | 30 B | 3 B | Generic chat |
| **Qwen3-Coder-Next** | **80 B** | **3 B** | **Agentic coder** |
| Qwen3.5-35B-A3B | 35 B | 3 B | Newer Qwen3.5 family |
| Qwen3.5-122B-A10B | 122 B | 10 B | Larger Qwen3.5 |
| Gemma-4-26B-A4B | 26 B | 4 B | 128 experts, top-2 routing |
| OLMoE-1B-7B | 7 B | 1 B | Fully open |

## Routing details (for the wiki's reference set)

- Top-K selection per layer (typically K=2). Each layer has E experts; router picks K of them per token.
- Shared experts (always active for every token) handle "common" computation.
- Routed experts are sparse and balanced via load-balancing losses during training.

## The 4 GB-VRAM specific story

Active-parameter footprint at FP16 is small (3-4 B → 6-8 GB at FP16, 1.5-2 GB at Q4 for active set). Total weights are the problem: 80 B at Q4 ≈ 40 GB cannot fit in VRAM.

Solution: offload. The 2026 stack:

1. **[KTransformers](../runtimes/ktransformers.md)**; substrate (shared experts on GPU, routed on CPU).
2. **[DALI](../runtimes/dali-moe.md)**; workload-aware GPU↔DRAM expert assignment.
3. **[FlashMoE](../runtimes/flashmoe.md)**; extends offload to SSD when DRAM is exceeded.
4. **HybriMoE** (DAC 2025); intra-layer scheduling improvements.

## The 2026 caveat

[Manik & Wang's "Dense vs MoE reasoning tradeoffs"](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md) (April 2026) found that on reasoning benchmarks (ARC, GSM8K, MATH, TruthfulQA), Gemma-4-E4B (dense) beat Gemma-4-26B-A4B (MoE) at much smaller VRAM. **Sparse activation does not automatically guarantee the best operating point.**

The MoE win for [Qwen3-Coder-Next](../models/qwen3-coder-next.md) appears to come from *agentic-coding-specific RL with environment feedback*, not from MoE alone. Solo devs choosing between dense and MoE for a 4 GB target should:
- Default to dense (simpler stack, no offload required) for general use.
- Choose MoE only when the model is specifically trained for the target task (agentic coding) and the offload stack is in place.

## See Also

- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
- [Dense vs MoE reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md)
- [DALI](../runtimes/dali-moe.md)
- [KTransformers](../runtimes/ktransformers.md)
- [FlashMoE](../runtimes/flashmoe.md)
- [Mamba/SSM hybrids](mamba-ssm-hybrids.md)
