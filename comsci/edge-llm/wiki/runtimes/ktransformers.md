# KTransformers

> **Summary:** Tsinghua MADSys, SOSP 2025. High-performance heterogeneous-inference system for MoE models. AMX-specialized CPU kernels + asynchronous CPU-GPU task scheduling. Shared experts on GPU, routed experts on CPU. **4.62-19.74x prefill speedup, 1.25-4.09x decode speedup vs prior systems. <0.5% mean accuracy drop.** Famous for running DeepSeek-V3 (671B) on a single GPU. The substrate that 2026 work (DALI, FlashMoE, HybriMoE) extends.

**Sources:** [raw/ktransformers.md](../../raw/ktransformers.md), [raw/dali-moe.md](../../raw/dali-moe.md), [raw/hybrimoe.md](../../raw/hybrimoe.md), [raw/flashmoe.md](../../raw/flashmoe.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md)

---

## The core architectural decision

For an MoE model: shared experts (frequently activated) live on GPU; routed experts (per-token sparse) live on CPU. KTransformers' AMX-specialized CPU kernels make CPU-side expert compute fast enough that the CPU↔GPU pipeline overlaps cleanly.

The Expert Deferral mechanism extends this: defer some expert computations across layers to maximize the overlap window between GPU and CPU. CPU utilization rises from typically <75% to nearly 100%, yielding +1.45x throughput on top of the base optimization.

## Headline numbers

| Metric | KTransformers vs prior |
|---|---|
| Prefill speedup | 4.62-19.74x |
| Decode speedup | 1.25-4.09x |
| Mean accuracy drop | <0.5% |
| Expert Deferral additional throughput | up to 1.45x |

Famous demo: DeepSeek-V3 (671B parameters) running on a single consumer GPU.

## Position in the 2026 stack

| Layer | Component |
|---|---|
| Substrate | **KTransformers** (CPU+GPU, shared/routed expert split) |
| Scheduling improvements | [HybriMoE](../techniques/) (intra-layer + impact-driven prefetch) |
| Expert assignment | [DALI](dali-moe.md) (0-1 integer optimization, residual prefetch) |
| SSD tier | [FlashMoE](flashmoe.md) (ML-based cache, beyond DRAM) |
| Verification budgeting | [MoE-Spec](../techniques/moe-spec.md) (under speculative decoding) |

KTransformers ships and works today. The 2026 papers offer additional gains over its baseline; not all are integrated yet.

## Relevance to 4 GB VRAM target

For [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (80B/3B-active) on 4 GB:
- Active 3B + shared experts ride in VRAM.
- Routed experts ride in DRAM (16-128 GB host RAM).
- KTransformers handles the orchestration.

This is the canonical 2026 deployment pattern for "MoE on a tight VRAM budget."

## See Also

- [DALI](dali-moe.md)
- [FlashMoE](flashmoe.md)
- [llama.cpp](llama-cpp.md)
- [MoE-Spec](../techniques/moe-spec.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
