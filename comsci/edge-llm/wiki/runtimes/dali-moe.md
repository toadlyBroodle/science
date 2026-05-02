# DALI: Workload-Aware MoE Offloading for Local PCs

> **Summary:** Zhu et al., February 2026 (arXiv:2602.03495). MoE expert-offloading framework targeting consumer GPU + host RAM. Three contributions: dynamic 0-1 integer optimization for expert-to-device assignment, residual-based prefetching using inter-layer signals, workload-aware GPU cache replacement. Significant speedups over prior MoE offloading frameworks in both prefill and decode.

**Sources:** [raw/dali-moe.md](../../raw/dali-moe.md), [raw/flashmoe.md](../../raw/flashmoe.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md)

---

## What DALI does differently

Prior MoE-offload frameworks (Fiddler, DAOP, kTransformers' first generation) made three suboptimal choices DALI fixes:

| Issue | Prior approach | DALI fix |
|---|---|---|
| Expert-to-device assignment | Static (compile-time) | Dynamic 0-1 integer optimization, runtime Greedy Assignment |
| Expert prefetch | Heuristic / nothing | Residual-based prediction using inter-layer signals |
| GPU cache replacement | LRU / LFU | Workload-aware exploiting temporal correlation in expert activations |

## Hardware target

The paper explicitly targets local PCs: a consumer GPU (4-24 GB VRAM) plus host RAM (16-128 GB). This is exactly the wiki's 4 GB VRAM scenario.

## Position in 2026 MoE-offload stack

DALI optimizes the GPU↔CPU partition. [FlashMoE](flashmoe.md) extends offload to SSD for the cold-expert tail (when system RAM is also exceeded). [MoE-Spec](../techniques/moe-spec.md) constrains expert footprint during speculative-decoding verification. Together these define the 2026 toolkit for "MoE on tight VRAM."

## Relevance to 4 GB VRAM target

For [Qwen3-Coder-Next](../models/qwen3-coder-next.md) at 80B/3B-active, DALI is the central piece of the deployment puzzle. Active 3B fits in 4 GB, but the 80B total of expert weights doesn't. DALI decides which experts ride in VRAM, which in DRAM, and how to pre-warm the next ones based on residual signals.

## Caveats

- Specific benchmark numbers not extracted from the abstract.
- Implementation maturity, kernel coverage, and ease of integration with llama.cpp / vLLM / KTransformers not yet documented in this wiki page.

## See Also

- [FlashMoE](flashmoe.md)
- [MoE-Spec](../techniques/moe-spec.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
