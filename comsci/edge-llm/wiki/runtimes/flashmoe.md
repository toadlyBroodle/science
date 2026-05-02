# FlashMoE: SSD-Based MoE Offloading with ML Cache Replacement

> **Summary:** Kim et al., January 2026 (arXiv:2601.17063). Pushes MoE expert offloading past DRAM, to SSD. Lightweight ML-based caching strategy combining recency and frequency signals. **Up to 51% improvement in cache hit rate over LRU/LFU; up to 2.6x speedup vs existing MoE inference systems.** Demonstrated on user-grade desktop hardware.

**Sources:** [raw/flashmoe.md](../../raw/flashmoe.md), [raw/dali-moe.md](../../raw/dali-moe.md)

---

## The DRAM ceiling

Earlier MoE offload systems (Fiddler, DAOP) assume the full set of expert weights fits in host DRAM. As MoE models cross hundreds of GB, this assumption breaks on consumer hardware. FlashMoE's contribution is keeping MoE inference viable when the model exceeds DRAM.

The active expert subset still rides in VRAM (small). Hot experts that don't fit in VRAM ride in DRAM. Cold experts ride on SSD. The novel piece is the ML-based cache that decides which experts get promoted/demoted across the three tiers.

## Cache strategy

Lightweight ML model (the paper does not specify architecture in the abstract) combining:
- Recency signals (LRU-style).
- Frequency signals (LFU-style).
- Likely additional features (token-position, layer index, recent-routing distribution).

Beats LRU and LFU by up to 51% in cache hit rate.

## Hardware target

User-grade desktop with NVMe SSD. 4 GB VRAM laptops with NVMe storage are the natural extension; the paper does not benchmark that envelope explicitly.

## Position vs DALI

[DALI](dali-moe.md) optimizes the GPU↔DRAM layer and assumes all experts fit in DRAM. FlashMoE extends below DRAM to SSD. They are complementary; the 2026 production stack for very-large MoE on consumer hardware likely composes them.

## Caveats

- "Up to 2.6x speedup" baseline not specified in the abstract; reasonable to read as "vs prior DRAM-offload systems on workloads that exceed DRAM."
- Latency floor: SSD read latency (~ 50-100 µs per random read on consumer NVMe) sets a minimum per-token cost when the cold path is hit.

## See Also

- [DALI](dali-moe.md)
- [MoE-Spec](../techniques/moe-spec.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
