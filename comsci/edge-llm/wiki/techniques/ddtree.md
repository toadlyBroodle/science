# DDTree: Block-Diffusion Draft Trees for Speculative Decoding

> **Summary:** Ringel & Romano, April 2026 (arXiv:2604.12989). Builds speculative-decoding draft trees from per-position distributions of a block-diffusion drafter (DFlash). Best-first heap selection under fixed node budget; single-pass target verification via ancestor-only attention masking. The block-diffusion drafter generates an entire draft block per forward pass and outperforms autoregressive drafters such as [EAGLE-3](eagle-3.md).

**Sources:** [raw/ddtree.md](../../raw/ddtree.md), [raw/eagle-3.md](../../raw/eagle-3.md), [raw/saguaro-ssd.md](../../raw/saguaro-ssd.md)

---

## What's new

Two compounding ideas:

1. **Block-diffusion drafter.** A non-autoregressive drafter (DFlash, the underlying architecture) emits per-position distributions for an entire draft block in one forward pass. Compared to autoregressive draft heads (EAGLE family), this saves N-1 sequential draft steps per block of length N.

2. **Draft tree from those distributions.** Rather than commit to a single draft sequence, DDTree constructs a tree of candidate continuations using a best-first heap algorithm under a fixed node budget. Ancestor-only attention masking lets the target model verify the entire tree in a single pass.

## Compared to EAGLE family

EAGLE-3 (autoregressive drafter, NeurIPS 2025) sets the prior SD baseline. DDTree's central claim is that block-diffusion drafters beat autoregressive drafters on speculative-decoding throughput, with DDTree's tree-construction algorithm extracting more value from the per-position distributions than autoregressive sampling can.

Specific speedup-over-EAGLE-3 numbers were not in the fetched abstract.

## Relevance to 4 GB VRAM target

The trade vs EAGLE-3 on a 4 GB device:
- DFlash drafter weights vs EAGLE-3 head weights: footprint comparison not yet quantified in the wiki.
- Per-step latency: DFlash one-shot vs EAGLE-3 N sequential steps. On a small GPU with limited parallelism, the win may be smaller than on a server GPU.
- Pairs with [Saguaro/SSD](saguaro-ssd.md) (parallel scheduling) and [MoE-Spec](moe-spec.md) (expert budgeting under SD) for stacking.

## See Also

- [EAGLE-3](eagle-3.md)
- [Saguaro / SSD](saguaro-ssd.md)
- [MoE-Spec](moe-spec.md)
