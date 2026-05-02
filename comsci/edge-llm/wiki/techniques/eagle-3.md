# EAGLE-3

> **Summary:** Li, Wei, Zhang, Zhang; NeurIPS 2025 (arXiv:2503.01840). Speculative-decoding draft-model technique. Replaces EAGLE's feature-prediction objective with direct token prediction; introduces multi-layer feature fusion via "training-time test." **Up to 6.5x acceleration over standard decoding; ~1.4x improvement over EAGLE-2.** As of mid-2025 the SOTA autoregressive draft head; the 2026 [Saguaro/SSD](saguaro-ssd.md) and [DDTree](ddtree.md) papers measure against it.

**Sources:** [raw/eagle-3.md](../../raw/eagle-3.md), [raw/saguaro-ssd.md](../../raw/saguaro-ssd.md), [raw/ddtree.md](../../raw/ddtree.md), [raw/moe-spec.md](../../raw/moe-spec.md)

---

## What changed in EAGLE-3

EAGLE-1/2 trained the draft head to predict the *features* (hidden states) of the target model. The intuition was that features carry richer signal than tokens. EAGLE-3 abandons that: as training data scaled up, feature prediction hit a ceiling that direct token prediction did not.

Two changes:

1. Direct token prediction objective.
2. Multi-layer feature fusion via "training-time test"; uses low, mid, and high-level features rather than only the top layer.

Result: removes the EAGLE-2 scaling ceiling.

## Headline numbers

- Up to 6.5x over standard decoding.
- ~1.4x over EAGLE-2.
- 1.38x throughput at batch size 64 in SGLang.
- Works on chat models and reasoning models across five tasks.

## Position in 2026 SD landscape

EAGLE-3 is the strong autoregressive baseline that 2026 papers either compose with or replace:

- **Compose:** [Saguaro/SSD](saguaro-ssd.md) wraps EAGLE-3 in parallel scheduling. [MoE-Spec](moe-spec.md) addresses MoE-specific verification cost when EAGLE-3 is used inside an MoE.
- **Replace:** [DDTree](ddtree.md) argues block-diffusion drafters beat autoregressive draft heads.

## Relevance to 4 GB VRAM target

EAGLE-style auxiliary draft heads are small (single-layer + LM head). Footprint cost is modest, often <500 MB. This makes EAGLE-3 the practical SD choice on a 4 GB GPU; full separate draft models (e.g., 1B draft + 7B target) don't fit. For self-speculative alternatives that avoid even the small head, see Medusa/lookahead (pending coverage).

## See Also

- [Saguaro / SSD](saguaro-ssd.md)
- [DDTree](ddtree.md)
- [MoE-Spec](moe-spec.md)
- [Spec decoding at 4 GB](../analysis/spec-decoding-at-4gb.md) (pending Phase 3)
