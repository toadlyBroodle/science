# EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test

**Source:** arXiv:2503.01840 (https://arxiv.org/abs/2503.01840)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang
**Venue:** NeurIPS 2025
**Submitted:** 2025-03-03; latest 2025-04-23
**Code:** https://github.com/SafeAILab/EAGLE

## Abstract / extracted content

Speculative decoding draft-model technique. Replaces EAGLE's feature-prediction objective with direct token prediction; introduces multi-layer feature fusion via "training-time test." EAGLE's feature-prediction constraint capped scaling gains; EAGLE-3 removes that ceiling. Evaluated on chat and reasoning models across five tasks.

## Key claims

- Direct token prediction + multi-layer fusion outperforms EAGLE-2's feature-prediction approach.
- Method works on both chat and reasoning model families.
- Scales with training data: prior EAGLE versions plateaued.

## Headline numbers

- Up to 6.5x acceleration over standard decoding.
- ~1.4x improvement over EAGLE-2.
- 1.38x throughput improvement at batch size 64 in SGLang.

## Family

- EAGLE-1 (ICML 2024)
- EAGLE-2 (EMNLP 2024)
- EAGLE-3 (NeurIPS 2025)

## Relevance to 4 GB edge target

Speculative decoding adds the draft model's footprint to VRAM. At 4 GB, EAGLE-style auxiliary draft heads (small) are tractable; full separate draft models (e.g., 0.5-1B) are not unless the target is heavily quantized. Self-speculative variants (Medusa, lookahead) are the alternative when VRAM is tight.
