# Speculative Speculative Decoding (Saguaro / SSD)

> **Summary:** Kumar, Dao, May; ICLR 2026 (arXiv:2603.03251). Parallelizes the speculation and verification stages of speculative decoding instead of running them sequentially. While verification is in flight, the draft model preemptively prepares the next round of speculations conditioned on predicted verification outcomes. Saguaro is the optimized algorithm. **30% faster than optimized SD baselines on average; up to 5x faster than autoregressive decoding.**

**Sources:** [raw/saguaro-ssd.md](../../raw/saguaro-ssd.md), [raw/eagle-3.md](../../raw/eagle-3.md), [raw/ddtree.md](../../raw/ddtree.md)

---

## The bottleneck SSD removes

Standard speculative decoding alternates draft → verify → draft → verify. The draft model sits idle during verification and the target model sits idle during drafting. SSD overlaps the two: while the target is verifying batch *t*, the draft predicts the most likely verification outcome and pre-builds batch *t+1*. If the prediction is correct, batch *t+1* is committed without waiting.

Three SSD-specific implementation challenges (rollback on misprediction, scheduling, and batched verification) are addressed by the Saguaro algorithm.

## Headline numbers

- Mean: 1.3x over optimized SD baselines.
- Max: 5x over autoregressive decoding.

The paper does not directly compare against [EAGLE-3](eagle-3.md) in the extracted abstract; the "optimized SD baseline" likely includes EAGLE-class drafters.

## Relevance to 4 GB VRAM target

SSD is a scheduling change. It does not add VRAM cost; the draft and target model footprints are unchanged. On 4 GB devices where draft-model size is precious, the parallelism gain compounds with whatever draft fits. Pairs cleanly with EAGLE-3 / EAGLE-style auxiliary heads.

## Caveats

- Wall-clock speedup depends on hardware concurrency; benefits assume real overlap between draft and target compute (single-stream vs dual-stream CUDA).
- Acceptance-rate analysis on coding distributions (vs general chat) not extracted.

## See Also

- [EAGLE-3](eagle-3.md)
- [DDTree (block-diffusion drafter)](ddtree.md)
- [MoE-Spec (expert budgeting)](moe-spec.md)
