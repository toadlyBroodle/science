# Speculative Speculative Decoding (SSD) / Saguaro

**Source:** arXiv:2603.03251 (https://arxiv.org/abs/2603.03251)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Tanishq Kumar, Tri Dao, Avner May
**Venue:** ICLR 2026
**Submitted:** 2026-03-03; revised 2026-03-22

## Abstract / extracted content

Parallelizes the speculation and verification stages of speculative decoding instead of running them sequentially. While verification of the current draft batch is in flight, the draft model preemptively predicts likely verification outcomes and prepares the next round of speculations conditioned on those predicted outcomes. If the actual verification result lands in the predicted set, the prepared speculation is returned immediately. Saguaro is the optimized algorithm that resolves three key SSD implementation challenges.

## Key claims

- 30% faster than optimized speculative decoding baselines on average.
- Up to 5x faster than standard autoregressive decoding.
- Removes the sequential dependency that bounds prior speculative-decoding speedups.

## Headline numbers

- Mean: 1.3x over optimized SD baselines.
- Max: 5x over autoregressive.

## Relevance to 4 GB edge target

SSD increases throughput without adding model VRAM cost (it's a scheduling change). On 4 GB devices where draft-model footprint is precious, the parallelism win compounds with whatever draft model fits.
