# DDTree: Accelerating Speculative Decoding with Block Diffusion Draft Trees

**Source:** arXiv:2604.12989 (https://arxiv.org/abs/2604.12989)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Liran Ringel, Yaniv Romano
**Submitted:** 2026-04-14

## Abstract / extracted content

Constructs draft trees from per-position distributions of a block-diffusion drafter (DFlash). Best-first heap algorithm under a fixed node budget selects continuations likely to match the target. Single-pass target-model verification via ancestor-only attention masking. Block diffusion drafters generate an entire draft block in a single forward pass and outperform autoregressive drafters such as EAGLE-3 on speculative-decoding throughput.

## Key claims

- Block-diffusion drafter (DFlash) outperforms EAGLE-3 in speculative decoding.
- Draft trees beat single-trajectory draft sequences.
- Best-first heap under fixed node budget; ancestor-only attention masking enables single-pass verification.

## Headline numbers

- Specific speedup numbers not in abstract.
- Beats EAGLE-3 (autoregressive drafter) on standard SD benchmarks.

## Relevance to 4 GB edge target

A diffusion-based drafter that produces a whole block per forward pass changes the throughput math vs auto-regressive drafters like EAGLE. On VRAM-constrained devices, the question is whether the diffusion-drafter weights cost less than EAGLE-3 head weights for equivalent acceptance rate; the paper does not yet quantify this footprint trade.
