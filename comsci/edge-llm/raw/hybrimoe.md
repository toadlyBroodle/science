# HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for MoE Inference

**Source:** arXiv:2504.05897 (https://arxiv.org/abs/2504.05897)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Shuzhang Zhong, Yanfan Sun, Ling Liang, Runsheng Wang, Ru Huang, Meng Li
**Venue:** DAC 2025
**Submitted:** 2025-04-08

## Abstract / extracted content

Hybrid CPU-GPU MoE inference framework. Built atop KTransformers. Three contributions:

1. Dynamic intra-layer scheduling balancing CPU and GPU work.
2. Impact-driven inter-layer prefetching.
3. Score-based caching for unpredictable expert activation.

## Headline numbers

- 1.33x prefill speedup vs SOTA hybrid MoE inference baseline.
- 1.70x decode speedup.

## Position vs DALI / KTransformers / FlashMoE

HybriMoE is a 2025 contribution that DALI (Feb 2026), FlashMoE (Jan 2026), and FluxMoE (April 2026) build on. KTransformers is its base framework. The 2026 papers frame their contributions in terms of further gains over HybriMoE-class hybrid scheduling.

## Code

github.com/PKU-SEC-Lab/HybriMoE
