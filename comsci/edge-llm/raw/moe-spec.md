# MoE-Spec: Expert Budgeting for Efficient Speculative Decoding

**Source:** arXiv:2602.16052 (https://arxiv.org/abs/2602.16052)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan
**Submitted:** 2026-02-17

## Abstract / extracted content

Speculative decoding's draft-tree verification activates many unique experts in MoE models, creating memory pressure that can erase the benefit of speculation entirely. MoE-Spec is a training-free verification-time method that enforces a fixed per-layer expert capacity, loading only the experts that contribute most to verification and dropping the long tail.

## Key claims

- 10-30% higher throughput than EAGLE-3 baseline at comparable quality.
- Decouples speculation depth from memory cost; allows quality/latency trades via tighter expert budgets.
- Training-free; applied at verification time.

## Relevance to 4 GB edge target

MoE + speculative decoding on 4 GB has a worst-case explosion in active expert footprint during verification. MoE-Spec is the technique that prevents that. Paired with EAGLE-3 / Saguaro / DDTree, this is the 2026 stack for fast MoE inference under tight VRAM.
