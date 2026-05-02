# KTransformers: CPU/GPU Hybrid Inference for MoE Models

**Source:** SOSP 2025 (https://dl.acm.org/doi/10.1145/3731569.3764843)
**Fetched:** 2026-05-02 via WebSearch
**Authors:** Chen, Hongtao et al. (Tsinghua MADSys)
**Repo:** github.com/kvcache-ai/ktransformers

## Summary

High-performance heterogeneous-inference system for MoE. Optimized AMX-specialized CPU kernels + asynchronous CPU-GPU task scheduling. Shared experts on GPU, routed experts on CPU. Novel Expert Deferral mechanism increases CPU utilization from typically <75% to nearly 100%.

## Headline numbers

- 4.62-19.74x prefill speedup vs existing systems.
- 1.25-4.09x decode speedup.
- Up to 1.45x additional throughput from Expert Deferral on top.
- <0.5% mean accuracy drop across benchmarks.

## Position

KTransformers is the practitioner-default 2025 framework for "run a giant MoE on a consumer GPU + lots of RAM." It's the substrate that DALI (2026), HybriMoE (2025), and FlashMoE (2026) extend. Famous for running DeepSeek-V3 (671B) on a single GPU.

## Relevance to 4 GB VRAM target

For Qwen3-Coder-Next (80B/3B-active) on a 4 GB GPU, KTransformers is the baseline. DALI improves expert assignment; FlashMoE adds SSD tier; HybriMoE adds intra-layer scheduling. KTransformers itself ships and works today.
