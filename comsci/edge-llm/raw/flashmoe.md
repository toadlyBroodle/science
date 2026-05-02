# FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for MoE Inference on Edge Devices

**Source:** arXiv:2601.17063 (https://arxiv.org/abs/2601.17063)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Byeongju Kim, Jungwan Lee, Donghyeon Han, Hoi-Jun Yoo, Sangyeob Kim
**Submitted:** 2026-01-22

## Abstract / extracted content

Pushes MoE expert offloading past DRAM, to SSD. Earlier MoE offloading systems (Fiddler, DAOP) assume DRAM-based offload and break when models exceed hundreds of gigabytes on memory-constrained edge devices. FlashMoE pairs SSD-based expert offloading with a lightweight ML-based caching strategy that adaptively combines recency and frequency signals to maximize expert reuse.

## Key claims

- SSD-based offload is feasible for hundreds-of-GB MoE on edge.
- ML-based hybrid cache (recency + frequency) substantially beats LRU and LFU.
- Up to 51% improvement in cache hit rate over LRU/LFU.
- Up to 2.6x speedup vs existing MoE inference systems.

## Hardware

- User-grade desktop with real hardware implementation.

## Relevance to 4 GB edge target

For a 4 GB GPU + 16-32 GB system RAM + NVMe SSD, FlashMoE is the path that lets a >100 GB MoE actually run. Pairs naturally with DALI (DALI optimizes CPU-GPU allocation; FlashMoE handles the SSD tail). Together they define the 2026 SOTA for "MoE on a tight VRAM budget."
