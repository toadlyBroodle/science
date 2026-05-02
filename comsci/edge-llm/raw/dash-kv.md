# DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing

**Source:** arXiv:2604.19351 (https://arxiv.org/abs/2604.19351)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang
**Venue:** ACL 2026 Findings
**Submitted:** 2026-04-21; v2 2026-04-22

## Abstract / extracted content

Reformulates attention as approximate nearest-neighbor search via asymmetric deep hashing. Differential encoding of queries vs keys reflects their distinct precision-vs-reuse profiles. Reduces inference complexity from O(N²) to linear O(N) for long-context attention, while keeping a dynamic mixed-precision path for full-precision computation on tokens deemed critical.

## Key claims

- O(N²) → O(N) attention complexity for long-context inference.
- Asymmetric encoding for Q vs K respects their different roles.
- Dynamic mixed precision preserves full-precision computation where needed.
- Outperforms baseline KV-cache compression while matching full-attention quality.

## Benchmarks

- LongBench (numbers not in abstract).

## Relevance to 4 GB edge target

KV cache often dominates VRAM beyond 32k context. DASH-KV addresses both the memory and the FLOP cost of long-context attention. Among 2026 KV-compression papers, DASH-KV is the first claiming asymptotic O(N) attention while still matching full-attention quality.
