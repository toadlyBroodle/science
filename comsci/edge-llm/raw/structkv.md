# StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference

**Source:** arXiv:2604.06746 (https://arxiv.org/abs/2604.06746)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Zhirui Chen, Peiyang Liu, Ling Shao
**Venue:** ACL 2026 Findings
**Submitted:** 2026-04-08

## Abstract / extracted content

KV-cache compression for million-token-class context windows. Argues prior compression methods rely on local saliency metrics that discard tokens which look dormant at any single layer but act as global information hubs across layers. StructKV introduces:

1. Global In-Degree Centrality: aggregates attention patterns across network depth to identify global hubs.
2. Dynamic Pivot Detection: information-theoretic metric to adaptively choose the optimal layer for compression.
3. Structural Propagation and Decoupling: separates compute budget from storage budget at inference.

## Key claims

- Existing compression "systematically discards" globally important tokens flagged dormant at single layers.
- Preserves long-range dependencies and retrieval robustness.
- Validated on LongBench and RULER.

## Headline numbers

- Specific numbers not in abstract.

## Relevance to 4 GB edge target

The wiki's stated out-of-scope is >64k context, but agentic coding sometimes needs 32-64k effective context where KV cache competes with weights. StructKV is the 2026 SOTA reference for "what to drop" at compression time. Comparable thread to DASH-KV, which compresses *attention compute*; StructKV compresses *KV storage*.
