# StructKV: Structure-Aware KV-Cache Compression

> **Summary:** Chen, Liu, Shao; ACL 2026 Findings (arXiv:2604.06746). KV-cache compression for million-token context windows. Argues prior compression methods rely on local saliency metrics that drop tokens which look dormant at any single layer but act as global information hubs across layers. Introduces Global In-Degree Centrality, Dynamic Pivot Detection, and Structural Propagation/Decoupling. Validated on LongBench and RULER.

**Sources:** [raw/structkv.md](../../raw/structkv.md), [raw/dash-kv.md](../../raw/dash-kv.md), [raw/kivi.md](../../raw/kivi.md)

---

## The "structural skeleton" thesis

Prior KV-pruning methods (H2O, SnapKV, ChunkKV) score tokens by per-layer attention magnitude, then drop the low-scoring ones. StructKV's claim: a token can look dormant at any single layer while serving as a global information hub when attention patterns are aggregated across layers. Dropping it on a per-layer signal silently destroys long-range dependencies and retrieval robustness.

Three pieces:

1. **Global In-Degree Centrality.** Aggregate attention received by each token across all layers, identify hubs.
2. **Dynamic Pivot Detection.** Information-theoretic metric to choose the optimal layer for compression rather than compressing uniformly.
3. **Structural Propagation and Decoupling.** Separate the compute budget from the storage budget at inference; tokens kept for compute are not necessarily kept for storage.

## Compared to peers

- [KIVI](kivi.md): compresses bits per KV element (storage, no token drops).
- [DASH-KV](dash-kv.md): compresses attention compute via hashing.
- **StructKV**: drops tokens by global structural importance.

The three are complementary on the 4 GB target.

## Relevance

The wiki's stated out-of-scope is >64k context, but agentic coding sometimes needs 32-64k effective context. StructKV is the 2026 reference for "what to drop" when the cache must shrink. ACL 2026 Findings status (peer-reviewed) gives it more weight than the abstract-only candidates.

## See Also

- [KIVI](kivi.md)
- [DASH-KV](dash-kv.md)
