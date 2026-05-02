# DASH-KV: Asymmetric KV-Cache Hashing

> **Summary:** Guo et al., ACL 2026 Findings (arXiv:2604.19351). Reformulates attention as approximate nearest-neighbor search via asymmetric deep hashing. Differential encoding for queries vs keys reflects their different precision-vs-reuse profiles. Reduces inference complexity from O(N²) to linear O(N) for long-context attention while matching full-attention quality on LongBench.

**Sources:** [raw/dash-kv.md](../../raw/dash-kv.md), [raw/structkv.md](../../raw/structkv.md), [raw/kivi.md](../../raw/kivi.md)

---

## What DASH-KV compresses

Other 2026 KV-cache work compresses *storage* (drop tokens, quantize values). DASH-KV compresses *attention compute*: it replaces the dot-product over all KV pairs with hash-based nearest-neighbor lookup over a hashed KV index.

Asymmetric encoding:
- Queries are encoded once per step (not reused), can use shorter / coarser hash codes.
- Keys are reused across many query steps, justify deeper / more precise encoding.
- A dynamic mixed-precision path keeps full-precision compute on tokens deemed critical.

Net: O(N²) → O(N) attention.

## Compared to peers

| Method | Compresses | Approach |
|---|---|---|
| [KIVI](kivi.md) | KV storage (bits per element) | 2-bit asymmetric per-channel/per-token quantization |
| [StructKV](structkv.md) | KV storage (number of tokens) | Drops tokens by global in-degree centrality |
| **DASH-KV** | Attention compute | Asymmetric hash-based ANN search |

These are complementary: DASH-KV reduces the compute on whatever survives KIVI/StructKV's storage compression.

## Relevance to 4 GB VRAM target

KV cache often dominates VRAM beyond 32k context. Storage-side compression (KIVI, StructKV) addresses memory but does not help if you're attention-FLOP-bound on a small GPU. DASH-KV is the first 2026 paper claiming asymptotic O(N) attention while matching full-attention quality. For 4 GB VRAM at long context (32-64k coding agents), it's a load-bearing piece of the inference stack.

## Caveats

- Specific LongBench numbers not in fetched abstract.
- Implementation maturity (fused kernels for the hash-lookup path) unclear at submission time.

## See Also

- [KIVI: 2-bit KV quantization](kivi.md)
- [StructKV: structural KV pruning](structkv.md)
- [4 GB VRAM budget math](../analysis/four-gb-budget-math.md) (pending Phase 3)
