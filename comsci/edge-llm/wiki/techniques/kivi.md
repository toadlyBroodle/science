# KIVI: Asymmetric 2-bit KV-Cache Quantization

> **Summary:** Liu et al., ICML 2024 (arXiv:2402.02750). Tuning-free 2-bit KV-cache quantization. Empirical finding: key cache should be quantized per-channel, value cache per-token. **2.6x peak-memory reduction; up to 4x larger batch sizes; 2.35-3.47x throughput improvement.** Works across Llama, Falcon, Mistral without calibration.

**Sources:** [raw/kivi.md](../../raw/kivi.md), [raw/dash-kv.md](../../raw/dash-kv.md), [raw/structkv.md](../../raw/structkv.md)

---

## The asymmetric insight

Key activations have channel-level outliers: a small number of channels carry disproportionate magnitude. Per-channel quantization preserves this structure.

Value activations have token-level structure: outlier *tokens* (not channels) carry signal. Per-token quantization preserves this.

Asymmetry is empirically derived; the paper presents distribution analyses across multiple model families.

## What it compresses

KV *storage* (bits per element). Complementary to:
- [StructKV](structkv.md) which compresses *number of tokens* in the cache.
- [DASH-KV](dash-kv.md) which compresses *attention compute* via hashing.

## Relevance to 4 GB VRAM target

KV cache often exceeds weight memory beyond 32k context. KIVI cuts that by ~ 4x at 2 bits while remaining tuning-free. On 4 GB this can be the difference between a 16k and a 64k effective context.

## Caveat

2-bit KV at the SLM scale has not been verified by [SLMQuant](slmquant.md)'s methodology. Treat published 2.6x numbers as upper bounds for 7B+ models; expect smaller wins (and possibly larger quality drops) on 1-4 B models.

## See Also

- [DASH-KV](dash-kv.md)
- [StructKV](structkv.md)
- [SLMQuant](slmquant.md)
