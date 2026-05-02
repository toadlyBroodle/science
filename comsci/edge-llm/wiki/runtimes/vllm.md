# vLLM

> **Summary:** UC Berkeley + community. The dominant server-side LLM runtime. PagedAttention for KV-cache memory efficiency, continuous batching for throughput, prefix caching for repeated-prefix workloads, native speculative decoding (EAGLE-style draft heads, Medusa, ngram). Less common on 4 GB laptop GPUs than llama.cpp but increasingly viable as the project optimizes for smaller deployments.

**Sources:** vLLM documentation, references in [raw/lfm2.md](../../raw/lfm2.md)

---

## Why vLLM matters even at 4 GB

Three features that compound on the edge:

1. **PagedAttention.** KV cache stored in fixed-size pages, reducing fragmentation. On a tight VRAM budget, paged storage means actual usable context is closer to theoretical max.
2. **Prefix caching.** For an agentic loop where system prompt + tool definitions are reused turn after turn, prefix caching avoids re-encoding 2-5 K tokens per request. The biggest single latency win for agentic harnesses.
3. **Native speculative decoding.** First-class support for EAGLE / Medusa / ngram drafters; quicker integration of new SD techniques than llama.cpp.

## Quantization support

- AWQ, GPTQ, FP8 (Hopper), Marlin, INT4.
- Less universal than GGUF; format choice matters for sharing.

## Position vs llama.cpp

- llama.cpp wins on hardware breadth and quant breadth (CPU + every GPU vendor + every quant flavor).
- vLLM wins on server-side throughput and feature recency (KV-cache features, SD).
- For a single-user agentic-coding harness on a laptop, llama.cpp is the default; vLLM becomes attractive when SD or aggressive prefix caching is the bottleneck.

## Pairs with

- [Saguaro/SSD](../techniques/saguaro-ssd.md): the SD scheduling improvements have reference implementations in SGLang/vLLM.
- [EAGLE-3](../techniques/eagle-3.md): native draft-head support.
- [DASH-KV](../techniques/dash-kv.md), [StructKV](../techniques/structkv.md): expected to land as KV-cache backends as the 2026 work matures.

## See Also

- [llama.cpp](llama-cpp.md)
- [KTransformers](ktransformers.md)
- [EAGLE-3](../techniques/eagle-3.md)
