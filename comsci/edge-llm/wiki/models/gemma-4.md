# Gemma 4

> **Summary:** Google DeepMind's open-weight family (2026-04-02, Apache 2.0): E2B (2.3B effective), E4B (4.5B effective), 12B Unified (encoder-free multimodal), 26B A4B MoE (4B active), 31B dense. Per-Layer Embeddings and shared KV cache push effective-parameter footprints below total size; QAT checkpoints (2026-06-05) cut memory ~72% vs BF16 with Q4_0 GGUF and mobile formats, putting E2B text-only under 1 GB and the 26B MoE near 15 GB. E4B at LiveCodeBench v6 52.0 is the current best-documented coding score in the under-5B-active class.

**Sources:** [raw/gemma-4.md](../../raw/gemma-4.md)

---

## Family and architecture

Five sizes; all multimodal (image/text/video), audio on E2B/E4B/12B. Architecture: alternating local sliding-window (512-1024 tokens) and global full-context attention, continuing the KV-control lineage of [Gemma 3](gemma-3.md); pruned RoPE on global layers; Per-Layer Embeddings (PLE) holding total embeddings in slow memory while the effective-parameter core stays in fast memory; last-N-layer shared KV cache. E2B/E4B context 128K, larger sizes 256K.

| Model | Active/effective | Total | Coding (LCB v6) | MMLU Pro |
|---|---|---|---|---|
| E2B | 2.3B | 5.1B | - | 60.0 |
| E4B | 4.5B | 8B | 52.0 | 69.4 |
| 26B A4B | 4B | 26B | 77.1 | 82.6 |
| 31B dense | 31B | 31B | 80.0 | 85.2 |

Numbers: Google-published, instruction-tuned, unquantized, 2026-04-02. The 26B A4B at LMArena ~1441 with 4B active is the headline MoE efficiency claim; [dense-vs-MoE tradeoff data](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md) shows E4B winning on weighted accuracy per GB in practice.

## QAT checkpoints (2026-06-05)

Quantization simulated during a fine-tuning pass so weights compensate for precision loss; Google states quality above standard PTQ baselines (no quantitative deltas published). Formats: Q4_0 GGUF ([llama.cpp](../runtimes/llama-cpp.md) / [Ollama / LM Studio](../runtimes/ollama-and-friends.md)), mobile-specialized schema (LiteRT-LM), compressed-tensors ([vLLM](../runtimes/vllm.md) / SGLang), MLX. Memory: ~72% cut vs BF16 headline; E2B text-only without PLE under 1 GB; 26B A4B ~15 GB; 31B w4a16 ~66% less VRAM.

This is the first frontier-lab QAT release covering a full family top to bottom; it removes the [SLMQuant](../techniques/slmquant.md) PTQ-degradation risk for the E-series since the vendor ships the quantized weights.

## Relevance to 4 GB VRAM target

- **E2B QAT** is the new floor: under 1 GB text-only leaves 3 GB for KV cache and a draft model. Coding capability undocumented; MMLU Pro 60.0 suggests usable general competence.
- **E4B QAT** (~2.5-3 GB estimated at Q4 from the 72% claim; measured tuple not yet published) is the direct rival to [Nemotron-3-Nano-4B](nemotron-3-nano.md) (Q4_K_M 2.9 GB, LCB 51.8): near-identical coding scores, Gemma stronger on general reasoning, Nemotron stronger on published BFCL tool-calling. Head-to-head tool-call conformance at Q4 is an open eval gap ([missing evals](../analysis/missing-evals.md)).
- **26B A4B QAT at ~15 GB** stays out of 4 GB VRAM but lands in reach of the [KTransformers](../runtimes/ktransformers.md)/[DALI](../runtimes/dali-moe.md) CPU-offload path on 32 GB-RAM laptops.
- PLE is architecturally aligned with the 4 GB budget: it is a designed slow-memory/fast-memory split rather than an afterthought offload, same direction as [-ngl partial offload](../runtimes/llama-cpp.md) but built into training.

## See Also

- [Gemma 3](gemma-3.md)
- [Nemotron-3-Nano-4B](nemotron-3-nano.md)
- [LFM2.5](lfm2-5.md)
- [MoE active-parameter architectures](../architectures/moe-active-param.md)
- [Dense vs MoE reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md)
- [4 GB VRAM budget math](../analysis/four-gb-budget-math.md)
