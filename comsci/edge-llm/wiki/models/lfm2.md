# LFM2 (Liquid Foundation Models v2)

> **Superseded:** the LFM2.5 generation (May-June 2026) replaces this family for new deployments; see [LFM2.5](lfm2-5.md).

> **Summary:** Liquid AI, November 2025 (arXiv:2511.23404). Family of edge-targeted models 350M to 8.3B parameters. Hybrid architecture: gated short convolutions + grouped query attention. Hardware-in-the-loop architecture search. Pre-trained on 10-12T tokens with tempered, decoupled top-K knowledge distillation. Up to 2x faster prefill/decode on CPUs vs same-sized models.

**Sources:** [raw/lfm2.md](../../raw/lfm2.md)

---

## Variants

- Text: 350M, 700M (approx), 1.2B, 2.6B, 8.3B (size lineup approximated; see paper for exact list).
- LFM2-VL (vision)
- LFM2-Audio (speech, real-time speech-to-speech)
- LFM2-ColBERT (retrieval)

## Architecture

Hybrid backbone:
- Gated short convolutions for fast token mixing (similar in spirit to [Mamba/SSM hybrids](../architectures/) coverage planned).
- Grouped query attention layers interleaved.
- Discovered via hardware-in-the-loop architecture search optimizing for latency and memory on edge targets, not just accuracy.

This is one of the few recent families designed *backwards* from the deployment target.

## Training

- 10-12 T tokens pre-training.
- Tempered, decoupled top-K knowledge distillation (specifics in paper).
- Three-stage post-training:
  1. Supervised fine-tuning.
  2. Length-normalized preference optimization (DPO variant).
  3. Model merging.

## Benchmarks (LFM2-2.6B)

- IFEval: 79.56%
- GSM8K: 82.41%

Specific coding benchmarks not extracted.

## Deployment packages

Open-weight; first-class deployment for ExecuTorch, llama.cpp, and vLLM is shipped with the release.

## Relevance to 4 GB VRAM target

LFM2 is purpose-built for edge. The 1.2B and 2.6B variants are very comfortable in 4 GB at Q4 (sub-2 GB weights), leaving substantial headroom for KV cache. The 8.3B variant is borderline; at Q4 ~ 4.5 GB weights, would require offload.

The CPU-prefill speedup (2x vs same-size models) matters when the GPU is busy with KV cache or when the model is too large for the GPU and partial CPU inference is in play. Pairs naturally with [llama.cpp `-ngl` partial offload](../runtimes/) and [DALI](../runtimes/dali-moe.md)-style hybrid inference.

## See Also

- [Phi-4-Mini](phi-4-mini.md)
- [Gemma 3](gemma-3.md)
- [Qwen3-Coder-Next](qwen3-coder-next.md)
- [DALI](../runtimes/dali-moe.md)
