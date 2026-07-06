# Gemma 3

> **Superseded:** Gemma 4 (April 2026, QAT checkpoints June 2026) replaces this family for new deployments; see [Gemma 4](gemma-4.md).

> **Summary:** Google DeepMind's lightweight open-weight family (1B / 4B / 12B / 27B), March 2025. Multimodal, 128K+ context. The architectural headline is a rebalanced local-vs-global attention ratio with shorter local-attention spans, specifically to control KV-cache memory at long context. Trained with knowledge distillation. Gemma3-4B-IT competitive with Gemma2-27B-IT.

**Sources:** [raw/gemma-3.md](../../raw/gemma-3.md)

---

## Variants

- 1 B
- 4 B (the strongest dense candidate for 4 GB VRAM at Q4)
- 12 B
- 27 B

## Architecture: KV-cache mitigation

The standout technical decision in Gemma 3 is the explicit rebalancing of attention layers to fight KV-cache memory growth at 128K+ context:

- Increased ratio of local-attention to global-attention layers.
- Shortened local-attention span.
- Net effect: smaller KV cache footprint at long context (specific reduction percentage not in the extracted abstract).

This matters disproportionately for the 4 GB VRAM target, where KV cache often exceeds weight memory beyond 32K tokens. See [`analysis/four-gb-budget-math.md`](../analysis/four-gb-budget-math.md) (pending).

## Training

Knowledge distillation across all sizes; 4B-IT matches the prior generation's 27B on benchmark suites.

## Multimodal

Vision understanding integrated; specifics on tokenizer, vision encoder, and image-text alignment not extracted from abstract page.

## Benchmarks

Comparative claims:

- Gemma3-4B-IT ≈ Gemma2-27B-IT.
- Gemma3-27B-IT ≈ Gemini-1.5-Pro.

The newer Gemma 4 supersedes Gemma 3 (April 2026, MoE 26B / 4B-active variant); see [Gemma 4 / Phi-4 / Qwen3 reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md) for direct numbers.

## Relevance to 4 GB VRAM target

Gemma 3-4B at Q4 weights ≈ 2.4 GB. With Gemma 3's local-global attention split, KV cache at 32K context stays manageable in the remaining ~1.5 GB after CUDA workspace overhead. This is the leading dense 4 GB candidate for general-purpose use; the agentic-coder candidate is a quantized [Qwen3-Coder-Next](qwen3-coder-next.md) variant via offload.

## See Also

- [Phi-4-Mini](phi-4-mini.md)
- [LFM2](lfm2.md)
- [DeepSeek-R1 distills](deepseek-r1.md)
- [Gemma 4 / Phi-4 / Qwen3 reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md)
