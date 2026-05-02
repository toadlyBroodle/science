# Phi-4-Mini

> **Summary:** Microsoft, March 2025 (arXiv:2503.01743). 3.8B-parameter compact model; Phi-4-Multimodal variant integrates text, vision, and speech via Mixture-of-LoRAs (MoLoRAs) with modality-specific routers. Outperforms similar-sized open models and matches double-sized models on math and coding. The reasoning-enhanced variant rivals DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B.

**Sources:** [raw/phi-4-mini.md](../../raw/phi-4-mini.md), [raw/deepseek-r1.md](../../raw/deepseek-r1.md), [raw/gemma-phi-qwen-tradeoffs.md](../../raw/gemma-phi-qwen-tradeoffs.md)

---

## Architecture

- 3.8 B parameters.
- Vocabulary: 200K tokens (up from Phi-3.5-Mini, supports multilingual).
- Group query attention for efficient long-sequence generation.
- Speech/audio LoRA component: 460 M parameters.

## Mixture-of-LoRAs

Multimodal extension uses modality-specific LoRA adapters with routers, allowing combinations (vision+text, vision+speech, speech-only) without inter-modal interference. Reduces parameter overhead vs full multi-modal pretraining.

## Training data

High-quality web + synthetic; recipe weighted heavily toward math and coding (the Phi family signature).

## Benchmarks

- Matches 2x-larger models on math/coding reasoning (specific numbers not in extracted abstract).
- Phi-4-Multimodal: first place on OpenASR leaderboard.
- Reasoning variant on par with or surpassing [DeepSeek-R1-Distill-Qwen-7B](deepseek-r1.md) and DeepSeek-R1-Distill-Llama-8B.

The [Gemma 4 / Phi-4 / Qwen3 reasoning tradeoffs](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md) paper (April 2026) included Phi-4-mini-reasoning and Phi-4-reasoning. Phi excelled on TruthfulQA but Gemma-4-E4B was the overall winner.

## Relevance to 4 GB VRAM target

Phi-4-Mini at Q4 ≈ 2.3 GB weights. Strongest small-model candidate for math/coding tasks under tight VRAM. The reasoning variant adds non-trivial KV-cache pressure at long CoT outputs; pair with [KV-cache compression](../techniques/structkv.md) or [DASH-KV](../techniques/dash-kv.md) for long-context agentic use.

Caveat per [SLMQuant](../techniques/slmquant.md): published claims of "Q4 with negligible accuracy loss" derived on 7B+ LLMs may not hold at the 3.8 B SLM scale. Verify with task-specific evals before locking in a quant choice.

## See Also

- [Gemma 3](gemma-3.md)
- [DeepSeek-R1 distills](deepseek-r1.md)
- [LFM2](lfm2.md)
- [SLMQuant](../techniques/slmquant.md)
- [Mixture-of-Layers distillation](../training/mol-distillation.md)
