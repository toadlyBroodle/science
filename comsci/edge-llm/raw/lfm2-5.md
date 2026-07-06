# LFM2.5 family (Liquid AI, May-June 2026) — source extract

Extracted 2026-07-06 from:
- https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- https://www.liquid.ai/blog/lfm2-5-8b-a1b (published 2026-05-28)
- https://www.liquid.ai/blog/lfm2-5-230m (published 2026-06-25)

## LFM2.5-8B-A1B (released 2026-05-28)

- Open-weight on-device MoE. 8B total parameters (8.3B per third-party coverage), ~1B-1.5B active per token.
- Architecture: MoE + GQA + gated short convolution blocks (LFM2 lineage hybrid).
- Vocabulary 128,000 tokens (2x the 65,536 of LFM2-8B-A1B). Context 128K (4x the 32,768 of LFM2-8B-A1B).
- Pretraining: 38T tokens (up from 12T for LFM2-8B-A1B). Stages: embedding-only adaptation, full continued pretraining, 2T-token midtraining for 32K context, 400B-token midtraining for 128K extension, targeted preference optimization ("doom loop" mitigation for long reasoning traces), RL with knowledge-aware reward shaping (avg@k-based hallucination rewards).
- Reasoning-first: emits explicit chain-of-thought before final answers.

### Benchmarks (Liquid-published)

| Benchmark | Score |
|---|---|
| IFEval | 91.84 |
| IFBench | 56.47 |
| Multi-IF | 79.93 |
| MATH500 | 88.76 |
| AIME25 | 42.53 |
| AIME26 | 50.00 |
| BFCL v3 | 64.79 |
| BFCL v4 | 49.73 |
| Tau^2-Bench Telecom | 88.07 |
| Tau^2-Bench Retail | 39.82 |
| AA-Omniscience Index | -24.70 (range -100..100) |
| AA-Omniscience Non-Hallucination Rate | 63.47% |

Deltas vs LFM2-8B-A1B: Tau^2 Telecom 13.60 -> 88.07; IFBench 26.00 -> 56.47; non-hallucination 7.46 -> 63.47.

### Speed / memory (Liquid-published)

- Apple M5 Max CPU decode: 253 tok/s. AMD Ryzen AI Max+ 395: 146 tok/s. Snapdragon mobile: ~30 tok/s.
- Memory: under 6 GB on laptops (quantized).
- H100 SXM5: 18.5K output tok/s peak at high concurrency.
- Runtimes at launch: llama.cpp (GGUF), MLX, vLLM, SGLang, ONNX, LEAP.

## LFM2.5-230M (released 2026-06-25)

- 230M parameters, LFM2 architecture. 32K context. 19T pretraining tokens.
- Positioned as fine-tune-and-deploy base for narrow agentic tasks on any CPU.

### Benchmarks (Liquid-published)

| Benchmark | Score |
|---|---|
| GPQA Diamond | 25.41 |
| MMLU-Pro | 20.25 |
| IFEval | 71.71 |
| IFBench | 38.40 |
| Multi-IF | 37.70 |
| BFCL v3 | 43.26 |
| BFCL v4 | 21.03 |
| Tau^2-Bench Telecom | 5.26 |
| Tau^2-Bench Retail | 13.68 |
| CaseReportBench | 22.51 |

Liquid claims it beats models 2-4x its size on extraction and instruction following.

### Speed / memory (Liquid-published, 4-bit quantized)

- Raspberry Pi 5: 523 tok/s prefill, 42 tok/s decode, 293 MB.
- Galaxy S25 Ultra (Snapdragon Gen4): 1,158 tok/s prefill, 213 tok/s decode, 375 MB.
- Runtimes: llama.cpp, MLX, vLLM, SGLang, ONNX.
- Checkpoints: LFM2.5-230M and LFM2.5-230M-Base on Hugging Face.

## Notes

- License described as open-weight, downloadable/fine-tunable/deployable; exact license text not extracted (LFM Open License v1.0 applied to prior LFM2 releases; verify on model card).
- Third-party coverage: MarkTechPost 2026-05-28 (8B-A1B), MarkTechPost 2026-01-06 (LFM2.5 initial family), llmrumors.com.
