# LFM2.5 (Liquid Foundation Models v2.5)

> **Summary:** Liquid AI's 2026 successor to [LFM2](lfm2.md). Two edge-relevant releases inside the May-June 2026 window: LFM2.5-8B-A1B (2026-05-28), an 8B-total / ~1.5B-active on-device MoE with 128K context, reasoning-first output, and BFCL v3 64.79; and LFM2.5-230M (2026-06-25), a 230M CPU-anywhere agent base running in under 400 MB at 4-bit. Both ship GGUF/MLX/vLLM/ONNX support at launch. The 8B-A1B is the strongest open on-device MoE at its active-parameter count as of June 2026.

**Sources:** [raw/lfm2-5.md](../../raw/lfm2-5.md), [raw/lfm2.md](../../raw/lfm2.md)

---

## LFM2.5-8B-A1B (2026-05-28)

Architecture continues the LFM2 hybrid lineage (gated short convolutions + GQA) with MoE routing: 8B total, ~1B-1.5B active per token. Versus LFM2-8B-A1B: context 32K to 128K, vocabulary 65,536 to 128,000, pretraining 12T to 38T tokens. Post-training adds targeted preference optimization against reasoning "doom loops" and RL with knowledge-aware (avg@k) hallucination rewards. Emits explicit chain-of-thought before answers.

Agentic numbers (Liquid-published, 2026-05-28, unquantized, harness unknown):

| Benchmark | Score | Context |
|---|---|---|
| BFCL v3 | 64.79 | above [Nemotron-3-Nano-4B](nemotron-3-nano.md)'s 61.1, below [Mellum2](mellum2.md)'s 66.3 |
| BFCL v4 | 49.73 | |
| Tau^2 Telecom | 88.07 | up from 13.60 for LFM2-8B-A1B |
| IFEval | 91.84 | |
| MATH500 | 88.76 | |

No SWE-bench or LiveCodeBench numbers published; the family is tuned for tool use and structured output, not repository-scale coding.

## LFM2.5-230M (2026-06-25)

230M parameters, 32K context, 19T tokens. 4-bit footprint: 293 MB on Raspberry Pi 5 (42 tok/s decode), 375 MB on Galaxy S25 Ultra (213 tok/s decode). BFCL v3 43.26 at 230M supports the [350M-SFT thesis](../training/slm-agentic-tool-calling.md): narrow agentic competence is trainable at sub-1B scale. Ships base + post-trained checkpoints for fine-tuning.

## Relevance to 4 GB VRAM target

- 8B-A1B: under 6 GB quantized on laptops per Liquid; borderline for pure-VRAM residence at 4 GB but the conv+GQA hybrid keeps KV cache small and the 1.5B active path runs well split across CPU/GPU ([llama.cpp -ngl](../runtimes/llama-cpp.md), [KTransformers](../runtimes/ktransformers.md)). CPU-only decode at 146-253 tok/s on high-end laptop silicon makes it the first MoE where skipping the GPU entirely is plausible.
- 230M: irrelevant as a primary coder; strong candidate as a draft model for [speculative decoding](../analysis/spec-decoding-at-4gb.md) or as a fine-tuned tool-call formatter in front of a larger coder.
- Caveat per [SLMQuant](../techniques/slmquant.md): all numbers above are unquantized; verify tool-call conformance at Q4 before adopting.

## See Also

- [LFM2](lfm2.md)
- [Nemotron-3-Nano-4B](nemotron-3-nano.md)
- [Mellum2](mellum2.md)
- [Gemma 4](gemma-4.md)
- [MoE active-parameter architectures](../architectures/moe-active-param.md)
- [BFCL](../benchmarks/bfcl.md)
