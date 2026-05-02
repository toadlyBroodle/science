# AWQ (Activation-aware Weight Quantization)

> **Summary:** Lin et al., MLSys 2024 (Best Paper). Activation-aware weight-only quantization. Identifies salient channels via activation statistics, scales them before round-to-nearest 4-bit / 3-bit quantization. Generalizes well across language, code, math, and multimodal. TinyChat runtime: >3x speedup over FP16. Foundational reference; superseded by 2025-2026 methods at scale, but still the practitioner default for 4-bit weight-only.

**Sources:** [raw/awq.md](../../raw/awq.md), [raw/slmquant.md](../../raw/slmquant.md)

---

## Method (one paragraph)

Identify the ~1% of weight channels with the largest mean activation magnitude. Scale those channels (and inversely scale the corresponding activations) before quantizing all weights uniformly to low bits. Avoids mixed-precision storage; preserves the salient information without backprop.

## Why it's still the practitioner default

- Hardware-friendly uniform-bitwidth storage.
- Per-tensor scaling is cheap to apply.
- Available in llama.cpp, ExLlamaV2, vLLM, AutoAWQ.
- 4-bit AWQ on a 7B model + minimal calibration data ≈ near-FP16 quality at ~ ¼ the weights.

## Caveats at SLM scale

[SLMQuant](slmquant.md) (Nov 2025) argues LLM-era quant claims do not transfer cleanly to small models. AWQ at 4-bit on a 1-4 B model may degrade more than the original paper's tables suggest. Verify with task-specific evals on the actual model and quant scheme before locking it in.

## See Also

- [GPTQ](gptq.md)
- [AQLM](aqlm.md)
- [SLMQuant](slmquant.md)
