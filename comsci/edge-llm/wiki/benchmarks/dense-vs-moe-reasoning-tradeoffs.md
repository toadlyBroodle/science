# Gemma 4 / Phi-4 / Qwen3: Dense vs MoE Reasoning Tradeoffs (Manik & Wang, April 2026)

> **Summary:** Controlled empirical benchmark of 7 instruction-tuned reasoning models across dense and MoE designs. 8,400 evaluations across ARC-Challenge, GSM8K, Math L1-3, TruthfulQA MC1. **Best overall: Gemma-4-E4B with few-shot CoT, weighted accuracy 0.675, mean VRAM 14.9 GB. Sparse activation alone does not guarantee the best practical operating point.**

**Sources:** [raw/gemma-phi-qwen-tradeoffs.md](../../raw/gemma-phi-qwen-tradeoffs.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md), [raw/phi-4-mini.md](../../raw/phi-4-mini.md), [raw/gemma-3.md](../../raw/gemma-3.md)

---

## Models tested

- **Gemma-4:** E2B, E4B, 26B-A4B (April 2026 family)
- **Phi-4:** mini-reasoning, reasoning
- **Qwen3:** 8B, 30B-A3B

## Headline results

| Model | Weighted accuracy | Mean VRAM |
|---|---|---|
| **Gemma-4-E4B (FS-CoT)** | **0.675** | **14.9 GB** |
| Gemma-4-26B-A4B | 0.663 | 48.1 GB |

(Other variants' numbers were not extracted from the abstract.)

Per-task observations:

- Gemma dominated mathematical reasoning.
- Phi excelled on TruthfulQA.
- GSM8K showed dramatic prompt sensitivity (large delta from prompt template alone).

## The central finding

Sparse activation is not free quality. The 26B-A4B MoE (4 B active) underperformed the 4B dense in this benchmark suite at much higher VRAM cost. Architecture, prompt strategy, and task composition jointly determine the operating point; "MoE = better quality-efficiency tradeoff" is not automatic.

This is a useful corrective for the wiki's analysis: the [Qwen3-Coder-Next](../models/qwen3-coder-next.md) 80B/3B-active MoE is impressive *for coding agents trained with environment feedback*, but the same MoE→dense argument cannot be made automatically for general reasoning.

## Caveats

- Reasoning benchmarks (ARC, GSM8K, MATH, TruthfulQA) are not coding benchmarks. SWE-Bench Verified, Aider polyglot, and BFCL v3 may invert the dense-vs-MoE story.
- Single-paper, single-team result; replication needed.
- Mean VRAM numbers are FP16/BF16; quantized footprints differ.

## Relevance to 4 GB VRAM target

The leader (Gemma-4-E4B at 14.9 GB BF16) is over-budget at full precision. At Q4 ≈ 4 GB weights it fits, with KV cache pressure; the wiki's [4 GB budget math](../analysis/four-gb-budget-math.md) page (pending) needs to verify Q4 quality on this exact model. The smaller Gemma-4-E2B and Phi-4-mini-reasoning are the in-budget candidates the paper covers.

## See Also

- [Gemma 3](../models/gemma-3.md)
- [Phi-4-Mini](../models/phi-4-mini.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
- [SLMQuant](../techniques/slmquant.md)
