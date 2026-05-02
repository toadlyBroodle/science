# DeepSeek-R1 and Distill Family

> **Summary:** DeepSeek-AI, January 2025; published in Nature 645 (2025). Demonstrates that LLM reasoning can be developed via pure RL, removing dependence on human-labeled reasoning trajectories. Reasoning capability transfers via distillation to smaller dense bases (Qwen, Llama). The 1.5B and 7B distills are the relevant edge-LLM targets.

**Sources:** [raw/deepseek-r1.md](../../raw/deepseek-r1.md), [raw/phi-4-mini.md](../../raw/phi-4-mini.md), [raw/mol-distillation.md](../../raw/mol-distillation.md)

---

## Core paper

arXiv:2501.12948; Nature 645, 633-638 (2025). DeepSeek-AI as primary author with 200+ contributors.

The headline contribution is methodological: pure RL (no SFT on human traces) on a reasoning reward signal produces emergent self-reflection and verification behaviors, beating supervised baselines on math, coding, STEM. Self-correction and backtracking emerge without explicit annotation.

## Distill family (the practical edge artifacts)

DeepSeek released distillations of R1 reasoning into well-known dense bases:

- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Llama-70B

The distill recipe uses R1-generated reasoning traces as SFT data on the dense base, no RL. Subsequent 2026 work ([Mixture-of-Layers distillation](../training/mol-distillation.md), HEAL) explores improvements to the distillation step itself, motivated by reports that the original recipe leaves capability on the table.

## Edge-relevant variants

- **DeepSeek-R1-Distill-Qwen-1.5B**: at Q4 ~ 0.9 GB. Fits comfortably in 4 GB with room for KV cache and a draft head. Lower reasoning ceiling.
- **DeepSeek-R1-Distill-Qwen-7B**: at Q4 ~ 4 GB weights. Tight; KV cache at 32K context likely pushes past the ceiling; requires offload.

[Phi-4-mini-reasoning](phi-4-mini.md) (3.8 B) reportedly matches or exceeds R1-Distill-Qwen-7B and R1-Distill-Llama-8B at smaller weight footprint, making it the more attractive edge candidate for reasoning-heavy workloads.

## Caveats

- The fetched arXiv abstract did not include benchmark numbers (AIME, MATH, GPQA, Codeforces). See the Nature paper or Hugging Face model cards.
- Distillation quality of the R1-Distill family is suspected to underperform what's possible: see [MoL distillation](../training/mol-distillation.md) and the HEAL paper for 2026 improvements.

## See Also

- [Phi-4-Mini](phi-4-mini.md); reasoning variant claimed to match/exceed R1-Distill-7B/8B
- [Mixture-of-Layers distillation](../training/mol-distillation.md); 2026 improvement to CoT distillation
- [SLMQuant](../techniques/slmquant.md); quant sensitivity for SLM-class targets
