# AQLM (Additive Quantization for LLMs)

> **Summary:** Egiazarian et al., ICML 2024 (arXiv:2401.06118). Multi-codebook quantization adapted from information retrieval. Two innovations: input-adaptive learned additive quantization per weight matrix, and joint codebook optimization across transformer blocks. Authors claim Pareto optimality below 3 bits per parameter, with notable strength in the extreme 2-bit regime.

**Sources:** [raw/aqlm.md](../../raw/aqlm.md), [raw/slmquant.md](../../raw/slmquant.md)

---

## When to consider AQLM over AWQ/GPTQ

- At 4-bit and above: AWQ/GPTQ are simpler and equivalent in quality.
- At 3-bit: AQLM starts winning.
- At 2-bit: AQLM is materially better than AWQ/GPTQ (the latter degrade sharply).

## Position vs BitNet

[BitNet 1.58b](./) (covered as a stub elsewhere) achieves ternary weights via QAT (quantization-aware training, requires retraining). AQLM is post-training: applied to an existing FP16/BF16 checkpoint. The trade is recipe complexity vs flexibility.

## Caveat at SLM scale

[SLMQuant](slmquant.md) flags that extreme low-bit (2-3 bit) compression is *more* fragile at SLM scale, not less. AQLM's Pareto-optimality claim was validated on 7B+ LLMs. For 1-4 B targets, a controlled re-evaluation is needed before relying on 2-bit AQLM.

## See Also

- [AWQ](awq.md)
- [GPTQ](gptq.md)
- [SLMQuant](slmquant.md)
