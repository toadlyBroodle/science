# SLMQuant: Benchmarking Small Language Model Quantization

> **Summary:** Wang et al., November 2025 (arXiv:2511.13023). The first systematic benchmark of LLM-style quantization methods (SmoothQuant, OmniQuant, SpinQuant, AWQ, GPTQ family) applied to small language models. **Central finding: SLMs and LLMs have fundamentally different quantization sensitivity. Direct transfer of LLM-optimized techniques to SLMs leads to suboptimal results.**

**Sources:** [raw/slmquant.md](../../raw/slmquant.md), [raw/awq.md](../../raw/awq.md), [raw/gptq.md](../../raw/gptq.md), [raw/aqlm.md](../../raw/aqlm.md)

---

## Why this paper matters for the wiki

The wiki's target use case is small models at extreme quant on a 4 GB device. Most published quant claims ("Q4 with negligible accuracy loss," "AQLM Pareto-optimal under 3 bits") were measured on 7B-70B+ LLMs. SLMQuant is the first to ask whether those results transfer to the 1-4 B SLM regime.

Answer: not cleanly. The paper documents fundamental disparities driven by:

- Architectural differences (denser activation distributions in SLMs).
- Different outlier structure.
- Different training-token-to-parameter ratios.

## Implications

- A claim like "Phi-4-mini at Q4 with negligible accuracy loss" inherits its credibility from LLM-era studies; SLMQuant suggests this needs re-verification at the SLM scale.
- AWQ-style activation-aware quant may behave differently when activations are denser (less obvious salient channel structure).
- Extreme low-bit (AQLM, BitNet-style) is *more* fragile at small scale, not less.

## Methods covered

The abstract references SmoothQuant, OmniQuant, SpinQuant family. Full method list and headline numbers not extracted; see paper for the comparative tables.

## Practical takeaway

When the wiki's analysis pages cite a benchmark figure for a small model at Q4, they should include SLMQuant as a reference for the per-method sensitivity, not just the original quant paper.

## See Also

- [AWQ](awq.md)
- [GPTQ](gptq.md)
- [AQLM](aqlm.md)
- [Quant-vs-capability frontier](../analysis/quant-vs-capability-frontier.md) (pending Phase 3)
