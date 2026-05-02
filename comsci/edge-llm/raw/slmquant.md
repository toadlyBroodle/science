# SLMQuant: Benchmarking Small Language Model Quantization for Practical Deployment

**Source:** arXiv:2511.13023 (https://arxiv.org/abs/2511.13023)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Jiacheng Wang, Yejun Zeng, Jinyang Guo, Yuqing Ma, Aishan Liu, Xianglong Liu
**Submitted:** 2025-11-17

## Abstract / extracted content

First systematic benchmark of LLM-style quantization methods (SmoothQuant, OmniQuant, SpinQuant, AWQ, GPTQ family) applied to small language models. Central finding: SLMs and LLMs have fundamentally different quantization sensitivity profiles. Direct transfer of LLM-optimized techniques to SLMs leads to suboptimal results. Authors propose actionable design principles for SLM-tailored compression aimed at low-end edge devices.

## Key claims

- SLMs and LLMs have different quantization bottlenecks; one-size-fits-all transfer fails.
- Architectural and training-dynamics differences (denser activation distributions, different outlier structure, different training-token-to-parameter ratios) drive the divergence.
- The paper proposes SLM-specific design principles rather than a single new method.

## Why this matters for 4 GB edge target

The wiki's target use case is exactly SLMs at extreme quant (Q4 / Q3 / Q2). SLMQuant is the first benchmark to question whether well-known LLM quant methods (AWQ, GPTQ, AQLM) actually preserve SLM accuracy. Cited claims about "70% on SWE-bench at Q4" need this paper as a sanity check.
