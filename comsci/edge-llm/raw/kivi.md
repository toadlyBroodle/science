# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

**Source:** arXiv:2402.02750 (https://arxiv.org/abs/2402.02750)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu
**Venue:** ICML 2024
**Submitted:** 2024-02-05; revised 2024-07-25

## Abstract / extracted content

KV-cache quantization scheme based on empirical analysis of element distributions: key cache should be quantized per-channel, value cache per-token. Tuning-free (no calibration / hyperparameter search). Asymmetric 2-bit quantization. Targets the memory bottleneck of long-context LLM serving where KV cache often exceeds weight memory.

## Key claims

- Asymmetric strategy: per-channel for K, per-token for V (justified by empirical distribution analysis).
- Works across Llama, Falcon, Mistral families without tuning.

## Headline numbers

- 2.6x reduction in peak memory (model weights + KV).
- Up to 4x larger batch size.
- 2.35x to 3.47x throughput improvement on real workloads.
