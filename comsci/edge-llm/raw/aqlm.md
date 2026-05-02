# Extreme Compression of LLMs via Additive Quantization (AQLM)

**Source:** arXiv:2401.06118 (https://arxiv.org/abs/2401.06118)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
**Venue:** ICML 2024
**Submitted:** 2024-01-11; final 2024-09-11

## Abstract / extracted content

AQLM generalizes additive quantization (a multi-codebook scheme from information retrieval) to LLM weight compression. Two innovations: (1) learned input-adaptive additive quantization per weight matrix, (2) joint codebook optimization across transformer blocks. Targets the extreme regime: 2-3 bits per parameter. Authors claim Pareto optimality below 3 bits.

## Key claims

- Pareto optimal in accuracy vs size below 3 bits per parameter.
- Significantly outperforms prior schemes at 2-bit.
- GPU and CPU implementations; matches or exceeds optimized FP16 throughput at smaller memory.

## Headline numbers

- Quantitative benchmarks not extracted from abstract page.
- See paper for per-model perplexity tables.

## Key contributions

1. Additive multi-codebook quantization adapted to LLMs.
2. Input-adaptive learned quantization.
3. Cross-block codebook joint optimization.
4. GPU + CPU kernels for efficient generation.
