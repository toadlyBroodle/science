# DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs

**Source:** arXiv:2602.03495 (https://arxiv.org/abs/2602.03495)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Zeyu Zhu, Gang Li, Peisong Wang, Zitao Mo, Minnan Pei, Zhuoran Song, Xiaoyao Liang, Jian Cheng
**Submitted:** 2026-02-03

## Abstract / extracted content

MoE expert offloading framework targeting local PCs (consumer GPU + host RAM). Addresses three concrete inefficiencies of prior offloading systems:

1. Static expert assignment causes CPU-GPU load imbalance. DALI models expert-to-device assignment as a 0-1 integer optimization solved via a Greedy Assignment strategy at runtime.
2. Inaccurate expert prefetch predictions. DALI uses inter-layer residual signals for accurate high-workload expert prediction.
3. Suboptimal GPU cache policies. DALI exploits temporal correlation in expert activations via workload-aware cache replacement.

## Key claims

- Significant speedups in prefill and decoding vs prior MoE offloading frameworks.
- Specific numbers not extracted from abstract.

## Relevance to 4 GB edge target

For an 80B-total / 3B-active MoE like Qwen3-Coder-Next, only the active experts plus shared parameters need to be on the GPU. DALI targets exactly the local-PC consumer-GPU + host-RAM setup that the wiki's 4 GB-VRAM target implies. This is the most directly relevant single 2026 paper for "how do I run Qwen3-Coder-Next on 4 GB."
