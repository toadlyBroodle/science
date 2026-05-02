# Mamba / SSM and Hybrid Architectures

> **Summary:** State-space models (Mamba, Mamba-2, Mamba-3) and SSM-Transformer hybrids (Jamba, Zamba2, Hymba). Linear-time sequence modeling with selective state spaces. The structural insight: SSMs and Transformers are connected via state-space duality (Mamba-2). Hybrids interleave Mamba layers with shared attention layers, getting the throughput of SSM/RNN with the quality of Transformer attention. **For 4 GB VRAM, the hybrid family's small KV cache is the structural advantage.**

**Sources:** [raw/lfm2.md](../../raw/lfm2.md). External: arXiv:2312.00752 (Mamba), arXiv:2405.21060 (Mamba-2), Mamba-3 (ICLR 2026), Zyphra Zamba2 model cards.

---

## Mamba family

- **Mamba (2023):** Linear-time sequence modeling with selective state spaces. Replaces attention with a state-space mechanism that admits a linear recurrence.
- **Mamba-2 (ICML 2024, arXiv:2405.21060, Dao & Gu):** State-space duality (SSD) framework formally connecting SSMs and attention. Mamba-2's core layer is 2-8x faster than Mamba's selective SSM at competitive quality.
- **Mamba-3 (ICLR 2026):** Further architectural refinements. Details not extracted (PDF fetch failed); see ICLR 2026 OpenReview ID HwCvaJOiCj.

## Hybrid models

Pure Mamba/SSM models underperform attention on retrieval-heavy and copy-heavy tasks. Hybrids fix this by interleaving SSM layers (cheap, fast, no KV cache) with a few attention layers (where retrieval matters).

Notable members:
- **Jamba 1.5 (AI21):** Mamba + attention + MoE.
- **Zamba2 (Zyphra, Nov 2024):** Mamba2 backbone + 2 shared attention layers. 1.2B and 7B variants. ~4x throughput of an equal-parameter transformer. Among small models (≤8B), claims to lead in quality and performance per parameter.
- **Hymba (NVIDIA, 2024):** Mamba + attention with parallel processing.
- **[LFM2](../models/lfm2.md) (Liquid AI, Nov 2025):** Gated short convolutions + GQA; different building blocks but same hybrid spirit.

## Why hybrids matter for 4 GB VRAM

KV cache is the dominant VRAM cost beyond 32k context. Mamba-style layers need *no* KV cache (they carry a fixed-size state instead). A hybrid with 2-4 attention layers stores KV for those layers only, reducing cache footprint roughly proportionally.

For the 4 GB target this means: a 7B hybrid at long context can fit where a 7B pure-attention model cannot.

Concrete: Zamba2-7B at Q4 (~4 GB weights) fits the budget; KV cache for its 2 shared attention layers at 32k context is small enough to leave headroom. A pure-attention 7B in the same budget caps out around 8-16k effective context.

## Caveats

- Hybrids trail pure-attention SOTA on rapidly-moving benchmarks; ecosystem (kernels, harness compatibility, quant support) lags.
- Tool-call format conformance not as well measured for hybrids; possible (untested) failure mode.
- Coder-specific hybrids (e.g., a Mamba-Coder) do not yet exist as of early 2026.

## See Also

- [LFM2](../models/lfm2.md)
- [DASH-KV](../techniques/dash-kv.md) (compute-side; complements SSM-style state compression)
- [StructKV](../techniques/structkv.md)
- [MoE active-param efficiency](moe-active-param.md)
