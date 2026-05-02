# 4 GB VRAM Budget Math

> **Summary:** Concrete VRAM accounting for (model × quant × context length) on a 4 GB GPU. **Weights are only one of four contributors**; KV cache, activations, and runtime workspace each take meaningful slices. KV cache often dominates beyond 32k context. This page identifies the actually-feasible (model, quant, ctx) tuples for the 4 GB target.

**Sources:** [models/phi-4-mini.md](../models/phi-4-mini.md), [models/gemma-3.md](../models/gemma-3.md), [models/lfm2.md](../models/lfm2.md), [models/qwen3-coder-next.md](../models/qwen3-coder-next.md), [techniques/kivi.md](../techniques/kivi.md), [techniques/dash-kv.md](../techniques/dash-kv.md), [techniques/structkv.md](../techniques/structkv.md), [techniques/slmquant.md](../techniques/slmquant.md)

---

## The four contributors

```
VRAM_total = weights + kv_cache + activations + runtime_workspace
```

| Contributor | Scaling | Typical fraction at 32k context |
|---|---|---|
| Weights | static; bits-per-param × N | 30-60% |
| KV cache | 2 × layers × seq_len × hidden_dim_kv × bits | 25-50% (more for non-GQA) |
| Activations | per-batch × seq_len × hidden_dim × layers (peaks during prefill) | 10-25% |
| Runtime workspace | CUDA context, kernels, temp buffers | 200-500 MB fixed |

**The 4 GB ceiling is not the weights ceiling.** A "fits in 4 GB at Q4" claim must account for all four.

## Weights footprint table

For dense candidates at common quant levels:

| Model | Params | FP16 | Q8 | Q5_K_M | Q4_K_M | Q3_K_M | IQ2_XS |
|---|---|---|---|---|---|---|---|
| OPT-350M | 0.35 B | 0.7 GB | 0.4 | 0.25 | 0.2 | 0.15 | 0.1 |
| DS-R1-Distill-Qwen-1.5B | 1.5 B | 3.0 | 1.5 | 1.1 | 0.9 | 0.7 | 0.5 |
| LFM2-1.2B | 1.2 B | 2.4 | 1.2 | 0.9 | 0.7 | 0.55 | 0.4 |
| Zamba2-1.2B | 1.2 B | 2.4 | 1.2 | 0.9 | 0.7 | 0.55 | 0.4 |
| xLAM-2-3B-r | 3 B | 6.0 | 3.0 | 2.2 | 1.8 | 1.4 | 1.0 |
| Phi-4-mini | 3.8 B | 7.6 | 3.8 | 2.8 | 2.3 | 1.7 | 1.2 |
| Gemma 3-4B | 4 B | 8.0 | 4.0 | 3.0 | 2.4 | 1.8 | 1.3 |
| LFM2-2.6B | 2.6 B | 5.2 | 2.6 | 1.95 | 1.6 | 1.2 | 0.85 |
| Gemma 4-E4B | ~4 B | 8.0 | 4.0 | 3.0 | 2.4 | 1.8 | 1.3 |
| LFM2-8.3B | 8.3 B | 16.6 | 8.3 | 6.2 | 4.5 | 3.5 | 2.5 |
| DS-R1-Distill-Qwen-7B | 7 B | 14.0 | 7.0 | 5.2 | 4.2 | 3.2 | 2.3 |
| Zamba2-7B | 7 B | 14.0 | 7.0 | 5.2 | 4.2 | 3.2 | 2.3 |

(Q4_K_M ≈ 4.7 effective bits/param; Q3_K_M ≈ 3.6; IQ2_XS ≈ 2.5.)

For active-parameter MoE on 4 GB, only the active set fits; total weights need offload.

## KV cache footprint per token

```
KV_per_token_bytes = 2 (K and V) × num_layers × hidden_dim_kv × bytes_per_element
```

GQA reduces `hidden_dim_kv` by the GQA ratio (typically 4-8x). Modern small models all use GQA.

| Model | Layers | hidden_dim_kv (FP16) | KV per token (FP16) | At 32k ctx (FP16) |
|---|---|---|---|---|
| Phi-4-mini (GQA-8) | 32 | ~512 | ~64 KB | ~2.0 GB |
| Gemma 3-4B (GQA-4) | 26 | ~640 | ~67 KB | ~2.1 GB |
| Zamba2-1.2B (Mamba+2-attn) | 2 attn × ~1024 | ~4 KB/token (state-only for SSM) | ~10 KB | ~0.3 GB |
| LFM2-1.2B (hybrid) | mostly conv | ~12 KB/token | ~12 KB | ~0.4 GB |

These are **order-of-magnitude estimates**; exact KV size depends on per-model head/dim choices not always in the abstract.

KV-cache quantization compounds:
- [KIVI](../techniques/kivi.md) at 2-bit: divides KV by ~ 4x.
- [StructKV](../techniques/structkv.md) at 50% retention: divides token count by 2.
- [DASH-KV](../techniques/dash-kv.md): doesn't reduce KV memory but eliminates O(N²) attention compute.

Stacking KIVI + StructKV at modest settings → 6-8x KV reduction at acceptable quality (still subject to [SLMQuant](../techniques/slmquant.md) caveat at SLM scale).

## Activations + workspace

- Activations during prefill scale as O(batch × seq_len × hidden × layers); for batch=1 and 32k context on a 4B model, peak activation memory is 200-500 MB depending on attention implementation.
- Flash-attention-style fused kernels eliminate the O(seq²) intermediate; without them, activation memory explodes.
- Runtime workspace: ~ 200-500 MB fixed (CUDA context, kernel cache).

Budget: reserve ~ 700 MB for activations + workspace at 32k context.

## Feasible (model, quant, ctx) tuples for 4 GB

After subtracting 700 MB for activations + workspace, **3.3 GB available for weights + KV cache**.

| Tuple | Weights | KV (no compression) | KV (KIVI 2-bit + StructKV 50%) | Total | Feasible? |
|---|---|---|---|---|---|
| Phi-4-mini Q4 + 8k FP16 KV | 2.3 | 0.5 | 0.5 | 2.8 | ✅ |
| Phi-4-mini Q4 + 32k FP16 KV | 2.3 | 2.0 | 2.0 | 4.3 | ❌ over |
| Phi-4-mini Q4 + 32k compressed KV | 2.3 | 2.0 | 0.3 | 2.6 | ✅ |
| Gemma 3-4B Q4 + 32k compressed KV | 2.4 | 2.1 | 0.3 | 2.7 | ✅ |
| LFM2-2.6B Q4 + 32k FP16 KV (hybrid arch) | 1.6 | 0.4 | 0.4 | 2.0 | ✅ |
| Zamba2-1.2B Q4 + 64k FP16 KV (hybrid) | 0.7 | 0.6 | 0.6 | 1.3 | ✅ comfortable |
| DS-R1-Distill-Qwen-7B Q4 + any KV | 4.2 | n/a | n/a | >4.2 | ❌ over (weights alone) |
| Gemma 4-E4B Q4 + 32k compressed KV | 2.4 | ~2.1 | 0.3 | 2.7 | ✅ |
| Qwen3-Coder-Next active 3B Q4 (offload total) | 1.8 active | small | small | < 4 GB GPU | ✅ via offload only |

## Headline takeaways

1. **The 4 GB ceiling is roughly a 4 B-parameter ceiling at Q4** with compressed KV cache, *if* the model uses aggressive GQA.
2. **Hybrid SSM/attention models (LFM2, Zamba2) get a structural win** because most layers carry no KV cache. They support longer effective context inside the same 4 GB.
3. **Q3 / IQ2 are tempting but suspicious at SLM scale** ([SLMQuant](../techniques/slmquant.md)). Verify with task-specific evals before deploying.
4. **MoE (Qwen3-Coder-Next, Gemma-4-26B-A4B) is feasible only via offload.** Active 3-4 B fits; total 26-80 B does not. The [DALI](../runtimes/dali-moe.md) + [FlashMoE](../runtimes/flashmoe.md) + [MoE-Spec](../techniques/moe-spec.md) stack is the deployment path.
5. **DS-R1-Distill-Qwen-7B is just over budget** at Q4. Either downsize to the 1.5B distill, switch to Phi-4-mini-reasoning (claimed equivalent at 3.8B), or accept partial CPU offload.

## See Also

- [Quant-vs-capability frontier](quant-vs-capability-frontier.md)
- [Spec decoding at 4 GB](spec-decoding-at-4gb.md)
- [Runtime comparison](runtime-comparison.md)
