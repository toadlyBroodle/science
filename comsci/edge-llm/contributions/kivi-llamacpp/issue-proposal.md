# Proposal: opt-in asymmetric 4-bit KV-cache quantization (KIVI-style)

## Summary

Add two new opt-in KV-cache quantization types, `q4_asym_k` and `q4_asym_v`, exposing the KIVI scheme (Liu et al., ICML 2024, [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)) to `--ctk` / `--ctv`: keys quantized per-channel, values quantized per-token. The motivating use case is long-context inference on small GPUs, where KV cache dominates VRAM past ~16k tokens and existing symmetric `q4_0` / `q4_K` KV cache shows measurable quality degradation that the asymmetric scheme reclaims.

Posting this as a proposal before opening a PR to confirm the maintainers want this work and to align on the four design questions at the bottom.

## Motivation

For Llama-3.1-8B at 32k context, KV cache in `f16` is ~4 GB, roughly the size of the Q4 weights. The two existing opt-in KV types (`q8_0`, `q4_0`) cut this 2x and 4x respectively, but symmetric `q4_0` shows non-trivial perplexity and long-context retrieval degradation on small models (1-8B class).

KIVI's empirical finding is that key and value activations have different outlier structure:

- **Keys** have channel-level outliers: a small number of hidden-dim channels carry disproportionate magnitude across nearly all tokens. Per-token quantization forces a large shared scale, crushing precision on every non-outlier channel. Per-channel quantization preserves the structure.
- **Values** have token-level outliers: specific tokens (not channels) carry outsized signal. Per-channel is wrong for V; per-token is the natural axis.

The paper reports 2.6x peak-memory reduction, up to 4x larger batch sizes, and 2.35-3.47x throughput improvement at 7B+ scale, tuning-free, across Llama, Falcon, and Mistral. The 4-bit variant (not the headline 2-bit) is the merge-conservative target here.

## Proposed scope (MVP PR)

Opt-in, no default changes:

```bash
./llama-cli -m model.gguf --ctk q4_asym --ctv q4_asym -c 32768 ...
```

What lands in PR #1:

- `GGML_TYPE_Q4_ASYM_K` and `GGML_TYPE_Q4_ASYM_V` enum + trait-table entries. Block size matches the existing Q4_K family for SIMD reuse; the only difference vs Q4_K is the axis along which scales are computed.
- Reference C quantize / dequantize in `ggml/src/ggml-quants.c`.
- `tests/test-quantize-fns.cpp` round-trip coverage and a synthetic channel-outlier / token-outlier test demonstrating asymmetric preserves structure that symmetric Q4_0 destroys.
- KV-cache wiring in `src/llama-kv-cache.cpp` and the `--ctk` / `--ctv` flag plumbing in `common/`.
- Auto-disable FlashAttention with a one-line stderr notice when either asymmetric type is selected, since the fused kernel cannot consume per-channel-K scales without a kernel rewrite (see Q3 below).
- `examples/kv-cache-quality/` script for long-context (NIAH-style) retrieval validation.

What is **not** in PR #1 (filed as labeled follow-up PRs):

- 2-bit asymmetric (KIVI's headline). Gated on getting SLM-scale quality data first.
- CUDA kernels. PR #1 is CPU only; users on GPU run `-ngl 0` for the new types or keep symmetric Q4_K. CUDA dequant is PR #2.
- Metal / ROCm / Vulkan. Sequenced after CUDA.
- FlashAttention with per-channel-K scales (the real perf unlock). Requires a kernel rewrite; deferred until the dequant-then-attend CPU path is in master and there is measured demand.

## Preliminary numbers (will be replaced by measured results before any PR)

Reproducing KIVI's published 4-bit results on Llama-2-7B is the first step before opening a PR. The validation matrix the PR description will carry:

| Model | Context | KV type | PPL (Wikitext-2) | NIAH retrieval | KV bytes |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct | 4k / 8k / 16k / 32k | f16 / q8_0 / q4_0 / q4_asym | TBD | TBD | TBD |
| Llama-3.1-8B-Instruct | 4k / 8k / 16k / 32k | f16 / q8_0 / q4_0 / q4_asym | TBD | TBD | TBD |
| Gemma-3-4B-it | 4k / 8k / 16k / 32k | f16 / q8_0 / q4_0 / q4_asym | TBD | TBD | TBD |

Acceptance gate for opening the PR: perplexity within +1% of `f16`, NIAH within -2 absolute points. If 4-bit misses on the small models, the PR rescopes to 5-bit asymmetric before maintainer review starts.

## Prior art in the tree

- `Q4_K` / `Q8_0` KV cache (`--ctk` / `--ctv`): symmetric, block-quantized, single axis. Closest existing pattern; this proposal extends rather than displaces.
- EAGLE-3 draft-head support (recent): precedent for landing 2024-vintage inference research as opt-in additions.
- Prior KV-cache quantization discussions: happy to be pointed at the most relevant threads to cite directly.

## Open questions for maintainers

1. **Type slot allocation.** Two new `GGML_TYPE_*` enum values are needed. Are slots `NN` and `NN+1` (next contiguous after the last allocated) the preferred placement, or do you reserve slot ranges by family?

2. **Flag naming.** `--ctk q4_asym` is the proposed string. Alternatives: `q4_asym_k` (explicit), `kivi_q4` (named after the paper), `q4_pc` (per-channel). Preference?

3. **FlashAttention behavior when asymmetric types are selected.** Three options:
   - **Auto-disable with stderr notice** (proposed): user gets the new KV type, FA silently turns off, one-line warning. Lowest friction.
   - **Error and require explicit `--no-flash-attn`**: more verbose but safer; user opts in to the perf trade.
   - **Require `--no-flash-attn` to even parse the asymmetric flag**: strictest.
2 or 3 are fine if you prefer explicitness over ergonomics.

4. **CPU-only PR #1 acceptable?** The implementation plan ships CPU first, CUDA as PR #2, other backends sequenced after. Confirming this is the preferred shape rather than expecting a CUDA stub in PR #1, which would balloon scope.

Will not start any C++ work until there is a signal on these. If the answer is "we already considered this and don't want it," that's also a clear answer and saves the effort.

## References

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750). Reference implementation: [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI).
