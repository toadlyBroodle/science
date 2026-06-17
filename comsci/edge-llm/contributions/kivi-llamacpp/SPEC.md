# KIVI-in-llama.cpp MVP SPEC

> Canonical spec for landing an MVP of KIVI-style asymmetric KV-cache quantization in llama.cpp. Scoped for "most likely to be merged" rather than "captures the full breakthrough." Every step reads this file before deciding what to do, and updates it (along with `TODO.md`) in the same commit as any code change.

## Status

**ENGAGED, 2026-05-17.** Pre-PR discovery surfaced significant in-flight work in the same surface that supersedes the originally-drafted MVP:

- [ggml-org/llama.cpp#21089](https://github.com/ggml-org/llama.cpp/pull/21089) (TurboQuant TBQ3/TBQ4 CPU KV cache, OPEN, last activity 2026-04-25): already implements the structural plumbing (opt-in CPU-only KV type, new `ggml_type` slots, narrow first-PR shape) that this spec's Phases 1-3 propose. A competing KIVI-typed PR would duplicate it.
- [#21551](https://github.com/ggml-org/llama.cpp/pull/21551) (existing-quant KV asymmetric exploration, OPEN draft): finds `q3_K`-K + `q2_K`-V is a sweet spot using existing quant types, no new `ggml_type` needed.
- [#21591](https://github.com/ggml-org/llama.cpp/issues/21591) (NexusQuant findings, CLOSED): K3 → K4 buys +0.06pp PPL (softmax error floor at ~3 bits). Confirms this spec's 4-bit-K target is over-allocated; K3 is the sweet spot.
- Recent upstream KV rotation work partially obviates KIVI's per-channel-K insight, since rotation removes the channel-outlier structure that motivates it.

**Action taken (2026-05-16):** posted comment on #21089 surfacing two specific contributions from #21591 not yet cited in the thread: (1) the K3 softmax-error-floor curve and (2) the layer-position fp16 boundary protection for architecture-sensitive cases (Qwen2.5-7B). Comment framed as data, not direction.

**Engagement (2026-05-17, 8.5 hours after posting):** @jagmarques (NexusQuant author) replied directly with a new datapoint: on Qwen2.5-7B-Instruct, per-head fp16 masking of the lowest-2% KV-heads matches K3V2 boundary-protect on retrieval, at **2.28 vs 3.93 average K-bits/element**. Explicitly framed as "not a quant type, so orthogonal to this PR, but a relevant knob for the K-bit/arch-sensitivity question." Implication: per-head protection dominates per-layer protection at lower average bit cost, but the mechanism is runtime-architectural (not a new ggml type) and so does not affect #21089's scope directly.

**Decision tree, updated post-engagement:**

- **Branch A — #21089 merges with asymmetric K/V support.** Most likely outcome given jagmarques's reply keeps the thread alive on the asymmetric-K-bit-vs-architecture-sensitivity sub-question. Pivot Phase 0 PyTorch validation into asymmetric K3V2 measurements on Qwen3-4B / Llama-3.1-8B / Gemma-3-4B and contribute as evidence in the asymmetric config selection.
- **Branch B — #21089 merges as symmetric-only.** Revive this spec as a follow-up PR adding asymmetric K/V on top of the new TBQ types. Bitcount target shifts from Q4 to Q3 per #21591.
- **Branch C — Per-head fp16 KV-head masking as a fresh contribution surface.** New, surfaced by jagmarques's reply. Orthogonal to #21089 (not a quant type; runtime-architectural). Status in upstream llama.cpp: unknown; needs grep of `llama-kv-cache.cpp` and the `n_head_kv` plumbing. If absent, could be a clean small PR adding per-head fp16 override via `--kv-head-precision`-style flag plus optional outlier-head identification heuristic. Skill match is good (no kernel rewrite; data-structure-level change). Cite jagmarques + #21591.
- **Branch D — #21089 dies.** Revisit Path 5 pairings entirely (Saguaro/SSD in llama.cpp is the next-best candidate).

Branches A and C are the live ones. Branch C is now the most interesting because it's a fresh PR surface with no in-flight competitor, the upstream mechanism (per-head precision override) is structurally simpler than a new quant type, and the citation path is clean (jagmarques's own measurement + #21591). Worth a focused grep before committing to it.

The structural Phase 0-5 plan below remains intact as a reusable template; bitcount, paper citation, and headline framing all change per the branch.

---

## Goal

Land an opt-in 4-bit asymmetric KV-cache quantization path in upstream llama.cpp: keys quantized per-channel, values quantized per-token, exposed as new `--ctk q4_asym` and `--ctv q4_asym` types. CPU backend only on the first PR; CUDA dequant path as an immediate follow-up; FlashAttention auto-disabled when these types are active. The goal is acceptance into upstream, not feature-completeness — so the contribution is narrow, opt-in, benchmark-validated at SLM scale, and changes no defaults.

The downstream value (for the wider edge-llm thesis) is that 4-bit asymmetric KV at 32k context cuts KV memory ~3.5x vs Q8_0 and ~2x vs Q4_0-symmetric while preserving long-context retrieval quality. On a 4 GB GPU running a 4B coder, this is the difference between ~12k and ~32k effective context for an agentic coding harness. The MVP does not capture the full 2-bit win from the KIVI paper; that is a follow-up after the merge.

## Architecture / stack (one-liner each)

- Upstream target: `ggerganov/llama.cpp`, `master` branch.
- Local working tree: `/tmp/llama.cpp` (clone), `origin` = fork (user-supplied), `upstream` = ggerganov.
- Reference impl: KIVI authors' PyTorch repo (`jy-yuan/KIVI`), used only for offline parity checks.
- New ggml types: `GGML_TYPE_Q4_ASYM_K` (per-channel scales), `GGML_TYPE_Q4_ASYM_V` (per-token scales). Block size matches existing Q4_K family for CPU SIMD reuse.
- KV-cache wiring: `src/llama-kv-cache.cpp` + `src/llama-kv-cache.h`; flag plumbing in `common/arg.cpp`.
- Tests: `tests/test-quantize-fns.cpp` for round-trip and dequant accuracy; `tests/test-quantize-perf.cpp` for CPU throughput baseline. Long-context quality validation via `examples/perplexity` and a RULER-subset retrieval script.
- Validation targets: Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-4B-it. Contexts 4k / 8k / 16k / 32k. Compared against fp16 KV reference and Q8_0 / Q4_0 symmetric KV baselines.

## Phases

### Phase 0: SLM-scale validation before any C++ work

The wiki's KIVI caveat (`wiki/techniques/kivi.md` line 27-29) is that 2-bit asymmetric has not been verified at 1-4B scale and may degrade quality more than the 7B+ headline numbers suggest. The Q4 variant has no published SLM validation at all. Before writing llama.cpp code, replicate the asymmetric scheme in PyTorch on the actual MVP targets and confirm parity. If it doesn't hold, fix scope (drop to 8-bit asymmetric, or expand to 5-bit) before opening the PR, not after.

- [ ] 0.1 [medium] Fork `jy-yuan/KIVI`, replicate the per-channel-K / per-token-V quantization scheme at 4-bit (paper publishes 2-bit and 4-bit; verify the 4-bit path runs and matches reported numbers on Llama-2-7B).
- [ ] 0.2 [medium] Adapt for GQA. KIVI was published on MHA models; modern targets use grouped-query attention (Qwen3, Llama-3, Gemma-3). Derive whether per-channel-K statistics survive head-group sharing, or whether the outlier channels need re-identification per K head. Document the finding.
- [ ] 0.3 [hard] Run Q4 asymmetric on Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-4B-it. Measure perplexity on Wikitext-2 and one long-context retrieval task (RULER NIAH at 8k / 16k / 32k) vs fp16 KV reference and symmetric Q4 KV baseline. Acceptance gate: perplexity within +1% of fp16; retrieval within -2 absolute points of fp16. If miss, escalate to 5-bit asymmetric and re-test.
- [ ] 0.4 [easy] Write a one-page validation report (`validation-report.md`) with the three model × four context × three KV-type result matrix. This becomes part of the PR description.

### Phase 1: ggml type definitions

Add the two new types as a non-default extension of the Q4_K block family. Reuse the existing 32-element block layout for SIMD path compatibility; the only difference vs Q4_K is the scale-grouping axis: `Q4_ASYM_K` groups scales by channel (hidden dim), `Q4_ASYM_V` groups scales by token (position). No new SIMD intrinsics needed; the existing Q4_K dequant path can be parametrized over the axis.

- [ ] 1.1 [medium] Add `GGML_TYPE_Q4_ASYM_K = NN` and `GGML_TYPE_Q4_ASYM_V = NN+1` to `ggml/include/ggml.h` enum; reserve the next two unused slots. Mirror the trait table entries in `ggml/src/ggml.c` (block size, type size, dequant function pointer, `from_float` function pointer, vec-dot pointer set to nullptr for now since attention does dequant-then-attend on the CPU path).
- [ ] 1.2 [medium] Implement `quantize_row_q4_asym_k_ref` and `quantize_row_q4_asym_v_ref` (reference C, no SIMD) in `ggml/src/ggml-quants.c`. Both produce the same physical block layout; the difference is the axis along which scales are computed.
- [ ] 1.3 [medium] Implement `dequantize_row_q4_asym_k` and `dequantize_row_q4_asym_v` in the same file.
- [ ] 1.4 [easy] Register both types with `ggml_get_type_traits`; ensure `ggml_type_name`, `ggml_type_size`, `ggml_blck_size` resolve correctly.

### Phase 2: CPU implementation and unit tests

Round-trip correctness and dequant accuracy must be testable in isolation before any KV-cache wiring. Follow the existing `tests/test-quantize-fns.cpp` pattern.

- [ ] 2.1 [easy] Add `Q4_ASYM_K` and `Q4_ASYM_V` to the quantize-fns test enumeration. Verify round-trip error stays within the existing tolerance band for Q4_K.
- [ ] 2.2 [medium] Add a synthetic-tensor test that constructs a known channel-outlier pattern (for K) and a known token-outlier pattern (for V), confirms that asymmetric quant preserves it while symmetric Q4_0 does not. This is the load-bearing test for the maintainer-facing argument.
- [ ] 2.3 [easy] CPU perf baseline in `tests/test-quantize-perf.cpp`. Should be within 10% of Q4_K on quantize + dequantize throughput; the asymmetric variant only changes the scale-grouping loop, not the inner SIMD path.

### Phase 3: KV-cache wiring and CLI flags

Plumb the new types through `llama-kv-cache.cpp` and the `--ctk` / `--ctv` CLI flags. Auto-disable FlashAttention when either new type is selected (the fused-attention kernel cannot consume per-channel-K scales without a kernel rewrite, which is explicitly out of scope for the MVP). Emit a clear stderr notice when this fallback fires.

- [ ] 3.1 [medium] Extend `common::kv_cache_type_from_str` (in `common/common.cpp`) to parse `q4_asym` for K and V. Add `q4_asym` to the documented enum in `--ctk` / `--ctv` help text.
- [ ] 3.2 [hard] In `src/llama-kv-cache.cpp`, route the K and V cache allocation to the new types when configured. Verify the cache lifecycle (allocate, fill, evict, reset) round-trips correctly with mixed batches.
- [ ] 3.3 [medium] In `src/llama-context.cpp` (or wherever FA selection happens — confirm during impl), detect `Q4_ASYM_*` cache types and force-disable FA with a one-line stderr notice: `kv-cache: q4_asym selected; flash-attention disabled (no fused kernel; CPU path used)`. Do not error; just inform.
- [ ] 3.4 [easy] Add `examples/main` smoke check: load Qwen3-4B-Instruct with `--ctk q4_asym --ctv q4_asym -c 8192`, generate 256 tokens, verify no crashes and coherent output.

### Phase 4: Long-context quality validation in-tree

Replicate Phase 0's quality validation but using `examples/perplexity` and a RULER-style retrieval script committed under `examples/`. The validation script must run on a maintainer's machine without external dependencies beyond a GGUF model file and a small text corpus. Output is the numbers that go into the PR description.

- [ ] 4.1 [medium] Run `examples/perplexity -m qwen3-4b-instruct.Q4_K_M.gguf -f wiki.test.raw --ctk q4_asym --ctv q4_asym -c 4096` and the symmetric baselines (`q4_0`, `q8_0`, `f16`). Confirm Phase 0's PyTorch result reproduces in-engine within noise.
- [ ] 4.2 [medium] Build a minimal NIAH (needle-in-a-haystack) retrieval script under `examples/kv-cache-quality/` that runs at 8k / 16k / 32k and reports retrieval accuracy. Add a short README explaining how to reproduce.
- [ ] 4.3 [easy] Repeat 4.1 and 4.2 on Llama-3.1-8B-Instruct and Gemma-3-4B-it. Update the validation report.
- [ ] 4.4 [medium] Measure VRAM (or RSS, on CPU) for KV-cache only at 32k context, across `f16` / `q8_0` / `q4_0` / `q4_asym`. Report the absolute bytes and the ratio. This is the maintainer-facing "why merge this" headline.

### Phase 5: Upstream PR and iteration

The PR description leads with the validation matrix from Phase 4 and the orthogonality argument: this is opt-in, changes no defaults, and lays the foundation for `q2_asym` and CUDA-native asymmetric paths as follow-up PRs. Cite the KIVI paper, the prior llama.cpp discussion threads on KV-cache quantization variants, and any related work already in `master` (`Q4_K`, `Q8_0` KV).

- [ ] 5.1 [easy] Squash to three commits on a feature branch: (a) ggml types + tests, (b) KV-cache wiring + CLI, (c) validation script + docs. Rebase clean on upstream master.
- [ ] 5.2 [easy] Ask the user to push to their fork.
- [ ] 5.3 [medium] Open the PR via `gh pr create`. Body sections: Summary, Why, Changes, Validation, Benchmarks, Follow-ups, Checklist. No marketing language; numbers up front.
- [ ] 5.4 [hard] Iterate with maintainers. Anticipated review points: (i) why not just use Q4_K KV — answer with the channel-outlier test from 2.2; (ii) FA disable is a perf regression — answer that FA users opt out by not selecting `q4_asym`; (iii) why not 2-bit — answer that 2-bit is a follow-up gated on SLM-scale validation work that doesn't exist yet.

## Deferred / out of scope

- **2-bit asymmetric KV.** The KIVI paper's headline result. Deferred to a follow-up PR after the 4-bit path is merged and the SLM-scale quality data is in. Revisit when Phase 0's 4-bit validation succeeds and the maintainers have signaled appetite for more KV-cache work.
- **CUDA-native asymmetric KV.** The current PR ships CPU only. CUDA users can run with `-ngl 0` for the new types or use existing symmetric Q4_K on GPU. A follow-up PR adds a CUDA dequant kernel; the kernel that fuses asymmetric quant into FA is a separate, much harder follow-up. Revisit after the CPU PR merges.
- **Metal / ROCm / Vulkan backends.** Same shape as CUDA. Sequence: CPU merges, CUDA dequant merges, then port to the other backends one at a time as separate PRs.
- **FlashAttention with asymmetric K scales.** The real performance unlock. Requires a kernel rewrite to load per-channel scales mid-tile. Out of scope for the MVP and the immediate CUDA follow-up. Revisit only after the dequant-then-attend path is in `master` and there is measured demand for the fused variant.
- **GQA scale-recomputation per K head if Phase 0.2 fails.** If the published per-channel statistics don't survive GQA, the fix is to identify outlier channels per K-head-group, which doubles the metadata. Decide during Phase 0; do not pre-design.
- **Calibration step.** KIVI is tuning-free (statistics measured at runtime). If empirical statistics turn out to need a small calibration set at SLM scale, design it then; do not pre-build the asset pipeline.

## Glossary (project-specific terms)

- **KIVI**: Liu et al., ICML 2024, arXiv:2402.02750. The originating paper. See `wiki/techniques/kivi.md`.
- **Per-channel quantization (for K)**: one scale per hidden-dim channel, shared across all tokens in the cache.
- **Per-token quantization (for V)**: one scale per token position, shared across all channels.
- **Asymmetric (in this spec)**: refers to K and V using *different* quantization axes, not to signed-vs-unsigned scale offsets. Both axes use the existing symmetric (offset-zero) numeric scheme inside each scale group.
- **Q4_K family**: llama.cpp's existing 4-bit block-quantization scheme. The new types reuse its block layout and SIMD path; only the scale-grouping axis differs.
- **NIAH**: Needle-in-a-haystack. The standard long-context retrieval probe; insert a known fact at position P in a long irrelevant context, ask the model to retrieve it.
- **MVP merge gate**: the smallest set of changes that (a) demonstrates the KIVI asymmetric insight in upstream, (b) changes no defaults, (c) has benchmark validation at SLM scale, and (d) leaves CUDA / FA / 2-bit / cross-backend work as labeled follow-ups.

---

### How this file evolves

- An item closes by flipping `- [ ]` → `- [x]` in the same commit as the code change.
- When all items in a phase close, append a "completed" block: 1-paragraph result + bulleted file citations + the validation-report delta. Keep the checklist; it is the historical record.
- Mid-phase discoveries go to `TODO.md`'s "Next up", not directly here. The next cycle decides whether they merit a new phase or were follow-ups to the current one.

### Difficulty labels

Every open `- [ ]` carries `[easy]` / `[medium]` / `[hard]` immediately after the checkbox. Same routing semantics as the skill-set template: easy = Haiku/low effort, medium = Sonnet/medium effort, hard = Opus/high effort. Closed `- [x]` items don't carry labels (historical).

### Sub-item IDs

Every item carries a stable `<phase>.<n>` ID prepended before the difficulty bracket. IDs are 1-indexed within each phase block, assigned once, never renumbered. Inserts between existing items use letter suffixes (`<phase>.<n>a`).
