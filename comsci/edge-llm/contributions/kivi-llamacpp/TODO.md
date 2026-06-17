# KIVI-in-llama.cpp TODO (handoff doc)

> Cross-cycle state. Read on start, update on close. Three sections, in this order. Primary spec: `SPEC.md`.

## In flight

Decision pending on how to use jagmarques's 2026-05-17 reply. Three immediate options live: (a) draft a short technical follow-up reply asking one clarifying question about per-head identification (offline calibration vs runtime activation statistics); (b) silent ack and pivot to scoping the Branch C per-head fp16 masking PR surface; (c) wait for further thread activity before either. Awaiting user direction.

## Just shipped (last cycle)

- Pre-PR discovery: surveyed existing KV-cache quantization work in ggml-org/llama.cpp. Found #21089 (TurboQuant CPU KV cache, OPEN), #21551 (existing-quant asymmetric exploration, OPEN draft), #21591 (NexusQuant findings, CLOSED), and the recently-landed KV-rotation work. Conclusion: the originally-drafted KIVI MVP duplicates #21089's structural plumbing and targets the wrong bitcount (Q4 vs Q3 sweet spot per #21591). — at 2026-05-16T22:30Z
- Posted comment on ggml-org/llama.cpp#21089 surfacing two specific NexusQuant findings not yet cited in the thread: (1) K3 → K4 PPL delta is +0.06pp (softmax error floor at ~3 bits, K8 over-allocated); (2) layer-position fp16 boundary protection recovers Qwen2.5-7B catastrophic break without raising bulk K bitcount. Framed as data, not direction. — by toadlyBroodle at 2026-05-16T23:01:47Z
- Engagement received: @jagmarques (NexusQuant author) replied 8.5 hours later with a new datapoint — per-head fp16 masking of the lowest-2% KV-heads on Qwen2.5-7B-Instruct matches K3V2 boundary-protect on retrieval at 2.28 vs 3.93 K-bits/element. Explicitly framed as orthogonal to #21089's quant-type scope but relevant to the K-bit/arch-sensitivity question. — observed at 2026-05-17T07:42:05Z

## Next up (queued for next cycle)

Two immediate items, then a longer conditional queue. The first three are actionable now; the rest remain gated on branch selection (SPEC.md "Status").

- [easy] Grep upstream `ggml-org/llama.cpp` for existing per-head precision-override plumbing in `src/llama-kv-cache.cpp`, `src/llama-context.cpp`, and surrounding kv files. Look for: any path that lets a subset of K or V heads run at a different precision than the rest (`n_head_kv` per-head loop, head-level masking, head-selective dequant). If present, the Branch C contribution would be a CLI knob plus an outlier-head heuristic; if absent, it's a structural change to KV allocation. Output: one-paragraph status note appended to this file. Reason: Branch C scoping; under an hour of work; determines whether per-head fp16 masking is a small or large PR.
- [medium] Read the recently-landed KV-rotation PR (referenced from #21551). Understand whether rotation's K-side effect interacts with per-head outlier statistics (does rotation flatten outliers per-head as well as per-channel?). If yes, jagmarques's "lowest-2% KV-heads" set may shift or vanish post-rotation, which changes the Branch C value proposition. Capture findings in a new `wiki/techniques/kv-rotation.md`. Reason: load-bearing for any Branch C PR design; rotation is a confound on the published per-head numbers.
- [easy] Decide whether to post a short follow-up reply on #21089 to jagmarques. Candidate content: thank them for the data; one clarifying question on per-head identification (offline calibration set vs runtime activation statistics); note we are scoping the upstream per-head precision-override surface as a separate-PR candidate. Reason: jagmarques is the most authoritative voice on this surface and engaged within hours; a single thoughtful follow-up sustains the connection without becoming a thread participant. Decline if user prefers silent ack.
- [medium] Conditional on Branch A (asymmetric K/V lands in #21089): pivot Phase 0 PyTorch validation into asymmetric K3V2 measurements on Qwen3-4B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-4B-it. Post as a data comment on the merged PR's follow-up issue. Reason: contributes evidence to asymmetric config selection without competing PR.
- [hard] Conditional on Branch B (symmetric-only merges): revive SPEC.md Phase 1-5 retargeted at Q3 asymmetric on top of the new TBQ types. Reason: KIVI's central insight still applies; bitcount drops from Q4 to Q3 per #21591.
- [hard] Conditional on Branch C (most interesting new branch): design and implement a per-head fp16 KV-head precision-override path in llama.cpp. Scope depends on the upstream grep above. If small: CLI knob `--kv-head-precision <comma-list-of-head-indices>:f16` plus an optional heuristic that identifies outlier heads from activation magnitudes during prompt processing. If large: structural change to KV-cache allocation to support mixed-precision per-head storage. Reason: fresh PR surface; cited by NexusQuant author; orthogonal to #21089 so no in-flight competitor; clean citation chain.
- [medium] Conditional on Branch D: re-evaluate Path 5 pairings from wiki/analysis/contribution-roadmap.md. Saguaro/SSD scheduling for llama.cpp is the next-best candidate. Reason: KIVI/asymmetric-KV angle is dead in this scenario.
- [easy] Regardless of branch: read TheTom's referenced docs in their turboquant_plus fork (m5-max-stress-test.md, layer-aware-v-compression.md, sparse-v-dequant.md, block-size-experiment.md). They contain measurement detail that's not in the PR thread itself. Reason: cheap context-gathering.

<!--
  Items below this line are the original Phase 0-5 task queue, now blocked by the
  SPEC.md "Status" gate. Do not pick any of these until a branch is selected.
-->

- [easy] [BLOCKED] Clone `ggml-org/llama.cpp` to `/tmp/llama.cpp`, set up remotes once the user supplies a fork URL. Reason: SPEC Phase 1-5 prereq; blocked pending branch selection.
- [easy] [BLOCKED] Clone `jy-yuan/KIVI` reference repo. Reason: SPEC 0.1 prereq; blocked pending branch selection.
- [medium] [BLOCKED] Download Qwen3-4B-Instruct / Llama-3.1-8B-Instruct / Gemma-3-4B-it artifacts. Reason: SPEC 0.3 + 4.1-4.3 input; blocked pending branch selection.
- [hard] [BLOCKED] SPEC 0.2: GQA outlier-statistics derivation. Reason: blocked pending branch.
- [medium] [BLOCKED] SPEC 0.3: PyTorch validation matrix. Reason: blocked pending branch (will retarget to K3V2 under Branch A or remain Q4 under Branch B).
- [easy] [BLOCKED] SPEC 0.4: validation-report.md.
- [medium] [BLOCKED] Pre-PR draft issue on llama.cpp. Reason: superseded by 2026-05-16 #21089 comment; only re-actionable under Branch B/C.
- [medium] [BLOCKED] SPEC 1.1-1.4: ggml type definitions. Reason: Branch B only.
- [easy] [BLOCKED] SPEC 2.1-2.3: CPU implementation + unit tests. Reason: Branch B only.
- [hard] [BLOCKED] SPEC 3.1-3.4: KV-cache wiring + CLI flags. Reason: Branch B only.
- [medium] [BLOCKED] SPEC 4.1-4.4: long-context quality validation in-tree. Reason: Branch A pivots to standalone data; Branch B carries this in.
- [easy] [BLOCKED] SPEC 5.1-5.4: upstream PR + iteration. Reason: Branch B only.
