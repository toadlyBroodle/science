# Speculative Decoding at the 4 GB VRAM Ceiling

> **Summary:** Speculative decoding adds the draft model's footprint to VRAM. At 4 GB the budget is brutal: full separate draft models (e.g., 1B draft + 7B target) don't fit. **EAGLE-3-style auxiliary draft heads (single layer + LM head, ~ 200-500 MB) are the practical default.** The 2026 stack composes Saguaro (parallel scheduling), MoE-Spec (expert budgeting), and emerging block-diffusion drafters (DDTree).

**Sources:** [techniques/eagle-3.md](../techniques/eagle-3.md), [techniques/saguaro-ssd.md](../techniques/saguaro-ssd.md), [techniques/ddtree.md](../techniques/ddtree.md), [techniques/moe-spec.md](../techniques/moe-spec.md), [analysis/four-gb-budget-math.md](four-gb-budget-math.md)

---

## The speculative decoding budget problem

Standard SD: draft model (small) generates K tokens; target model (large) verifies all K in one parallel forward pass. If accepted, big throughput win. The catch: **both models need to be in memory simultaneously**, and the draft model's compute happens at every step.

At 4 GB VRAM, candidate model + draft + KV cache + activations + workspace all share the same 4 GB. A 1 B draft (~ 0.6 GB at Q4) + 4 B target (~ 2.4 GB at Q4) leaves 1 GB for KV cache + activations + workspace; workable but tight.

## Three viable patterns at 4 GB

### Pattern 1: EAGLE-style auxiliary head (default)

[EAGLE-3](../techniques/eagle-3.md) draft head is a single transformer layer + LM head trained on the target model's hidden states. Footprint is small: ~ 150-400 MB depending on hidden dim.

- **Pros:** Tiny VRAM cost. Up to 6.5x speedup over standard decoding. Mature ecosystem (vLLM, recent llama.cpp).
- **Cons:** Acceptance rate depends on training data alignment with deployment distribution. Coder-tuned models need a coder-tuned EAGLE head (a [contribution opportunity](contribution-roadmap.md)).

### Pattern 2: Self-speculative (no draft model)

Medusa, lookahead, and Jacobi-style decoders add multiple LM heads to the *target* model itself, sampling independent next-token candidates that are jointly verified.

- **Pros:** Zero auxiliary VRAM cost.
- **Cons:** Smaller speedup than EAGLE-3 (~ 1.5-2.5x typical). Quality of candidates is lower than a trained draft head.

### Pattern 3: Block-diffusion drafter (emerging)

[DDTree](../techniques/ddtree.md) (April 2026) uses a block-diffusion drafter (DFlash) that generates an entire draft block in one forward pass, beating autoregressive drafters like EAGLE-3.

- **Pros:** Better acceptance / throughput than EAGLE-3. Tree-structured drafts via best-first heap.
- **Cons:** New (April 2026), tooling not yet mature. VRAM footprint comparison vs EAGLE-3 not yet published.

## Composition: Saguaro

[Saguaro / SSD](../techniques/saguaro-ssd.md) (ICLR 2026) is a *scheduling* change, not a model change. It parallelizes speculation and verification: while target verifies batch *t*, draft pre-builds batch *t+1* conditioned on predicted verification outcomes. **30% over optimized SD baselines, no extra VRAM cost.**

Compose: EAGLE-3 (or DDTree) drafter + Saguaro scheduling = state-of-art on dense models.

## MoE-specific: MoE-Spec

For [Qwen3-Coder-Next](../models/qwen3-coder-next.md) and other MoE-on-4-GB scenarios, the SD verification step activates many unique experts simultaneously. Without intervention, this can blow the active-expert VRAM budget and make SD *slower* than autoregressive.

[MoE-Spec](../techniques/moe-spec.md) caps per-layer expert capacity at verification time, dropping the long tail. **+10-30% throughput vs EAGLE-3 baseline on MoE.**

## The full 4 GB-MoE-agentic stack

For Qwen3-Coder-Next (80B/3B-active) on 4 GB:

```
Application layer:    aider / Cline / Goose harness
Inference engine:     KTransformers + DALI (expert assignment) + FlashMoE (SSD tier)
Speculative decoding: EAGLE-3 draft head + Saguaro scheduling + MoE-Spec verification budgeting
KV management:        DASH-KV (compute) + StructKV (storage) + KIVI (bits)
Quant:                AWQ-4 weights, KV at 4-bit, with SLMQuant verification on the actual model
```

This is the maximalist stack. Production deployments today use a subset (typically: KTransformers + AWQ-4 + KV-cache-quant + EAGLE-2/3); the 2026 papers offer further gains as they integrate.

## Caveats

- All speedup numbers are upper bounds measured on server-grade hardware. On a 4 GB laptop GPU, expect 50-70% of paper-claimed speedups.
- Acceptance rate depends on draft-target distribution match. A general-purpose EAGLE-3 head paired with a coder-tuned target is suboptimal; matched draft training is a real-but-tractable contribution.
- Saguaro and DDTree implementations are reference-quality at submission time, not production-tested.

## Solo-dev contribution opportunity

Curating / training a 0.5-1B coder-distribution-matched EAGLE-3 draft head for [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (or smaller dense coders) is a tractable, measurable, publishable contribution. See [contribution roadmap](contribution-roadmap.md) (pending Phase 5).

## See Also

- [4 GB budget math](four-gb-budget-math.md)
- [Runtime comparison](runtime-comparison.md)
- [EAGLE-3](../techniques/eagle-3.md)
- [Saguaro / SSD](../techniques/saguaro-ssd.md)
- [MoE-Spec](../techniques/moe-spec.md)
