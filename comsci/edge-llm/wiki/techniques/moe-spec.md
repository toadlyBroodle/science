# MoE-Spec: Expert Budgeting for MoE Speculative Decoding

> **Summary:** McDanel et al., February 2026 (arXiv:2602.16052). Training-free verification-time technique that fixes a per-layer expert capacity during draft-tree verification, loading only the top-contributing experts and dropping the long tail. **10-30% higher throughput than EAGLE-3 baseline at comparable quality.** Decouples speculation depth from memory cost.

**Sources:** [raw/moe-spec.md](../../raw/moe-spec.md), [raw/eagle-3.md](../../raw/eagle-3.md), [raw/dali-moe.md](../../raw/dali-moe.md), [raw/qwen3-coder-next.md](../../raw/qwen3-coder-next.md)

---

## The problem MoE-Spec solves

Speculative decoding verifies many candidate tokens in parallel. In a dense model, parallel verification is cheap. In an MoE, each verified token activates its own experts; a draft tree of *k* candidates can request *k* × (top-K) unique experts, blowing up memory. Without intervention, SD can be *slower* than autoregressive on an MoE because the verification step exceeds the memory budget.

MoE-Spec's fix: at verification time, enforce a fixed expert capacity *C* per layer, load only the *C* experts contributing most to verification, drop the long tail.

## Numbers

- 10-30% higher throughput vs [EAGLE-3](eagle-3.md) on MoE targets.
- Comparable output quality (training-free; quality bounded by the expert-dropping decision).
- Tight expert budgets trade quality for latency, configurable.

## Position in the 2026 stack

For an MoE on a 4 GB GPU running a coder agent, the expected stack is:

1. [EAGLE-3](eagle-3.md) draft head (small, fits VRAM).
2. **MoE-Spec** at verification time.
3. [DALI](../runtimes/dali-moe.md) for CPU+GPU expert allocation.
4. [Saguaro/SSD](saguaro-ssd.md) for parallel scheduling.
5. [FlashMoE](../runtimes/flashmoe.md) for SSD-tier offload of cold experts.

## Relevance to 4 GB VRAM target

For [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (80B/3B-active), MoE-Spec is the difference between speculative decoding being viable or actively harmful. Without it, draft-tree verification can exceed the active-expert VRAM budget; with it, expert footprint stays bounded by *C*.

## See Also

- [EAGLE-3](eagle-3.md)
- [Saguaro / SSD](saguaro-ssd.md)
- [DALI](../runtimes/dali-moe.md)
- [FlashMoE](../runtimes/flashmoe.md)
- [Qwen3-Coder-Next](../models/qwen3-coder-next.md)
