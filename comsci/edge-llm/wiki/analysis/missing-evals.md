# Missing Evals at the 4 GB Frontier

> **Summary:** What nobody has measured publicly that the wiki's target use case requires. Each gap is a candidate solo-dev contribution. The headline gap: **no published (model × quant × runtime × harness) matrix at the 4 GB envelope on agentic-coding metrics.**

**Sources:** synthesis across the wiki; benchmark pages; Phase 3 analyses.

---

## Gap 1: SLM-scale quantization ablation on agentic metrics

[SLMQuant](../techniques/slmquant.md) shows LLM quant doesn't transfer cleanly to SLMs. But its evaluation tasks are general (perplexity, MMLU-class), not agentic-coding-specific.

**Missing:** Phi-4-mini × {Q5_K_M, Q4_K_M, Q3_K_M, IQ3_XXS, IQ2_XS} × {[BFCL v3 multi-turn](../benchmarks/bfcl.md), [Aider polyglot edit-correctness](../benchmarks/aider-polyglot.md), [SWE-Bench Lite](../benchmarks/swe-bench.md)}.

This is a 5 × 3 = 15 cell matrix. Cost: < $50 cloud compute (these are all small-model evals). Highest-leverage immediate measurement.

## Gap 2: Hybrid architecture coder performance

[Zamba2](../architectures/mamba-ssm-hybrids.md) and [LFM2](../models/lfm2.md) both claim better quality-per-VRAM for general tasks. **No published numbers on agentic coding benchmarks**; only general MMLU/IFEval/GSM8K.

**Missing:** Zamba2-7B and LFM2-2.6B / 8.3B on Aider polyglot, BFCL v3, SWE-Bench Lite.

If hybrids are actually competitive on agentic coding, the long-context KV-cache advantage is decisive at 4 GB.

## Gap 3: Speculative decoding at the 4 GB envelope

EAGLE-3, Saguaro, DDTree, MoE-Spec all benchmark on server-grade GPUs. **No published numbers on consumer 4 GB hardware** with realistic agentic prompt distributions.

**Missing:** Phi-4-mini-Q4 + EAGLE-3 draft head + Saguaro scheduling vs autoregressive baseline on actual coding traces (e.g., replays of aider sessions). Wall-clock end-to-end latency for a representative 4-turn agent loop.

## Gap 4: Tool-call format conformance at SLM scale

Anecdotally, small models fail at agentic loops because of tool-call format errors, not capability. **No paper has decomposed this rigorously** at the 1-4B scale.

**Missing:** For each candidate model (Phi-4-mini, Gemma 3-4B, Qwen3-Coder-3B-class, xLAM-2-3B), measure:
- Tool-call format-correctness rate per turn.
- Format-error rate after Q4 quant.
- Recovery-from-error rate when test feedback is provided.

This breaks the "small model can't agent" claim into the actual sub-component that's broken.

## Gap 5: SSD-tier MoE on real consumer hardware

[FlashMoE](../runtimes/flashmoe.md) demonstrates SSD-tier MoE on "user-grade desktop." **No measurement on a typical laptop with a typical NVMe drive** running a real agentic coding workload.

**Missing:** Qwen3-Coder-Next (80B/3B-active) on a 4 GB laptop GPU + 32 GB DDR5 + Gen4 NVMe, measured on Terminal-Bench 2.0 with full agent loop.

Latency floor matters: SSD random read is 50-100 µs; if the loop hits cold experts every step, the per-token cost is dominated by storage, not compute.

## Gap 6: Distillation-quality ceiling for 1-4B agentic coders

[MoL distillation](../training/mol-distillation.md), HEAL, and the original DeepSeek-R1-Distill recipe each claim improvements but on different bases and benchmarks. **No head-to-head comparison** at the 1-4B scale on coding-agent metrics.

**Missing:** Same teacher (e.g., DeepSeek-R1 or Claude Sonnet 4.6 traces), same student base (e.g., Qwen3-Coder-3B), three distillation recipes (vanilla SFT vs MoL vs HEAL), measured on Aider polyglot and SWE-Bench Lite.

## Gap 7: KV compression interaction with multi-turn agentic loops

[StructKV](../techniques/structkv.md), [DASH-KV](../techniques/dash-kv.md), [KIVI](../techniques/kivi.md) measured on long-context retrieval (LongBench, RULER). Agentic loops have a different access pattern: short turns, frequent re-reading of a stable context (system prompt + tool defs + recent files).

**Missing:** Compounding loss across 8+ turns when KV is aggressively compressed at every turn. Does StructKV's "global hub" preservation hold up in a tool-loop where importance shifts per turn?

## Gap 8: Open published reproducible eval harness

Nobody ships a "give me a model + quant + runtime, get back the (Aider polyglot, BFCL, SWE-Bench Lite) numbers in one command" tool tied to a 4 GB-VRAM constraint.

**Missing:** A docker-packaged eval suite. Input: HF model name, quant scheme, runtime config. Output: a standardized scorecard.

This is itself a contribution path: see [harness-eval-suite-design.md](harness-eval-suite-design.md).

## Priority ranking for the solo-dev

| Gap | Effort | Cost | Leverage |
|---|---|---|---|
| 1: SLM quant × agentic | Low | <$50 | High (immediately publishable) |
| 4: Format conformance breakdown | Low | <$30 | High (validates the wiki's central thesis) |
| 8: Reproducible eval harness | Medium | <$200 | Very high (multiplier on every other gap) |
| 6: Distillation comparison | Medium | $200-500 | High (publishable) |
| 2: Hybrid coder benchmarks | Medium | $100-200 | Medium-high |
| 3: SD at 4 GB end-to-end | Low | $50-100 | Medium |
| 7: KV-compression in agent loops | Low | $50-100 | Medium |
| 5: MoE on laptop NVMe | Medium | $0 (hardware available) | Medium |

Three of these (1, 4, 8) compose into a single coordinated work package; see [contribution roadmap](contribution-roadmap.md).

## See Also

- [Open questions](open-questions.md)
- [Contribution roadmap](contribution-roadmap.md)
- [Harness eval suite design](harness-eval-suite-design.md)
