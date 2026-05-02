# Open Questions and Source Contradictions

> **Summary:** Where the literature contradicts itself or the wiki cannot resolve a claim from current sources. These are the items to verify before relying on cited numbers in production decisions.

**Sources:** synthesis across all wiki pages.

---

## Q1: Does Q4 actually preserve agentic capability at 1-4B?

**The claim chain:** AWQ paper says "Q4 with negligible loss" on 7B+ Llama. Phi-4-Mini paper claims competitive performance "at smaller footprint." Practitioners assume Phi-4-mini-Q4 ≈ Phi-4-mini-FP16 for agentic use.

**The contradiction:** [SLMQuant](../techniques/slmquant.md) shows quant transfer is broken at SLM scale. Specific agentic-coding numbers at SLM × Q4 are not published.

**Status:** Open. See [missing-evals Gap 1](missing-evals.md).

---

## Q2: Does MoE win on agentic coding, or just on training data?

**The claim chain:** [Qwen3-Coder-Next](../models/qwen3-coder-next.md) (80B/3B-active) competitive on SWE-Bench at small active-param. Suggests MoE is the right architecture for agentic coders.

**The contradiction:** [Manik & Wang's tradeoffs paper](../benchmarks/dense-vs-moe-reasoning-tradeoffs.md) (April 2026) found dense Gemma-4-E4B beat MoE Gemma-4-26B-A4B at much smaller VRAM on reasoning. Suggests MoE is *not* automatically better.

**Resolution attempt:** MoE may win on agentic coding specifically because of *RL with environment feedback* (Qwen3-Coder-Next's training recipe), not MoE itself. The active-param efficiency lets you run more update steps per dollar during RL. Whether the same MoE without that RL recipe would beat the dense baseline is unmeasured.

**Status:** Partially open; lean toward "MoE + agentic RL is a joint win, not MoE alone."

---

## Q3: How big is EAGLE-3's draft head, exactly?

**The claim chain:** EAGLE-3 paper says ~1.4x over EAGLE-2, up to 6.5x over standard. Practitioners assume a small footprint (single-layer + LM head).

**The contradiction:** Papers don't always quote exact head sizes; the multi-layer feature fusion in EAGLE-3 *might* require additional projection layers. The wiki currently estimates 150-400 MB for the head; this is an order-of-magnitude estimate, not a measurement.

**Status:** Open; verify against actual EAGLE-3 release (github.com/SafeAILab/EAGLE).

---

## Q4: Do block-diffusion drafters (DDTree) actually beat EAGLE-3 *on a 4 GB GPU*?

**The claim chain:** [DDTree](../techniques/ddtree.md) (April 2026) claims block-diffusion drafter beats EAGLE-3.

**The contradiction:** Speedup-over-EAGLE-3 numbers were not in the abstract. Block diffusion typically requires more parameters than an EAGLE head; on a VRAM-tight device, the parameter delta could erase the throughput win.

**Status:** Open; check full DDTree paper or wait for benchmarks.

---

## Q5: SSD-tier MoE latency floor on consumer NVMe

**The claim chain:** [FlashMoE](../runtimes/flashmoe.md) achieves "up to 2.6x speedup" on user-grade desktop with SSD tier.

**The contradiction:** SSD random-read latency is ~ 50-100 µs; if the loop hits cold experts every layer, the per-token latency exceeds the GPU compute time and the throughput claim has a context-dependent ceiling.

**Status:** Open; FlashMoE workloads may have hot-cache patterns; on novel coding distributions (which lack temporal locality), worst-case latency could be much worse than reported.

---

## Q6: Does the "350M can do agentic tool use" result generalize beyond ToolBench?

**The claim chain:** [350M tool-calling paper](../training/slm-agentic-tool-calling.md) shows 77.55% on ToolBench, beating ToolLLaMA's 30%.

**The contradiction:** ToolBench is now considered easier than [BFCL v3/v4](../benchmarks/bfcl.md) and far easier than real agentic coding loops. The 77.55% does not directly compare to BFCL multi-turn scores. OPT-350M is a weak base; with a modern 350M base the result should be higher, but the gap to "production-grade agentic coder" remains unmeasured.

**Status:** Open. The result is encouraging directionally; absolute capability ceiling at 350M unmeasured on harder benchmarks.

---

## Q7: How aggressive can KV compression go before agentic loops break?

**The claim chain:** [StructKV](../techniques/structkv.md) and [DASH-KV](../techniques/dash-kv.md) both claim "matches full attention quality." [KIVI](../techniques/kivi.md) at 2-bit "preserves quality." Compounded, these promise 10x+ KV reduction.

**The contradiction:** All measured on long-context retrieval (LongBench, RULER); single-turn tasks. Agentic coding has multi-turn KV access patterns where importance shifts per turn (the "globally important" tokens of turn 1 may be irrelevant by turn 4). No published evaluation in this regime.

**Status:** Open. Conservative recommendation: 4x KV reduction safe; 10x+ untested.

---

## Q8: Can an EAGLE-3 head trained on general distribution serve a coder-tuned target?

**The claim chain:** EAGLE-3 papers train draft heads on general chat data; deployment uses general chat targets.

**The implicit assumption:** Draft head trained on chat distribution serves coder-tuned target adequately.

**Status:** Almost certainly suboptimal. Distribution mismatch reduces acceptance rate. Curating a coder-distribution-matched draft head is a small but high-leverage contribution opportunity (see [contribution roadmap](contribution-roadmap.md)).

---

## Q9: Does Claude Code's tool format actually generalize to local Cline / Goose deployments?

**The claim chain:** Claude Code uses Anthropic JSON tool calls. Cline uses XML-tagged. Goose uses MCP. They are similar but not identical.

**The contradiction:** A model fine-tuned for Claude Code's exact format will have format errors when deployed to Cline. The error rate is not measured.

**Status:** Open. The wiki's [harness comparison](harness-comparison.md) recommends matching the deployment harness in the SFT data; verifying the cross-harness transfer cost is unmeasured.

---

## Q10: Is the 4 GB target tractable at all for "Claude Code-like" performance?

**The hard question.** Combining the above unknowns honestly:

- A 4 B model at Q4 fits in 4 GB with compressed KV cache.
- Targeted SFT on harness format gets format conformance to ~99%.
- EAGLE-3 + Saguaro give ~ 6-9x throughput.
- For raw IQ on coding, the gap to Claude Sonnet 4.6 is large.

**Best-case operating point:** Phi-4-mini-reasoning Q4 + Cline + EAGLE-3 + Saguaro + ToolACE-style SFT data probably reaches "useful agentic coder" but not "Claude Sonnet 4.6 equivalent." More like "Claude Haiku 3.5-class capability with Claude Code-class format conformance, at $0/turn."

That's still a meaningful win for many solo-dev workflows.

**Status:** This is the question the wiki is trying to answer. Phase 5's [contribution roadmap](contribution-roadmap.md) lays out the experimental program to settle it.

## See Also

- [Missing evals](missing-evals.md)
- [Contribution roadmap](contribution-roadmap.md)
- [4 GB budget math](four-gb-budget-math.md)
