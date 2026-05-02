# Solo-Dev Contribution Roadmap

> **Summary:** Six ranked contribution paths for a solo developer with a small budget (≤$500/mo cloud + a 4 GB laptop GPU). The wiki's central thesis: **the bottleneck is not model size or quantization research, it is targeted format-conformant SFT data and reproducible eval infrastructure.** Both are tractable for one person.

**Sources:** synthesis across the wiki; [missing-evals](missing-evals.md); [open-questions](open-questions.md).

---

## The framing

A solo dev with a 4 GB laptop and a small cloud budget cannot:
- Pretrain a foundation model.
- Train novel architectures from scratch (Mamba/SSM hybrids).
- Run extensive multi-GPU experiments.
- Compete with Anthropic / OpenAI / Google on raw model capability.

A solo dev *can*:
- Curate / synthesize high-quality SFT datasets.
- Fine-tune small open-weight bases via Unsloth + LoRA.
- Build reproducible benchmark suites.
- Contribute to llama.cpp / Unsloth / vLLM as a developer.
- Distill from existing frontier models (Claude / GPT) via API.

The contribution paths below are ranked by **leverage** (likelihood of being load-bearing for downstream community work) × **tractability** (within the solo budget).

---

## Path 1 (highest leverage): Agentic-format SFT dataset + recipe

**Hypothesis:** Targeted SFT on a 1-3B base with the *exact* tool-call format of a deployment harness (Cline / aider / Goose) closes the gap to much larger models on agentic-coding metrics. Direct evidence: [350M tool-calling paper](../training/slm-agentic-tool-calling.md) shows it works at 350M.

**Concrete plan:**

1. Replay 5-50K Claude Code (or aider with Claude Sonnet 4.6) sessions on real OSS issues. Cost: < $200 in Claude API.
2. Filter for successful sessions (passed tests).
3. Re-format the captured traces into the target harness's format (e.g., Cline XML or aider SEARCH/REPLACE).
4. SFT a Qwen3-Coder-3B / Phi-4-mini base via Unsloth + LoRA. Cost: < $100 cloud.
5. Evaluate on [BFCL v3](../benchmarks/bfcl.md), [Aider polyglot](../benchmarks/aider-polyglot.md) edit-correctness column, [SWE-Bench Lite](../benchmarks/swe-bench.md).
6. Release: dataset + LoRA adapter + model card + evaluation scorecard.

**Expected outcome:** A small open-weight model that matches a much larger general-purpose model on agentic coding metrics inside one specific harness. Format-conformance rate near 99%.

**Total cost:** $300-500. **Time:** 4-8 weeks.

**Why it's load-bearing:** Multiple downstream papers will cite it; the dataset itself is reusable across all subsequent work. This is the paper Anthropic / DeepSeek would not write themselves.

---

## Path 2 (highest infrastructure leverage): 4 GB-envelope eval harness

**Hypothesis:** Nobody publishes a reproducible (model × quant × runtime × harness × benchmark) matrix at the 4 GB envelope. The community needs one.

**Concrete plan:**

1. Docker-packaged eval suite. Input: model name, quant scheme, runtime, harness adapter. Output: standardized scorecard.
2. Initial benchmark coverage: [BFCL v3](../benchmarks/bfcl.md), [Aider polyglot](../benchmarks/aider-polyglot.md) (full + edit-correctness column), [SWE-Bench Lite](../benchmarks/swe-bench.md), [LiveCodeBench](../benchmarks/livecodebench.md), [Terminal-Bench 1.0](../benchmarks/terminal-bench.md).
3. Initial coverage: 10-15 candidate (model × quant) tuples on llama.cpp / KTransformers / vLLM.
4. CI: nightly re-run on a fixed seed with new model releases.
5. Publish results dashboard.

**Expected outcome:** The default citation for "what should I run on 4 GB?" The first published numbers settling the SLM-quant-vs-agentic-capability gap.

**Total cost:** $200/mo ongoing for cloud eval runs. **Time:** 6-12 weeks for v1.

**Why it's load-bearing:** Multiplier on every other contribution. Path 1's SFT recipe needs Path 2's eval suite to demonstrate the win.

---

## Path 3: Quant-aware tool-call fine-tune

**Hypothesis:** Most agentic SFTs are FP16/BF16; the model degrades after Q4 quant, especially in tool-call format conformance. A QLoRA + GPTQ-aware (or AWQ-aware) training closes this gap.

**Concrete plan:**

1. Take Path 1's dataset.
2. Apply quantization-aware fine-tuning via Unsloth's QLoRA workflow.
3. Compare: (FP16 SFT → Q4 deploy) vs (QLoRA-with-Q4-target SFT → Q4 deploy) on Path 2's eval suite.

**Expected outcome:** A 2-5 percentage-point recovery of post-quant agentic performance. Cheap, novel-ish, publishable as a follow-up to Path 1.

**Total cost:** $100-200. **Time:** 2-4 weeks (depends on Path 1 being done).

---

## Path 4: Coder-distribution-matched EAGLE-3 draft head

**Hypothesis:** Published EAGLE-3 heads are trained on general chat distribution. Coder-tuned targets get suboptimal acceptance rates. A coder-distribution draft head increases speedup on the deployment workload.

**Concrete plan:**

1. Generate / collect coder-distribution training data (replay coder-target outputs).
2. Train an EAGLE-3 head on that distribution against a coder-tuned target (e.g., Qwen3-Coder-3B).
3. Measure acceptance rate vs the published general-purpose head, vs autoregressive baseline.
4. Release the head + training script.

**Expected outcome:** 1.2-1.5x throughput improvement over a generic EAGLE-3 head on coding tasks; tractable because EAGLE-3 heads are tiny.

**Total cost:** < $100. **Time:** 2-4 weeks.

**Why it's load-bearing:** Speeds every downstream coder-target deployment.

---

## Path 5: llama.cpp / Unsloth / vLLM contributions

**Hypothesis:** Many of the 2026 papers (Saguaro/SSD, DDTree, MoE-Spec, DASH-KV, StructKV) are not yet integrated into the practitioner runtimes. PRs implementing them are tractable for an experienced developer, with no compute budget required.

**Concrete plan:**

1. Pick one (e.g., Saguaro/SSD scheduling for vLLM, or KIVI implementation for llama.cpp Q-cache, or MoE-Spec for KTransformers).
2. Implement, test, PR.
3. Iterate with maintainers.

**Expected outcome:** Real production impact; community standing; warm-up for harder contributions later.

**Total cost:** $0. **Time:** variable (2-12 weeks per PR).

---

## Path 6 (lowest leverage): Synthetic trace dataset release

**Hypothesis:** Even without running the SFT yourself, a high-quality replayed Claude-Code trace dataset is a standalone contribution. Downstream labs will use it.

**Concrete plan:**

1. Replay 50K Claude Code sessions on diverse OSS repos.
2. Filter, deduplicate, format into HuggingFace Dataset.
3. Document the schema; release.

**Expected outcome:** 100s of citations / forks; ecosystem multiplier.

**Total cost:** $200-300 in Claude API. **Time:** 3-6 weeks.

**Why it's listed last:** Lower personal leverage (someone else gets the SFT win), but a real contribution if Path 1 is too ambitious.

---

## Lower-leverage / explicitly avoid

- **Pretraining anything from scratch.** Compute-prohibitive; no marginal contribution.
- **Novel architectures (custom Mamba/SSM hybrid).** Too far from working agentic coder for solo budget. Wait for Liquid AI, Zyphra, NVIDIA to ship the next versions.
- **Generic instruction tuning.** Saturated.
- **Yet another small chat model.** Saturated.

---

## Sequencing recommendation

```
Week 1-2:   Path 5 (llama.cpp / vLLM warm-up PR)
Week 3-8:   Path 2 (eval harness v1) running in parallel with
Week 4-10:  Path 1 (SFT dataset + recipe)
Week 11-14: Path 3 (quant-aware fine-tune as Path 1 follow-up)
Week 15-18: Path 4 (coder-matched EAGLE head)
Anytime:    Path 6 (dataset release; can spin out of Path 1's data prep)
```

By month 4-5, the solo dev has shipped: an eval suite the community uses, a published SFT recipe + dataset, a quant-aware variant, a coder draft head, and accumulated standing in 1-2 runtime communities. That's a substantive year-of-work output.

## See Also

- [Agentic SFT recipe](agentic-sft-recipe.md)
- [Harness eval suite design](harness-eval-suite-design.md)
- [Missing evals](missing-evals.md)
- [Open questions](open-questions.md)
