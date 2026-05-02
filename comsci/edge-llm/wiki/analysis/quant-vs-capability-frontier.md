# Quant-vs-Capability Frontier

> **Summary:** Where capability degrades for each quantization scheme, with explicit corrections for the small-language-model regime per [SLMQuant](../techniques/slmquant.md). The headline: **published 4-bit "negligible loss" claims were measured on 7B+ LLMs and do not transfer cleanly to 1-4 B SLMs.** Practitioners should re-verify quant choices on the actual deployment model and task.

**Sources:** [techniques/awq.md](../techniques/awq.md), [techniques/gptq.md](../techniques/gptq.md), [techniques/aqlm.md](../techniques/aqlm.md), [techniques/slmquant.md](../techniques/slmquant.md), [techniques/kivi.md](../techniques/kivi.md), [models/phi-4-mini.md](../models/phi-4-mini.md), [models/gemma-3.md](../models/gemma-3.md)

---

## The general LLM-era frontier (7B+)

| Quant | Effective bits | Quality vs FP16 | Recommended for |
|---|---|---|---|
| FP16 / BF16 | 16 | 100% | Reference |
| Q8 / INT8 | 8 | 99.5%+ | Conservative |
| Q5_K_M | ~5.6 | 99-99.5% | Quality-critical |
| Q4_K_M / AWQ-4 / GPTQ-4 | ~4.5 | 97-99% | Practitioner default |
| Q3_K_M | ~3.6 | 92-97% | Borderline |
| AQLM 2-bit | ~2 | 90-96% | Extreme; specialty |
| IQ2_XS / IQ3_XXS | ~2.5-3 | 85-95% | Importance-matrix; better than naive Q2/Q3 |
| BitNet 1.58 | ~1.58 | varies; QAT only | Specialty |

These ranges are based on aggregated reports from AWQ / GPTQ / AQLM papers and practitioner benchmarks on 7B-70B targets.

## SLM-specific corrections (1-4 B)

Per [SLMQuant](../techniques/slmquant.md) (Wang et al., Nov 2025), the LLM-era ranges shift down for small models:

- **Q4 quality drop is larger.** The "97-99% of FP16" range likely becomes "92-97%" at 1-4B scale.
- **Q3 frequently breaks.** Where Q3 was "borderline" at 7B, it can be "unusable on some tasks" at 1-4B.
- **Activation-aware methods (AWQ) may behave differently** because activation distributions in SLMs are denser; the salient-channel selection is less obvious.
- **Calibration sensitivity rises.** GPTQ-style methods can overfit a small calibration set more visibly in SLMs.

The paper proposes SLM-tailored design principles but is a benchmark, not a new method. The practical implication: assume one tier-of-quality drop per category vs the published 7B+ numbers.

## Per-task quant frontier shifts

Different agentic-coding tasks degrade at different quant levels:

| Task type | First quant level where capability noticeably drops |
|---|---|
| Code completion (single-line) | Q3 |
| Whole-function generation | Q4 (SLM regime) |
| Tool-call format conformance | Q3 (sharp; format regex starts breaking) |
| Multi-turn agentic loops | Q4 (SLM regime; failure compounds across turns) |
| Long-context retrieval (>16k) | Q5 (KV-quant interactions; small-model memory effects) |

Tool-call format conformance is the most quant-sensitive task because the harness rejects malformed output entirely. A 99% format-correctness model becomes a 95% format-correctness model after Q4, which is dramatic for multi-turn loops where errors compound.

## KV-cache quant frontier

[KIVI](../techniques/kivi.md) at 2-bit: documented to preserve quality on 7B+ Llama / Falcon / Mistral. SLM behavior unverified. Conservative estimate: 4-bit KV is the safe floor at 1-4B scale; 2-bit KV warrants per-task verification.

[KIVI](../techniques/kivi.md) + [StructKV](../techniques/structkv.md) compose: KIVI compresses bits per element; StructKV compresses tokens kept. Compounding 2x × 2x on each is a 4x KV reduction with documented small quality loss; pushing to 4x × 4x = 16x is aggressive and untested at SLM scale.

## Recommended quant policy for the 4 GB target

For a serious deployment:

1. **Default: Q4_K_M weights + Q4 KV cache + 50% KV token retention via [StructKV](../techniques/structkv.md).** Gives a solid quality floor while fitting 4 GB at 32k context.
2. **Aggressive: IQ3_XXS weights + Q4 KV + 25% retention.** For 1.5-2B-class models that need extra context headroom. Verify with [BFCL](../benchmarks/bfcl.md) tool-call accuracy before committing.
3. **Conservative: Q5_K_M + FP16 KV + 100% retention.** When the model is the 1-2B class and the budget allows.

## Open question

Nobody has published a comprehensive (model × quant × task) matrix at the 1-4B SLM scale specifically on agentic coding metrics ([Aider polyglot](../benchmarks/aider-polyglot.md), [BFCL](../benchmarks/bfcl.md), [SWE-Bench Lite](../benchmarks/swe-bench.md)). This is one of the wiki's identified [contribution paths](contribution-roadmap.md) (pending Phase 5).

## See Also

- [SLMQuant](../techniques/slmquant.md)
- [4 GB budget math](four-gb-budget-math.md)
- [Spec decoding at 4 GB](spec-decoding-at-4gb.md)
- [Contribution roadmap](contribution-roadmap.md) (pending)
