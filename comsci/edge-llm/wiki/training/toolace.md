# ToolACE

> **Summary:** Liu et al., ICLR 2025 (arXiv:2409.00920). Automated data-generation pipeline for LLM function calling. Self-evolution synthesis curates a 26,507-API pool. Multi-agent interactive system generates dialogs. Dual-layer rule-based + model-based verification. **An 8B model trained on ToolACE-synthesized data hits SOTA on BFCL, rivalling GPT-4 family.**

**Sources:** [raw/toolace.md](../../raw/toolace.md), [raw/xlam-2.md](../../raw/xlam-2.md), [raw/bfcl.md](../../raw/bfcl.md)

---

## What ToolACE produces

ToolACE is a *data pipeline*, not a model. The pipeline outputs SFT-ready dialogs of agents calling tools. Three stages:

1. **API synthesis.** Self-evolution loop generates a diverse 26,507-API pool covering domains far beyond the seed set.
2. **Dialog synthesis.** Multi-agent simulation: a user agent issues requests, a tool-calling agent responds, an environment agent returns API results.
3. **Verification.** Rule-based checks (schema validity, API existence) plus model-based checks (semantic correctness) gate which dialogs enter the training set.

## Headline result

8B model trained on ToolACE data hits SOTA on [Berkeley Function Calling Leaderboard](../benchmarks/bfcl.md), rivalling GPT-4 family.

## Position in the agentic-SFT thread

| Component | Reference |
|---|---|
| Data pipeline | **ToolACE** |
| Larger trained models | [xLAM-2](xlam-2.md) |
| Smallest trained models that work | [350M tool-calling paper](slm-agentic-tool-calling.md) |
| Evaluation | [BFCL](../benchmarks/bfcl.md), [Aider polyglot](../benchmarks/aider-polyglot.md), [SWE-Bench](../benchmarks/swe-bench.md) |

## Relevance for solo-dev contribution roadmap

ToolACE is the open published reference for "how to synthesize tool-call training data." For a solo dev wanting to fine-tune a small base for agentic coding inside Claude Code (the wiki's stated goal), the practical question is whether to:

(a) Reuse the public ToolACE training data.
(b) Adapt ToolACE's pipeline to generate Claude-Code-replay traces specifically.
(c) Combine: ToolACE for breadth, Claude-Code replay for depth on the target distribution.

(c) is the highest-leverage path, addressed in the [contribution roadmap](../analysis/contribution-roadmap.md) (pending).

## See Also

- [xLAM-2](xlam-2.md)
- [SLM agentic tool calling (350M)](slm-agentic-tool-calling.md)
- [BFCL](../benchmarks/bfcl.md)
