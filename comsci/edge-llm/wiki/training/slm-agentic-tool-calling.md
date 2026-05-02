# Small Models for Agentic Tool Calling (Jhandi et al., 2025)

> **Summary:** Jhandi et al., AAAI 2026 Workshop (arXiv:2512.15943). A 350M-parameter model (OPT-350M base) fine-tuned via HuggingFace TRL SFT for one epoch beats much larger general-purpose tool-callers on ToolBench. **77.55% ToolBench pass rate vs 30.18% for ToolLLaMA-DFS, 26.00% for ChatGPT-CoT.** Direct evidence that targeted SFT closes the gap at the lowest end of the parameter scale.

**Sources:** [raw/slm-agentic-tool-calling.md](../../raw/slm-agentic-tool-calling.md), [raw/toolace.md](../../raw/toolace.md), [raw/xlam-2.md](../../raw/xlam-2.md)

---

## The result

| Model | ToolBench pass rate |
|---|---|
| **Fine-tuned OPT-350M** | **77.55%** |
| ToolLLaMA-DFS | 30.18% |
| ChatGPT-CoT | 26.00% |
| ToolLLaMA-CoT | 16.27% |

A 350-million-parameter model, fine-tuned for *one epoch* on tool-call traces, more than doubles ToolLLaMA's score and triples ChatGPT-CoT.

## Recipe

- Base: OPT-350M (Meta, 2022; not a frontier base model).
- Framework: HuggingFace TRL.
- Method: Supervised Fine-Tuning trainer.
- Duration: 1 epoch.

That is, no exotic recipe. The leverage is entirely in the SFT data being targeted at the deployment task.

## Why this is load-bearing for the wiki

The wiki's Phase 5 [contribution roadmap](../analysis/contribution-roadmap.md) (pending) ranks "agentic SFT recipe + dataset" as the highest-leverage solo-dev path. This paper is direct evidence the recipe scales *down* further than expected: even at 350M (well below the wiki's 1-4B target), targeted SFT recovers competitive agentic performance.

For 4 GB VRAM the implication is sharp: **a 1-3B base + targeted tool-call SFT may outperform an out-of-the-box 7-8B at agentic tasks**, while costing 1/3 the VRAM and 1/3 the throughput. This inverts the usual "bigger model = better agent" assumption inside the agentic-coding regime where format conformance dominates.

## Pairs with

- [xLAM-2](xlam-2.md): Salesforce's larger-scale published recipe for the same idea.
- [ToolACE](toolace.md): Data-pipeline reference for synthesizing the training set.
- [BFCL v3](../benchmarks/bfcl.md): Closer-to-production agentic evaluation that ToolBench precedes.

## Caveats

- ToolBench is now considered easier than BFCL v3 / v4. The 77.55% does not directly compare to BFCL leaderboard scores.
- One-epoch SFT can overfit narrow distributions; out-of-distribution generalization not measured in the abstract.
- OPT-350M has known instability and is a weak base. A modern 1-3B base with the same recipe should perform substantially better.

## See Also

- [xLAM-2](xlam-2.md)
- [ToolACE](toolace.md)
- [BFCL](../benchmarks/bfcl.md)
- [Contribution roadmap](../analysis/contribution-roadmap.md) (pending Phase 5)
