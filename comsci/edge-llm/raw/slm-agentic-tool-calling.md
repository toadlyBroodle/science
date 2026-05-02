# Small Language Models for Efficient Agentic Tool Calling

**Source:** arXiv:2512.15943 (https://arxiv.org/abs/2512.15943)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Polaris Jhandi, Owais Kazi, Shreyas Subramanian, Neel Sendas
**Venue:** AAAI 2026 Workshop on Agentic AI Benchmarks for Enterprise
**Submitted:** 2025-12-17; revised 2026-03-09

## Abstract / extracted content

A 350M-parameter model (OPT-350M base) fine-tuned with HuggingFace TRL SFT for one epoch beats much larger tool-calling baselines on ToolBench. The central claim: targeted SFT on tool-call traces lets a 350M model handle production tool-calling at fraction-of-a-percent the compute of larger models.

## Headline numbers (ToolBench pass rate)

- **Fine-tuned OPT-350M: 77.55%**
- ChatGPT-CoT: 26.00%
- ToolLLaMA-DFS: 30.18%
- ToolLLaMA-CoT: 16.27%

## Method

- Base model: OPT-350M (Meta, 2022).
- Framework: HuggingFace TRL.
- Recipe: Supervised Fine-Tuning trainer.
- Duration: 1 epoch.

## Why this matters for the wiki

The wiki's contribution roadmap (Phase 5) highlights "agentic SFT recipe + dataset" as the highest-leverage solo-dev path. This paper is direct evidence that the recipe works at the lowest end of the parameter scale: a 350M model fine-tuned for one epoch can outperform much larger general-purpose tool-callers. The constraint is *targeted training*, not model size.

For 4 GB VRAM the implication is sharp: a 1-3B base + targeted tool-call SFT may outperform an out-of-the-box 7-8B at agentic tasks, while costing 1/3 the VRAM and 1/3 the throughput.
