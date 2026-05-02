# xLAM / xLAM-2 (Salesforce)

> **Summary:** Salesforce's family of Large Action Models for AI agents, dense + MoE, 1B to 8x22B. xLAM-2 (2025) adds multi-turn conversation and tool use, trained via the APIGen-MT framework that synthesizes data from simulated agent-human interactions. xLAM-2-70b-fc-r reaches 56.2% on τ-bench, beating GPT-4o and approaching Claude 3.5 Sonnet.

**Sources:** [raw/xlam-2.md](../../raw/xlam-2.md), [raw/toolace.md](../../raw/toolace.md), [raw/slm-agentic-tool-calling.md](../../raw/slm-agentic-tool-calling.md)

---

## Model lineup

Function-calling variants released open-weight:
- xLAM-2-1B-r
- xLAM-2-3B-r
- xLAM-2-8B-r
- xLAM-2-32B-r
- xLAM-2-70B-r

`-fc-r` indicates function-calling reasoning variant. `r` indicates reasoning.

## APIGen-MT training data pipeline

Generates multi-turn agent-human interaction traces synthetically. The pipeline is the contribution; the model is the artifact.

## Headline numbers

- xLAM-2-70b-fc-r on τ-bench (multi-turn agentic): 56.2%, beats GPT-4o, approaches Claude 3.5 Sonnet.
- Smaller variants (3B/8B) competitive with much larger general-purpose chat models on function-calling.

## Relevance to 4 GB VRAM target

- xLAM-2-1B-r at Q4 ≈ 0.6 GB; fits trivially.
- xLAM-2-3B-r at Q4 ≈ 1.8 GB; fits comfortably with KV-cache headroom.
- xLAM-2-8B-r at Q4 ≈ 4.8 GB; over-budget for pure GPU; needs offload.

The 1B and 3B variants are the leading published candidates for "small open-weight model that does tool calling well." Pairs naturally with the [350M tool-calling result](slm-agentic-tool-calling.md) which shows the recipe scales down even further.

## See Also

- [Small models for agentic tool calling (350M paper)](slm-agentic-tool-calling.md)
- [ToolACE](toolace.md)
- [BFCL](../benchmarks/bfcl.md)
