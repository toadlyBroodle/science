# ToolACE: Winning the Points of LLM Function Calling

**Source:** arXiv:2409.00920 (https://arxiv.org/abs/2409.00920)
**Fetched:** 2026-05-02 via WebSearch
**Venue:** ICLR 2025
**Submitted:** 2024-09-02; revised 2025-07-25
**Models:** hf.co/Team-ACE

## Summary

Automated data-generation pipeline for LLM function calling. Self-evolution synthesis curates a 26,507-API pool. Multi-agent interactive system generates diverse dialogs. Dual-layer verification (rule-based + model-based).

## Headline numbers

- 8B model trained on ToolACE-synthesized data: SOTA on Berkeley Function Calling Leaderboard (BFCL), rivals GPT-4 family.

## Position

ToolACE is the data pipeline; xLAM-2 is the model family using a similar approach. Together they define how to *generate* the SFT data needed to make small models tool-call competently. Paired with the [350M agentic tool-calling paper](slm-agentic-tool-calling.md), the message is unambiguous: targeted SFT on synthesized tool-call data closes the gap.
