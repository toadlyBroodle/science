# xLAM-2: Salesforce Large Action Models

**Source:** Salesforce blog + HuggingFace cards. Original xLAM paper: arXiv:2409.03215.
**Fetched:** 2026-05-02 via WebSearch
**Repo:** github.com/SalesforceAIResearch/xLAM
**Models:** xLAM-2-1B-r, 3B-r, 8B-r, 32B-r, 70B-r (function-calling variants `-fc-r`)

## Summary

xLAM is Salesforce's family of large action models (LAMs) for AI agents, dense and MoE, 1B to 8x22B. xLAM-2 (2025) adds multi-turn conversation and tool use, trained via the APIGen-MT framework that synthesizes data through simulated agent-human interactions.

## Headline numbers

- xLAM-2-70b-fc-r: 56.2% on τ-bench (multi-turn agentic), beating GPT-4o, approaching Claude 3.5 Sonnet.
- Smaller variants (3B/8B) competitive with much larger general-purpose chat models on function-calling.

## Position

The 1B-3B-8B variants are the edge-relevant artifacts. xLAM-2-3B-r at Q4 ≈ 1.8 GB weights → fits comfortably in 4 GB VRAM. Released open-weight; hf.co/Salesforce.

## Relevance

xLAM-2 is the published reference recipe for training small models specifically for agentic tool use. APIGen-MT is the training-data synthesis pipeline a solo dev would either reuse or imitate.
