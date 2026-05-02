# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

**Source:** arXiv:2501.12948 (https://arxiv.org/abs/2501.12948); Nature 645, 633-638 (2025)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Primary author:** DeepSeek-AI (200+ contributors; named: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, et al.)
**Submitted:** 2025-01-22; revised 2026-01-04

## Abstract / extracted content

Demonstrates that LLM reasoning capabilities can be developed via pure RL, removing the dependence on human-labeled reasoning trajectories. Emergent self-reflection and verification behaviors. Outperforms supervised baselines on math, coding, STEM. Reasoning ability transfers to smaller models via distillation.

## Key claims

- Pure RL beats SFT-on-human-traces for reasoning.
- Advanced patterns (self-check, backtracking) emerge without explicit annotation.
- Reasoning capabilities distill into smaller dense models.

## Distilled variants (well-documented externally; not in fetched abstract)

DeepSeek released distills built on Qwen and Llama bases:
- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Llama-70B

The 1.5B and 7B variants are the relevant edge-LLM targets.

## Headline numbers

Specific reasoning benchmark numbers not extracted from arXiv abstract page; see Nature paper or model cards on Hugging Face for AIME / MATH / GPQA / Codeforces scores.
