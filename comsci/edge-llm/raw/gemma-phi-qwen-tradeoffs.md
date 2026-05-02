# Gemma 4, Phi-4, and Qwen3: Accuracy-Efficiency Tradeoffs in Dense and MoE Reasoning Models

**Source:** arXiv:2604.07035 (https://arxiv.org/abs/2604.07035)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Md Motaleb Hossen Manik, Ge Wang
**Submitted:** 2026-04-08

## Abstract / extracted content

Controlled empirical benchmark of 7 instruction-tuned reasoning models across dense and MoE designs:

- Gemma-4: E2B, E4B, 26B-A4B
- Phi-4: mini-reasoning, reasoning
- Qwen3: 8B, 30B-A3B

8,400 evaluations across ARC-Challenge, GSM8K, Math L1-3, TruthfulQA MC1.

## Key results

- Best overall: **Gemma-4-E4B with few-shot CoT, weighted accuracy 0.675, mean VRAM 14.9 GB.**
- Close: Gemma-4-26B-A4B at 0.663, mean VRAM 48.1 GB.
- Gemma dominated mathematical reasoning.
- Phi excelled on TruthfulQA.
- GSM8K showed dramatic prompt sensitivity.

## Central claim

"Sparse activation alone does not guarantee the best practical operating point." MoE designs do not automatically beat dense at reasoning tasks; architecture, prompt strategy, and task composition jointly determine the operating point.

## Relevance to 4 GB edge target

The 14.9 GB VRAM figure for the leader (Gemma-4-E4B) is well over a 4 GB budget at FP16. At Q4 it drops to ~ 4 GB weights, but KV cache remains. Below 4 GB the candidate is Gemma-4-E2B or Phi-4-mini-reasoning, which the paper covers but doesn't single out as winners. Important reference for the wiki's quant-vs-capability frontier.
