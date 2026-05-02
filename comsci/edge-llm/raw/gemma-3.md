# Gemma 3 Technical Report

**Source:** arXiv:2503.19786 (https://arxiv.org/abs/2503.19786)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Gemma Team, Google DeepMind (214+ authors including Aishwarya Kamath, Johan Ferret, Shreya Pathak; leadership Demis Hassabis, Jeff Dean, Oriol Vinyals)
**Submitted:** 2025-03-25

## Abstract / extracted content

Multimodal lightweight open models, 1B / 4B / 12B / 27B. Vision understanding, expanded language coverage, 128K+ context length. Architecture changes to control KV-cache memory at long context: increased ratio of local-attention to global-attention layers; shorter local-attention spans. Trained with knowledge distillation. Gemma3-4B-IT competitive with Gemma2-27B-IT; Gemma3-27B-IT comparable to Gemini-1.5-Pro.

## Key claims

- Local-vs-global attention ratio rebalanced specifically to constrain KV-cache memory growth at long context.
- Distillation training across all sizes.
- 4B-IT matches previous generation's 27B on benchmark suites.

## Variants

- 1 B
- 4 B
- 12 B
- 27 B

## Architecture: KV-cache mitigation

- Increased local : global attention layer ratio.
- Shortened local attention span.
- Net: reduced KV cache footprint at 128K context (specific reduction percentage not in extracted abstract).

## Headline numbers

- Specific benchmark scores not extracted from arXiv abstract page.
- Comparative claims: 4B-IT ≈ Gemma2-27B-IT; 27B-IT ≈ Gemini-1.5-Pro.

## Relevance to 4 GB edge target

Gemma3-4B at Q4 (~ 2.4 GB weights) plus a controlled KV cache makes Gemma 3 a leading dense candidate for the 4 GB target. The local-global rebalance is directly load-bearing for long-context agentic use.
