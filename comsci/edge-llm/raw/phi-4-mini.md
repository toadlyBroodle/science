# Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs

**Source:** arXiv:2503.01743 (https://arxiv.org/abs/2503.01743)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Microsoft team, 74 authors led by Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, et al.
**Submitted:** 2025-03-03; revised 2025-03-07 (v2)

## Abstract / extracted content

Two models: Phi-4-Mini (3.8B parameters, text) and Phi-4-Multimodal. Trained on high-quality web + synthetic data. Phi-4-Mini outperforms similar-sized open models and matches double-sized models on math and coding. Phi-4-Multimodal integrates text, vision, speech via Mixture-of-LoRAs with modality-specific routers.

## Key claims

- Phi-4-Mini matches 2x-larger models on complex reasoning (math, coding).
- Phi-4-Multimodal: first place on OpenASR leaderboard.
- Reasoning-enhanced variant on par with or exceeding DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B.
- Mixture-of-LoRAs (MoLoRAs) with modality-specific routers enables multi-modal inference without interference.

## Parameter counts

- Phi-4-Mini: 3.8 B
- Speech/audio LoRA component: 460 M

## Architecture changes vs Phi-3.5-Mini

- Vocabulary expanded to 200K tokens (multilingual).
- Group query attention.

## Training data

- High-quality web + synthetic.
- Curated synthetic recipe weighted toward math and coding.

## Headline numbers

Specific math/coding benchmark numbers not extracted from arXiv abstract page; refer to paper tables for HumanEval, GSM8K, MATH, MMLU, etc.
