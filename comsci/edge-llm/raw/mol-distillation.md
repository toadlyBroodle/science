# Mixture-of-Layers (MoL) Distillation with Stepwise Attention on Key Information

**Source:** arXiv:2604.15701 (https://arxiv.org/abs/2604.15701)
**Fetched:** 2026-05-02 via WebFetch
**Authors:** Yao Chen, Jiawei Sheng, Wenyuan Zhang, Tingwen Liu
**Venue:** EMNLP 2025
**Submitted:** 2026-04-17 (paper published at EMNLP 2025)

## Abstract / extracted content

CoT distillation framework that transfers the teacher's stepwise attention pattern over key information into the student. Observes that LMs exhibit progressive attention shifts toward key tokens during multi-step reasoning. Introduces a Mixture-of-Layers module enabling dynamic, learned alignment between teacher and student layer counts. Claimed first method to use stepwise attention as a distillation signal in CoT distillation.

## Key claims

- Stepwise attention transfer is a stronger distillation signal than final-output matching alone.
- Mixture-of-Layers module handles teacher-student layer-count mismatch via dynamic weighted alignment.
- Improvements across mathematical and commonsense reasoning datasets.

## Headline numbers

- Specific dataset numbers and teacher-student pairs not extracted from abstract page.

## Relevance to 4 GB edge target

Distillation is the practical lever for getting reasoning behavior into 4 GB-class models. MoL is one of the 2026 advances in *how* to distill reasoning specifically (vs generic SFT on traces). Useful complement to DeepSeek-R1-Distill recipes.
