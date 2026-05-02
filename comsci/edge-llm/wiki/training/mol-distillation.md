# Mixture-of-Layers (MoL) CoT Distillation with Stepwise Attention

> **Summary:** Chen, Sheng, Zhang, Liu; EMNLP 2025 (arXiv:2604.15701, paper revised April 2026). CoT distillation framework that transfers the teacher's stepwise attention pattern over key information into the student. First method to use stepwise attention as a distillation signal. Mixture-of-Layers module dynamically aligns mismatched teacher and student layer counts. Improvements across mathematical and commonsense reasoning datasets.

**Sources:** [raw/mol-distillation.md](../../raw/mol-distillation.md), [raw/deepseek-r1.md](../../raw/deepseek-r1.md)

---

## What's distilled

Standard CoT distillation: train the student on teacher-generated reasoning traces, possibly with logit-matching. MoL adds a third signal: the *attention distribution over input tokens at each reasoning step*. The observation is that LMs progressively narrow attention onto the few input tokens carrying decisive information; transferring that pattern is a stronger reasoning signal than final-output matching.

## The Mixture-of-Layers module

Teacher and student have different layer counts. MoL learns a soft assignment from each student layer to a weighted combination of teacher layers, allowing layer-by-layer attention alignment without forcing equal depth.

## Position vs DeepSeek-R1 distill recipe

[DeepSeek-R1's original distill recipe](../models/deepseek-r1.md) uses generated reasoning traces as SFT data on dense bases (Qwen, Llama). It is a strong but conservative baseline; it does not exploit teacher attention patterns. MoL is one of the 2026 papers exploring whether the original recipe leaves capability on the table; HEAL (arXiv:2603.10359, March 2026) is another.

## Headline numbers

- Specific datasets and teacher-student pairs not extracted from abstract.
- Improvements claimed across "multiple mathematical and commonsense reasoning datasets."

## Relevance to 4 GB VRAM target

Distillation is the practical lever for getting frontier-level reasoning into a 1-4 B model that fits the budget. MoL is the most promising 2026 method for small-model reasoning distillation. Likely candidate for replication in a solo-dev contribution path (see [Phase 5 contribution roadmap](../analysis/contribution-roadmap.md), pending).

## See Also

- [DeepSeek-R1 and Distill family](../models/deepseek-r1.md)
- [Phi-4-Mini](../models/phi-4-mini.md)
- [Contribution roadmap](../analysis/contribution-roadmap.md) (pending Phase 5)
