# GPTQ

> **Summary:** Frantar, Ashkboos, Hoefler, Alistarh; ICLR 2023. One-shot weight quantization using approximate second-order (Hessian) information. Quantizes 175B models in ~4 GPU hours to 3-4 bits with negligible accuracy loss. Foundational PTQ method; the basis for AutoGPTQ, ExLlama, and most early 4-bit LLM deployments.

**Sources:** [raw/gptq.md](../../raw/gptq.md), [raw/slmquant.md](../../raw/slmquant.md)

---

## Method (one paragraph)

Layer-wise reconstruction with Hessian-aware error compensation: quantize weights one column at a time, propagate the rounding error to the not-yet-quantized columns weighted by the inverse Hessian. Uses a small calibration set (typically 128 sequences).

## Headline numbers

- 175B model quantized in ~4 GPU hours.
- 3-4 bit per weight, negligible accuracy loss.
- 3.25x inference speedup on A100, 4.5x on A6000 vs FP16.

## Position vs AWQ

GPTQ does iterative reconstruction (slower calibration, can overfit calibration set); [AWQ](awq.md) does activation-aware scaling without backprop (faster, often equivalent quality). On 7B+ targets they are roughly tied; below that scale, [SLMQuant](slmquant.md) suggests both can degrade unexpectedly.

## See Also

- [AWQ](awq.md)
- [AQLM](aqlm.md)
- [SLMQuant](slmquant.md)
