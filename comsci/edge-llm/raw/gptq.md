# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Source:** arXiv:2210.17323 (https://arxiv.org/abs/2210.17323)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**Venue:** ICLR 2023
**Submitted:** 2022-10-31; revised 2023-03-22

## Abstract / extracted content

One-shot weight quantization based on approximate second-order information (uses the Hessian of layer-wise reconstruction loss). Quantizes 175B-parameter GPT models in ~4 GPU hours to 3-4 bits per weight with negligible accuracy loss. Doubles compression vs prior one-shot methods. Enables 175B inference on a single GPU. Extreme quantization (2-bit, ternary) feasible with reasonable degradation.

## Key claims

- Approximate second-order information enables accurate one-shot PTQ.
- 3.25x inference speedup on A100, 4.5x on A6000 vs FP16.
- First practical method to fit 175B in single-GPU memory while preserving accuracy.

## Headline numbers

- 175B model quantized in ~4 GPU hours.
- 3-4 bit per weight with negligible loss.
- Inference: 3.25x (A100) / 4.5x (A6000) over FP16.
- 2-bit / ternary feasible with reasonable accuracy.
