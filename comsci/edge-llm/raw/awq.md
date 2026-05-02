# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

**Source:** arXiv:2306.00978 (https://arxiv.org/abs/2306.00978)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
**Venue:** MLSys 2024 (Best Paper Award)
**Submitted:** 2023-06-01; last revised 2026-04-25
**Code:** https://github.com/mit-han-lab/llm-awq

## Abstract / extracted content

Hardware-friendly approach for LLM low-bit weight-only quantization. Identifies salient weights through activation distributions rather than weight magnitude. By protecting just 1% of weights (the salient channels) and applying per-channel scaling, AWQ reduces quantization error without backpropagation or reconstruction. Generalizes across language, code, math, and multi-modal benchmarks. TinyChat (companion runtime) achieves >3x speedup over HuggingFace FP16 on desktop and mobile GPUs; enables 70B Llama-2 on mobile-class GPUs.

## Key claims

- Not all weights are equally important; activation statistics (not weight values) determine salient channels.
- Scaling salient channels avoids hardware-inefficient mixed precision (preserves uniform low-bit storage).
- Avoids overfitting to a calibration set; better generalization than reconstruction-based methods (GPTQ).

## Headline numbers

- >3x speedup over FP16 baseline.
- 4-bit weight-only with negligible accuracy loss; competitive at 3-bit.
- 70B Llama-2 deployable on mobile GPUs via TinyChat.
