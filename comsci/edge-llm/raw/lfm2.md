# LFM2 Technical Report

**Source:** arXiv:2511.23404 (https://arxiv.org/abs/2511.23404)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Liquid AI team (33 authors): Alexander Amini, Anna Banaszak, Harold Benoit, Arthur Böök, Tarek Dakhran, Song Duong, Alfred Eng, Fernando Fernandes, Marc Härkönen, Anne Harrington, Ramin Hasani, Saniya Karwa, Yuri Khrustalev, Maxime Labonne, Mathias Lechner, Valentine Lechner, Simon Lee, Zetian Li, Noel Loo, Jacob Marks, Edoardo Mosca, Samuel J. Paech, Paul Pak, Rom N. Parnichkun, Alex Quach, Ryan Rogers, Daniela Rus, Nayan Saxena, Bettina Schlager, Tim Seyde, Jimmy T.H. Smith, Aditya Tadimeti, Neehal Tumma
**Submitted:** 2025-11-28

## Abstract / extracted content

Liquid Foundation Models v2: family for efficient on-device deployment, 350M to 8.3B parameters. Hardware-in-the-loop architecture search optimizing for latency and memory on edge targets. Hybrid backbone: gated short convolutions + grouped query attention. Three-stage post-training: SFT, length-normalized preference optimization, model merging. Pre-trained on 10-12T tokens with tempered, decoupled top-K knowledge distillation.

## Key claims

- Up to 2x faster prefill and decode on CPUs vs similarly-sized models.
- 32K context length across all variants.
- LFM2-2.6B: 79.56% IFEval, 82.41% GSM8K.

## Variants

- Text models: 350M to 8.3B parameters.
- LFM2-VL (vision), LFM2-Audio (speech), LFM2-ColBERT (retrieval).
- Real-time speech-to-speech.
- Open-weight; deployment packages for ExecuTorch, llama.cpp, vLLM.

## Architecture innovations

- Hardware-in-the-loop architecture search.
- Hybrid: gated short convolutions + grouped query attention (similar to Hymba/Zamba family).
- Tempered, decoupled top-K knowledge distillation.
