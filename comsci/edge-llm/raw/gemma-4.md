# Gemma 4 family + QAT checkpoints (Google DeepMind, April-June 2026) — source extract

Extracted 2026-07-06 from:
- https://huggingface.co/blog/gemma4 (family launch, 2026-04-02)
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ (launch post)
- https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/ (QAT checkpoints, 2026-06-05)
- Secondary: runaihome.com QAT hardware writeup, androidauthority.com QAT coverage, spheron.network deploy writeup

## Family (released 2026-04-02, Apache 2.0)

| Model | Effective/active params | Total params | Context | Audio |
|---|---|---|---|---|
| E2B | 2.3B | 5.1B with embeddings | 128K | yes |
| E4B | 4.5B | 8B with embeddings | 128K | yes |
| 12B Unified | 11.95B dense, encoder-free | 12B | 256K | yes |
| 26B A4B (MoE) | 4B active | 26B | 256K | no |
| 31B dense | 31B | 31B | 256K | no |

All multimodal (image, text, video). 12B Unified is encoder-free: raw patches/waveforms projected directly.

### Architecture

- Alternating local sliding-window and global full-context attention layers; 512-token windows on smaller models, 1024 on larger. Standard RoPE on sliding layers, pruned RoPE on global layers.
- Per-Layer Embeddings (PLE): secondary embedding table feeding residual signals to every decoder layer (lets E2B/E4B hold effective params in fast memory while total embeddings sit in slow memory).
- Shared KV cache: last N layers reuse KV states from earlier layers.
- E2B/E4B derived via MatFormer-style elastic nesting (E-series "effective parameter" naming).

### Benchmarks (instruction-tuned, Google-published)

| Benchmark | 31B | 26B A4B | E4B | E2B |
|---|---|---|---|---|
| MMLU Pro | 85.2 | 82.6 | 69.4 | 60.0 |
| AIME 2026 (no tools) | 89.2 | 88.3 | 42.5 | - |
| GPQA Diamond | 84.3 | 82.3 | 58.6 | - |
| LiveCodeBench v6 | 80.0 | 77.1 | 52.0 | - |
| Codeforces ELO | 2150 | 1718 | 940 | - |
| MMMU Pro | 76.9 | 73.8 | 52.6 | - |
| MRCR v2 128K | 66.4 | 44.1 | - | - |

LMArena estimated: 31B = 1452; 26B A4B = 1441 with 4B active.

### Runtimes at launch

Transformers, llama.cpp, MLX, transformers.js, Mistral.rs, ONNX, WebGPU. E4B fits ~6 GB (BF16 effective-param footprint per HF blog).

## QAT checkpoints (released 2026-06-05)

- Quantization-aware training: quantization simulated during a fine-tuning pass; weights learn to compensate. Google states QAT quality beats standard PTQ baselines; no quantitative deltas published in the blog.
- Sizes: E2B, E4B, 12B, 26B A4B, 31B (blog explicitly shows E2B, E4B, 26B MoE).
- Formats: Q4_0 GGUF for desktop; custom mobile-specialized quant schema for edge; compressed-tensors for vLLM/SGLang.
- Memory claims: ~72% VRAM cut vs BF16 headline (secondary coverage); 31B dense w4a16 ~66% less VRAM (spheron); 26B A4B runs in ~15 GB (runaihome); E2B text-only without PLE under 1 GB (Google blog, verbatim).
- Tooling: llama.cpp, Ollama, LM Studio (desktop); LiteRT-LM (edge); Transformers.js (web); SGLang + vLLM (serving); MLX (Apple Silicon).
- Ollama shipped Gemma 4 QAT weights on 2026-06-05 (v0.30.x point release).
