# Local runtime wave, May-June 2026 (llama.cpp / Ollama / vLLM / MLX / LM Studio) — source extract

Extracted 2026-07-06 from:
- https://codersera.com/blog/local-ai-runtimes-may-2026-update/ (May 2026 roundup)
- https://www.promptquorum.com/local-llms/top-open-source-models-ollama (Ollama June 2026, v0.30.x)
- llama.cpp GitHub (PR #22673)

## llama.cpp (May 2026, builds b8607-b9196+)

- PR #22673: Multi-Token Prediction (MTP) speculative decoding for Qwen 3.6. ~2x generation throughput on the dense 27B. On the 35B-A3B MoE: no net speedup on RTX 3090 at batch=1 (expert-union overhead).
- b9196 (2026-05-18): Windows prebuilts for CUDA 13.1, Vulkan, HIP, SYCL.
- CUDA kernel fusion: RTX 4090 baseline 77 tok/s -> 96 tok/s (+24%).

## Ollama

- v0.23.0-0.23.4 and v0.24.0 (2026-05-03..14): Gemma 4 MTP speculative decoding on Mac via MLX runner, >2x on Gemma 4 31B coding tasks (v0.23.1); Codex App support via `ollama launch codex-app`; reworked MLX sampler; cached /api/show responses ~6.7x lower median latency (v0.24.0).
- v0.30.x (June 2026): broadened GGUF compatibility via llama.cpp beyond Apple Silicon; Gemma 4 QAT weights added 2026-06-05; MLX engine upgrade 2026-06-11; improved prompt/KV-cache reuse; expanded tensor-name support for nvfp4/mxfp8 quantized MoE tensors.

## vLLM

- v0.21.0 (2026-05-15): TOKENSPEED_MLA attention backend (DeepSeek-R1 / Kimi-K2.5 prefill+decode); speculative decoding respects reasoning budgets; breaking: C++20 required, Transformers v4 deprecated.
- EAGLE 3.1 (2026-05-26, landing in v0.22.0): fixes attention drift via FC normalization after each target hidden state; up to 2x longer acceptance length on long-context workloads.

## MLX

- 0.31.x (2026-05-24): M5 Neural Accelerator support, up to 4x time-to-first-token speedup; +19-27% generation speed vs M4 from bandwidth alone. Requires macOS 26.2+.

## LM Studio

- 0.4.13 (2026-05-13): mlx-engine v1.8.1; parallel vision predictions for Qwen 3.5/3.6, Gemma 4.
- 0.4.14 (late May): MTP stable; 1.5-3x tok/s depending on hardware/model.
