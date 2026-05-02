# Ollama, LM Studio, ExLlamaV2, MLX

> **Summary:** The next-tier runtimes after llama.cpp / vLLM / KTransformers. Each fills a niche: Ollama and LM Studio wrap llama.cpp with friendlier UX; ExLlamaV2 is the fastest CUDA-only path; MLX is Apple Silicon's first-class runtime.

**Sources:** project documentation; ecosystem references throughout the wiki.

---

## Ollama

llama.cpp + a model registry + an HTTP API + automatic VRAM/CPU layer-split. Default for "I just want to run a local model." Loses some llama.cpp tunability (specific quant flavor, fine-grained `-ngl` control) in exchange for simplicity. Good first runtime for testing model candidates; not the runtime to use for the final 4 GB-budget benchmark.

## LM Studio

GUI wrapper around llama.cpp / MLX. Useful for human exploration, irrelevant for automated agentic loops.

## ExLlamaV2

CUDA-only, the fastest path on NVIDIA for AWQ / GPTQ / EXL2-quantized models. For a 4 GB CUDA laptop GPU, ExLlamaV2 with EXL2 quantization can produce throughput numbers llama.cpp cannot match. Trades portability for speed: no Apple Silicon, no AMD, no CPU fallback.

## MLX

Apple's first-party ML framework with growing LLM support. On Apple Silicon laptops with unified memory, MLX models run with substantially less overhead than llama.cpp Metal backend on the same hardware. Quant support is narrower (4-bit, 8-bit, MLX-native formats); model coverage lags llama.cpp by weeks-to-months.

## Position chart for 4 GB target

| Runtime | Best for | Avoid for |
|---|---|---|
| **llama.cpp** | Default, hardware breadth, quant breadth, partial offload | Maximum throughput on NVIDIA |
| **ExLlamaV2** | NVIDIA throughput at small scale | Anything non-NVIDIA |
| **vLLM** | Prefix caching + native SD | Tiny VRAM (overhead) |
| **KTransformers** | MoE-on-consumer-PC (the canonical 4 GB MoE path) | Pure dense small models |
| **Ollama** | Quick experimentation | Reproducible benchmarks |
| **MLX** | Apple Silicon first | Anything non-Apple |
| **LM Studio** | Human exploration | Automated agentic loops |

## See Also

- [llama.cpp](llama-cpp.md)
- [vLLM](vllm.md)
- [KTransformers](ktransformers.md)
