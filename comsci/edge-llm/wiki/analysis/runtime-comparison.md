# Runtime Comparison for the 4 GB VRAM Target

> **Summary:** Eight runtimes considered (llama.cpp, vLLM, KTransformers, ExLlamaV2, MLX, Ollama, LM Studio, DALI/FlashMoE-class). For dense 1-4B models on 4 GB, **llama.cpp is the default; ExLlamaV2 is the throughput maximizer on NVIDIA**. For MoE on 4 GB, **the KTransformers-derived stack is the only viable path**.

**Sources:** [runtimes/llama-cpp.md](../runtimes/llama-cpp.md), [runtimes/ktransformers.md](../runtimes/ktransformers.md), [runtimes/vllm.md](../runtimes/vllm.md), [runtimes/dali-moe.md](../runtimes/dali-moe.md), [runtimes/flashmoe.md](../runtimes/flashmoe.md), [runtimes/ollama-and-friends.md](../runtimes/ollama-and-friends.md)

---

## Comparison matrix

| Runtime | Hardware | Quant breadth | Partial offload | Prefix cache | EAGLE-style SD | MoE offload | KV-quant |
|---|---|---|---|---|---|---|---|
| llama.cpp | All (CPU/CUDA/ROCm/Metal/Vulkan) | All GGUF + AWQ/GPTQ | ✅ `-ngl` | partial | partial (recent) | basic | partial |
| vLLM | CUDA, ROCm | AWQ/GPTQ/FP8/Marlin | none | ✅ first-class | ✅ first-class | none | partial |
| KTransformers | CUDA + AMX-CPU | GGUF + AWQ/GPTQ | ✅ MoE-aware | ✅ | partial | ✅ shared/routed split | partial |
| DALI | KTransformers-based | as KT | ✅ workload-aware | inherits KT | inherits | ✅ best-in-class | inherits |
| FlashMoE | KTransformers-based | as KT | ✅ + SSD tier | inherits | inherits | ✅ SSD-aware | inherits |
| ExLlamaV2 | CUDA only | EXL2/AWQ/GPTQ | none/limited | none | partial | none | partial |
| MLX | Apple Silicon only | MLX-native, 4/8-bit | unified mem | partial | partial | none | partial |
| Ollama | wraps llama.cpp | as llama.cpp | ✅ auto | as llama.cpp | inherits | inherits | inherits |

(✅ = first-class support; partial = available but limited; none = not yet.)

## Decision tree for the 4 GB target

```
Is the target an MoE model (Qwen3-Coder-Next, Gemma-4-26B-A4B)?
├─ Yes → KTransformers (substrate) + DALI (assignment) + FlashMoE (SSD if needed)
└─ No  → Is the GPU NVIDIA?
         ├─ Yes → ExLlamaV2 for max throughput; llama.cpp for portability
         └─ No  → llama.cpp (CPU/AMD/Apple/Vulkan) or MLX (Apple Silicon)

Is prefix caching the bottleneck (long system prompt + tools, agentic loop)?
├─ Yes → vLLM if it fits, llama.cpp with prompt caching if not
└─ No  → continue with above choice
```

## Throughput context (rough)

For Phi-4-mini at Q4 on a 4 GB consumer GPU (e.g., RTX 3050 4GB / GTX 1650 4GB at the low end; RTX 4060 Laptop 8GB at the high end of "laptop"):

| Runtime | Tokens/sec (rough) | Notes |
|---|---|---|
| llama.cpp Q4 -ngl all | 30-60 | Standard |
| ExLlamaV2 EXL2 4-bit | 45-90 | NVIDIA-only |
| vLLM AWQ-4 | 25-50 | Server-mode overhead |
| Ollama (wraps llama.cpp) | 25-55 | Slight overhead |
| llama.cpp partial offload (e.g., 7B Q4 with -ngl 25) | 5-15 | When weights exceed VRAM |

These are order-of-magnitude only; exact numbers depend on driver, kernel, batch shape, and KV cache pressure.

## Speculative decoding cross-section

| Runtime | Native EAGLE-3 | Native Medusa | Saguaro/SSD planned | DDTree planned |
|---|---|---|---|---|
| llama.cpp | partial (PRs landing) | partial | not yet | not yet |
| vLLM | ✅ | ✅ | reference impl in SGLang/vLLM | not yet |
| ExLlamaV2 | partial | partial | not yet | not yet |

For a 4 GB device wanting speculative decoding, vLLM is currently the most mature path; llama.cpp is catching up.

## See Also

- [4 GB budget math](four-gb-budget-math.md)
- [Spec decoding at 4 GB](spec-decoding-at-4gb.md)
- [Harness comparison](harness-comparison.md)
