The K3V2 / softmax-error-floor finding in #21591 (NexusQuant, by @jagmarques) quantifies the K-side bitcount question that's open here, and complements @TheTom's q8_0-K asymmetric data with a different recovery path on the architecture-sensitive cases.

**Diminishing returns on K bitcount (NexusQuant, Mistral-7B):**

Going from 3-bit to 4-bit keys: +0.06pp PPL. Going from 4 to 5: rounding noise. The reported mechanism: K feeds softmax, so K-quantization noise propagates across all positions, but the softmax error floor is hit at ~3 bits; further K precision buys nothing. V is linearly combined so V noise stays proportional, and lower V bitcount remains cheap.

**Architecture-dependent layer-boundary breakage:**

NexusQuant reports the Qwen2.5-7B catastrophic break with symmetric KV quant recovers by **protecting the first and last 2 transformer layers at fp16**, leaving the rest of K at low bitcount. This is layer-position protection, distinct from @TheTom's token-position boundary-V in [layer-aware-v-compression.md](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md). Mistral and Phi-3 don't need the protection per their data.

Two implications for the asymmetric K/V configs this PR is being asked to support:

1. **K8 is likely over-allocated.** TheTom's q8_0-K / turbo3-V recovers Qwen2.5-7B from PPL 3,556 to 6.71, but per NexusQuant's K-bitcount curve the K side could drop to 3 bits at near-zero PPL cost and ~2.7x further K-side compression. Worth measuring asymmetric `tbq3_0`-K / `tbq3_0`-V (or `q3_K` / `tbq3_0`) against the K8 asymmetric baseline.

2. **Layer-boundary fp16 is a complementary lever.** On architectures where bulk-layer K3 still breaks, protecting layers 0-1 and N-1, N at fp16 may recover quality without raising bulk K bitcount across the rest of the stack. Cheaper at long context than q8_0 across all layers.

Refs:
- #21591 (NexusQuant findings, closed)
- Repo + numbers: https://github.com/jagmarques/nexusquant
