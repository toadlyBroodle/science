# Edge-LLM Knowledge Base

> SOTA in small/edge LLMs, scoped to running an agentic coding harness on 4 GB VRAM. Last updated: 2026-07-06. 70 articles compiled from 54 source extracts (cutting-edge 2026 work weighted; foundational reference where needed).

## Goal

Compile what's needed to make an informed choice about (model × quant × runtime × harness) on a 4 GB VRAM laptop GPU, and identify where a solo dev with small budget can contribute. Target capability: as close to Claude Haiku 4.5 (or Sonnet 4.6) inside Claude Code as possible.

## Models

- [Qwen3-Coder-Next](models/qwen3-coder-next.md): 80B-total / 3B-active MoE for coding agents (Feb 2026). Trained on verifiable coding tasks with executable environments via mid-training + RL. The canonical 2026 reference for "agentic-coder MoE."
- [Gemma 4](models/gemma-4.md): Google DeepMind, April 2026; QAT checkpoints June 2026. E2B / E4B / 12B / 26B-A4B / 31B, Per-Layer Embeddings + shared KV cache. QAT cuts memory ~72%: E2B text-only under 1 GB. E4B (LiveCodeBench v6 52.0) is the new leading sub-5B-active candidate at 4 GB.
- [Gemma 3](models/gemma-3.md): Google DeepMind, March 2025 (superseded by Gemma 4). 1B / 4B / 12B / 27B. 128K+ context with rebalanced local-vs-global attention to control KV cache.
- [Phi-4-Mini](models/phi-4-mini.md): Microsoft, March 2025. 3.8B; Mixture-of-LoRAs multimodal. Reasoning variant rivals DeepSeek-R1-Distill-Qwen-7B and -Llama-8B.
- [DeepSeek-R1 and Distill family](models/deepseek-r1.md): Pure-RL reasoning training (Nature 2025). Distill-Qwen-1.5B / 7B and Distill-Llama-8B are the edge-relevant variants.
- [LFM2.5](models/lfm2-5.md): Liquid AI, May-June 2026. 8B-A1B on-device MoE (128K ctx, BFCL v3 64.79, Tau^2 Telecom 88.07, <6 GB quantized, 146-253 tok/s CPU decode) + 230M CPU-anywhere agent base (<400 MB at 4-bit).
- [LFM2](models/lfm2.md): Liquid AI, November 2025 (superseded by LFM2.5). 350M-8.3B family, hybrid (gated short conv + GQA), HW-in-the-loop NAS. 2x faster prefill/decode on CPU.
- [Nemotron-3-Nano-4B](models/nemotron-3-nano.md): NVIDIA, March 2026. 3.97B hybrid Mamba-2 + Transformer; LiveCodeBench 51.8, BFCL v3 61.1; Q4_K_M 2.9 GB. Ready-to-run 4 GB default with strong tool-calling.
- [Mellum2](models/mellum2.md): JetBrains, June 2026 (Apache-2.0). 12B-total / 2.5B-active code MoE; BFCL v3 66.3, LiveCodeBench v6 69.9 (Thinking). First open code MoE tuned for commodity-GPU inference.

## Architectures

- [Mamba / SSM and hybrid architectures](architectures/mamba-ssm-hybrids.md): **Mamba-3 (March 2026)** achieves Mamba-2 perplexity at half the state size. Mamba-2, Jamba, Zamba2. Hybrids interleave SSM layers (no KV cache) with shared attention. Structural advantage at long context on tight VRAM.
- [MoE active-parameter architectures](architectures/moe-active-param.md): Active-vs-total parameter math. Qwen3-MoE-A3B, Gemma-4-26B-A4B, OLMoE, Qwen3-Coder-Next. Wins on agentic coding *only* when training and serving stack are both designed for it.

## Techniques

### 2026 advances

- [Saguaro / SSD](techniques/saguaro-ssd.md): Speculative speculative decoding (ICLR 2026). 30% faster than optimized SD baselines; up to 5x over autoregressive.
- [DDTree](techniques/ddtree.md): Block-diffusion draft trees (April 2026). Beats EAGLE-3 autoregressive drafters.
- [DEER diffusion drafter](techniques/deer-diffusion-draft.md): Single-pass diffusion drafter (Dec 2025). Acceptance lengths up to 32 tokens vs ~10 for EAGLE-3.
- [VSD](techniques/vsd-variational-spec.md): Variational speculative decoding (Feb 2026). +9.6% over EAGLE-3, drop-in replacement.
- [DASH-KV](techniques/dash-kv.md): Asymmetric KV-cache hashing (ACL 2026 Findings). O(N²) → O(N) attention.
- [StructKV](techniques/structkv.md): Structure-aware KV compression (ACL 2026 Findings). Global In-Degree Centrality preserves cross-layer information hubs.
- [Expected Attention](techniques/expected-attention.md): Training-free KV compression (Oct 2025). Closed-form importance estimation; lives in NVIDIA KVPress.
- [PolyKV](techniques/polykv.md): Shared asymmetric KV pool across multi-agent scaffolds (April 2026). K8/V3 with cross-persona sharing.
- [Self-Indexing KVCache](techniques/self-indexing-kv.md): 1-bit sign-based KV (March 2026). Compression and sparse-attention index in one representation.
- [MoE-Spec](techniques/moe-spec.md): Expert budgeting for MoE speculative decoding (Feb 2026). +10-30% over EAGLE-3 on MoE.
- [SLMQuant](techniques/slmquant.md): Benchmarking SLM quantization (Nov 2025). LLM quant techniques don't transfer cleanly to small models.
- [FastTTS](techniques/fasttts.md): Memory-constrained test-time scaling for edge (ASPLOS 2026). 2.2x goodput, 38-68% latency reduction vs vLLM. **Direct hit on the 4 GB target.**
- [EAGLE-3](techniques/eagle-3.md): Speculative decoding via training-time test (NeurIPS 2025). 6.5x over standard, 1.4x over EAGLE-2.
- [HARP](techniques/harp.md): Learnable Hadamard-preconditioned rotations for extreme quant (May 2026). Rescues 2-bit (Llama 2 7B 2-bit PPL 7.23 vs RHT 8.22), making a 7B 4 GB-viable.

### Foundational references

- [AWQ](techniques/awq.md): Activation-aware weight quantization (MLSys 2024). Practitioner default for 4-bit.
- [GPTQ](techniques/gptq.md): One-shot Hessian-aware PTQ (ICLR 2023).
- [KIVI](techniques/kivi.md): Asymmetric 2-bit KV-cache quantization (ICML 2024).
- [AQLM](techniques/aqlm.md): Additive multi-codebook quantization (ICML 2024). Pareto-optimal under 3 bits.

## Runtimes

- [llama.cpp](runtimes/llama-cpp.md): Universal CPU/GPU inference, GGUF, every quant flavor. The default. Partial offload via `-ngl` is the load-bearing 4 GB feature. **April 2026 wave** brought tensor parallelism, 1-bit, Hexagon NPU backend; **May 2026** added MTP speculative decoding (~2x on dense Qwen 3.6 27B, no gain on MoE at batch=1).
- [KTransformers](runtimes/ktransformers.md): Tsinghua, SOSP 2025. Heterogeneous MoE inference. 4.62-19.74x prefill speedup. Substrate for DALI/HybriMoE/FlashMoE.
- [vLLM](runtimes/vllm.md): Server-side throughput. PagedAttention, prefix caching, native EAGLE-style SD.
- [DALI](runtimes/dali-moe.md): Workload-aware MoE offloading for local PCs (Feb 2026). 0-1 integer optimization for expert-to-device assignment.
- [FlashMoE](runtimes/flashmoe.md): SSD-based MoE caching with ML cache replacement (Jan 2026). +51% hit rate vs LRU/LFU.
- [ExecuTorch](runtimes/executorch.md): PyTorch's edge runtime, GA October 2025. Wins on mobile and Snapdragon NPU; loses to llama.cpp on laptop NVIDIA dGPU.
- [ExLlamaV3 / EXL3](runtimes/exllamav3.md): turboderp, through June 2026. Arbitrary bits-per-weight QTIP-variant quant (1.6 bpw 70B coherent) + 2-8 bit KV quant + consumer tensor/expert parallel. Strongest at the extreme-low-VRAM end.
- [Ollama / LM Studio / ExLlamaV2 / MLX](runtimes/ollama-and-friends.md): Next-tier runtimes for specific niches. May-June 2026: Ollama v0.24-0.30.x (Gemma 4 QAT + MTP, KV-cache reuse), LM Studio MTP stable, MLX M5 Neural Accelerators.

## Training

- [Mixture-of-Layers (MoL) CoT distillation](training/mol-distillation.md): Stepwise-attention transfer for reasoning (EMNLP 2025).
- [Small models for agentic tool calling (350M)](training/slm-agentic-tool-calling.md): Jhandi et al., AAAI 2026. **One-epoch SFT on a 350M base hits 77.55% ToolBench, beating ToolLLaMA's 30%.** Direct evidence that targeted SFT at extreme small scale closes the gap.
- [Agentic RL for coding (2025-2026)](training/agentic-rl-coding.md): DeepSWE + Self-Play SWE-RL + SWE-RM + SWE-TRACE. The composed solo-dev recipe for RL-tuning a 3B-7B coder.
- [SWE-HERO](training/swe-hero.md): NVIDIA-affiliated, April 2026. Distills a 480B teacher into Qwen2.5-Coder 7B/14B/32B; 32B hits 62.2% SWE-bench Verified, 7B 52.7%. Open 300k+13k trajectories.
- [SWE-TRACE](training/swe-trace.md): April 2026. Rubric PRM + heuristic test-time scaling; Qwen3-4B to 40.7%, 30B-A3B to 71.2% SWE-bench Verified; cascaded SFT cuts tokens ~21%.
- [xLAM-2](training/xlam-2.md): Salesforce, 2025. Open-weight LAMs 1B-70B with APIGen-MT data pipeline. xLAM-2-70b-fc-r approaches Claude 3.5 Sonnet on τ-bench.
- [ToolACE](training/toolace.md): Liu et al., ICLR 2025. Self-evolution data pipeline; 26,507-API pool. 8B model trained on ToolACE data hits BFCL SOTA.

## Harnesses

- [Claude Code](harnesses/claude-code.md): The wiki's *target* harness; the goal is to come as close as possible on 4 GB.
- [aider](harnesses/aider.md): Open-source CLI, multi-provider, ships own benchmark with edit-conformance metrics.
- [Cline / Continue / Goose](harnesses/cline-continue-goose.md): Open-source IDE/CLI agentic harnesses. BYOM. Cline + local Qwen3-Coder/Phi-4-mini fine-tuned for Cline's format = most direct local Claude Code analog.
- [OpenHands (local-model support)](harnesses/openhands.md): May 2026 update adds saved local-LLM profiles, /model switching, sub-agent delegation, and a prompt-serialization tool-call fallback that props up small models lacking native function calling.

## Benchmarks

- [SWE-Bench (Original / Lite / Verified / Pro)](benchmarks/swe-bench.md): Real GitHub-issue resolution. The closest proxy for agentic coding capability.
- [SWE-rebench, Multi-SWE-bench, SWE-Universe](benchmarks/swe-rebench-multi-swe.md): 2025-2026 successors. **Decontaminated, multilingual, scaled to millions.** Prefer SWE-rebench for honest 4 GB-class numbers.
- [LongCodeBench](benchmarks/longcodebench.md): Coding eval at 32K-1M context (May 2025). Claude 3.5 Sonnet drops 29% → 3% from 32K to 256K. Long-context is broken even for frontier; route around it via filesystem tools.
- [BFCL (v3 / v4)](benchmarks/bfcl.md): Berkeley Function Calling Leaderboard. Multi-turn, state-based. The most diagnostic small-model agentic benchmark. GLM-4.5 leads v3 at 0.778.
- [Aider polyglot](benchmarks/aider-polyglot.md): 225 Exercism exercises, 6 languages, two-attempt protocol. Tests both code-correctness and structured-edit-format adherence. 2026 leaders: Claude Opus 4.5 (89.4%), GPT-5 (88.0%), DeepSeek V3.2-Exp (74.2%).
- [SWE-Chain](benchmarks/swe-chain.md) (May 2026): Chained release-level package upgrades; 155 transitions, errors compound across chains. Best frontier result 60.8% resolving (Claude-Opus-4.7 + Claude Code). No small-model numbers yet: open eval gap.
- [Terminal-Bench (1.0 / 2.0)](benchmarks/terminal-bench.md): Stanford + Laude, January 2026. 89 hard CLI tasks. Frontier models <65%.
- [LiveCodeBench](benchmarks/livecodebench.md): Holistic, contamination-free coding eval. Time-segmented. Tests self-repair, execution, test-output prediction.
- [Dense vs MoE reasoning tradeoffs (Manik & Wang)](benchmarks/dense-vs-moe-reasoning-tradeoffs.md) (April 2026): Best overall: Gemma-4-E4B (0.675 weighted accuracy / 14.9 GB VRAM). **Sparse activation alone does not guarantee the best operating point.**

## Analysis

### SOTA synthesis (Phase 3)

- [4 GB VRAM budget math](analysis/four-gb-budget-math.md): VRAM accounting for (model × quant × context length). Weights + KV cache + activations + workspace. **The 4 GB ceiling is roughly a 4 B-parameter ceiling at Q4 with compressed KV cache.** Hybrid SSM/attention models get a structural win at long context.
- [Long context via filesystem (not attention)](analysis/long-context-via-filesystem.md): The March 2026 thesis that filesystem-tool agents beat published long-context SOTA by 17.3% over corpora up to 3T tokens. **Validates the entire 4 GB-VRAM bet.**
- [Quant-vs-capability frontier](analysis/quant-vs-capability-frontier.md): Per-quant quality drops, with SLM-specific corrections per [SLMQuant](techniques/slmquant.md). Tool-call format conformance is the most quant-sensitive task.
- [Runtime comparison](analysis/runtime-comparison.md): Decision tree across llama.cpp / vLLM / KTransformers / ExLlamaV2 / MLX / Ollama. **For dense 1-4B on 4 GB: llama.cpp default, ExLlamaV2 max throughput. For MoE on 4 GB: KTransformers stack only.**
- [Harness comparison](analysis/harness-comparison.md): Tool-call format taxonomy across Claude Code / aider / Cline / Continue / Goose. Format conformance dominates failure modes at SLM scale.
- [Speculative decoding at 4 GB](analysis/spec-decoding-at-4gb.md): Three viable patterns; EAGLE-style auxiliary head (default), self-speculative (Medusa/lookahead), block-diffusion drafter (DDTree, emerging). Saguaro composes; MoE-Spec required for MoE targets.
- [Laptop GPU energy and thermal](analysis/laptop-gpu-energy-thermal.md): Watt Counts (April 2026) + LLM Inference at the Edge (March 2026). **RTX 4050 sustains 131.7 tok/s at 34.1 W**; the hard anchor for laptop wiki claims.

### Gap analysis (Phase 4)

- [Missing evals](analysis/missing-evals.md): Eight gaps in the published literature; ranked by leverage and cost. Headline gap: **no published (model × quant × runtime × harness) matrix at 4 GB envelope on agentic-coding metrics.**
- [Open questions](analysis/open-questions.md): Ten unresolved claims and source contradictions, with status and resolution paths.

### Contribution roadmap (Phase 5)

- [Contribution roadmap](analysis/contribution-roadmap.md): Six ranked solo-dev paths. **Top two: (1) agentic-format SFT dataset + recipe, (2) 4 GB-envelope reproducible eval suite.** Both are the "papers Anthropic / DeepSeek would not write themselves"; high community leverage at solo-dev cost.
- [Agentic SFT recipe](analysis/agentic-sft-recipe.md): Concrete plan for Path 1. Replay 5K Claude Code sessions on OSS issues; re-format to Cline / aider; SFT a 1-3B base via Unsloth + LoRA. Cost: $300-500. Time: 4-8 weeks.
- [Harness eval suite design](analysis/harness-eval-suite-design.md): Concrete plan for Path 2. Docker-packaged eval matrix. v1: 45 cells × $15-35 = $700-1,500. The infrastructure piece every other contribution depends on.

## Conventions

See [`../CLAUDE.md`](../CLAUDE.md) for ingest, query, and lint workflows. See [`../log.md`](../log.md) for operation history.
