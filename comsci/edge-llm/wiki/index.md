# Edge-LLM Knowledge Base

> SOTA in small/edge LLMs, scoped to running an agentic coding harness on 4 GB VRAM. Last updated: 2026-05-02. 47 articles compiled from 30 source extracts (cutting-edge 2026 work weighted; foundational reference where needed). All 5 build phases complete (scaffold, seed ingest, breadth ingest, SOTA synthesis, gap analysis, contribution roadmap).

## Goal

Compile what's needed to make an informed choice about (model × quant × runtime × harness) on a 4 GB VRAM laptop GPU, and identify where a solo dev with small budget can contribute. Target capability: as close to Claude Haiku 4.5 (or Sonnet 4.6) inside Claude Code as possible.

## Models

- [Qwen3-Coder-Next](models/qwen3-coder-next.md): 80B-total / 3B-active MoE for coding agents (Feb 2026). Trained on verifiable coding tasks with executable environments via mid-training + RL. The canonical 2026 reference for "agentic-coder MoE."
- [Gemma 3](models/gemma-3.md): Google DeepMind, March 2025. 1B / 4B / 12B / 27B. 128K+ context with rebalanced local-vs-global attention to control KV cache. Gemma 3-4B is the leading dense candidate at 4 GB.
- [Phi-4-Mini](models/phi-4-mini.md): Microsoft, March 2025. 3.8B; Mixture-of-LoRAs multimodal. Reasoning variant rivals DeepSeek-R1-Distill-Qwen-7B and -Llama-8B.
- [DeepSeek-R1 and Distill family](models/deepseek-r1.md): Pure-RL reasoning training (Nature 2025). Distill-Qwen-1.5B / 7B and Distill-Llama-8B are the edge-relevant variants.
- [LFM2](models/lfm2.md): Liquid AI, November 2025. 350M-8.3B family, hybrid (gated short conv + GQA), HW-in-the-loop NAS. 2x faster prefill/decode on CPU.

## Architectures

- [Mamba / SSM and hybrid architectures](architectures/mamba-ssm-hybrids.md): Mamba-3 (ICLR 2026), Mamba-2, Jamba, Zamba2. Hybrids interleave SSM layers (no KV cache) with shared attention. Structural advantage at long context on tight VRAM.
- [MoE active-parameter architectures](architectures/moe-active-param.md): Active-vs-total parameter math. Qwen3-MoE-A3B, Gemma-4-26B-A4B, OLMoE, Qwen3-Coder-Next. Wins on agentic coding *only* when training and serving stack are both designed for it.

## Techniques

### 2026 advances

- [Saguaro / SSD](techniques/saguaro-ssd.md): Speculative speculative decoding (ICLR 2026). 30% faster than optimized SD baselines; up to 5x over autoregressive.
- [DDTree](techniques/ddtree.md): Block-diffusion draft trees (April 2026). Beats EAGLE-3 autoregressive drafters.
- [DASH-KV](techniques/dash-kv.md): Asymmetric KV-cache hashing (ACL 2026 Findings). O(N²) → O(N) attention.
- [StructKV](techniques/structkv.md): Structure-aware KV compression (ACL 2026 Findings). Global In-Degree Centrality preserves cross-layer information hubs.
- [MoE-Spec](techniques/moe-spec.md): Expert budgeting for MoE speculative decoding (Feb 2026). +10-30% over EAGLE-3 on MoE.
- [SLMQuant](techniques/slmquant.md): Benchmarking SLM quantization (Nov 2025). LLM quant techniques don't transfer cleanly to small models.
- [EAGLE-3](techniques/eagle-3.md): Speculative decoding via training-time test (NeurIPS 2025). 6.5x over standard, 1.4x over EAGLE-2.

### Foundational references

- [AWQ](techniques/awq.md): Activation-aware weight quantization (MLSys 2024). Practitioner default for 4-bit.
- [GPTQ](techniques/gptq.md): One-shot Hessian-aware PTQ (ICLR 2023).
- [KIVI](techniques/kivi.md): Asymmetric 2-bit KV-cache quantization (ICML 2024).
- [AQLM](techniques/aqlm.md): Additive multi-codebook quantization (ICML 2024). Pareto-optimal under 3 bits.

## Runtimes

- [llama.cpp](runtimes/llama-cpp.md): Universal CPU/GPU inference, GGUF, every quant flavor. The default. Partial offload via `-ngl` is the load-bearing 4 GB feature.
- [KTransformers](runtimes/ktransformers.md): Tsinghua, SOSP 2025. Heterogeneous MoE inference. 4.62-19.74x prefill speedup. Substrate for DALI/HybriMoE/FlashMoE.
- [vLLM](runtimes/vllm.md): Server-side throughput. PagedAttention, prefix caching, native EAGLE-style SD.
- [DALI](runtimes/dali-moe.md): Workload-aware MoE offloading for local PCs (Feb 2026). 0-1 integer optimization for expert-to-device assignment.
- [FlashMoE](runtimes/flashmoe.md): SSD-based MoE caching with ML cache replacement (Jan 2026). +51% hit rate vs LRU/LFU.
- [Ollama / LM Studio / ExLlamaV2 / MLX](runtimes/ollama-and-friends.md): Next-tier runtimes for specific niches.

## Training

- [Mixture-of-Layers (MoL) CoT distillation](training/mol-distillation.md): Stepwise-attention transfer for reasoning (EMNLP 2025).
- [Small models for agentic tool calling (350M)](training/slm-agentic-tool-calling.md): Jhandi et al., AAAI 2026. **One-epoch SFT on a 350M base hits 77.55% ToolBench, beating ToolLLaMA's 30%. Direct evidence that targeted SFT at extreme small scale closes the gap.**
- [xLAM-2](training/xlam-2.md): Salesforce, 2025. Open-weight LAMs 1B-70B with APIGen-MT data pipeline. xLAM-2-70b-fc-r approaches Claude 3.5 Sonnet on τ-bench.
- [ToolACE](training/toolace.md): Liu et al., ICLR 2025. Self-evolution data pipeline; 26,507-API pool. 8B model trained on ToolACE data hits BFCL SOTA.

## Harnesses

- [Claude Code](harnesses/claude-code.md): The wiki's *target* harness; the goal is to come as close as possible on 4 GB.
- [aider](harnesses/aider.md): Open-source CLI, multi-provider, ships own benchmark with edit-conformance metrics.
- [Cline / Continue / Goose](harnesses/cline-continue-goose.md): Open-source IDE/CLI agentic harnesses. BYOM. Cline + local Qwen3-Coder/Phi-4-mini fine-tuned for Cline's format = most direct local Claude Code analog.

## Benchmarks

- [SWE-Bench (Original / Lite / Verified / Pro)](benchmarks/swe-bench.md): Real GitHub-issue resolution. The closest proxy for agentic coding capability.
- [BFCL (v3 / v4)](benchmarks/bfcl.md): Berkeley Function Calling Leaderboard. Multi-turn, state-based. The most diagnostic small-model agentic benchmark. GLM-4.5 leads v3 at 0.778.
- [Aider polyglot](benchmarks/aider-polyglot.md): 225 Exercism exercises, 6 languages, two-attempt protocol. Tests both code-correctness and structured-edit-format adherence. 2026 leaders: Claude Opus 4.5 (89.4%), GPT-5 (88.0%), DeepSeek V3.2-Exp (74.2%).
- [Terminal-Bench (1.0 / 2.0)](benchmarks/terminal-bench.md): Stanford + Laude, January 2026. 89 hard CLI tasks. Frontier models <65%.
- [LiveCodeBench](benchmarks/livecodebench.md): Holistic, contamination-free coding eval. Time-segmented. Tests self-repair, execution, test-output prediction.
- [Dense vs MoE reasoning tradeoffs (Manik & Wang)](benchmarks/dense-vs-moe-reasoning-tradeoffs.md) (April 2026): Best overall: Gemma-4-E4B (0.675 weighted accuracy / 14.9 GB VRAM). **Sparse activation alone does not guarantee the best operating point.**

## Analysis

### SOTA synthesis (Phase 3)

- [4 GB VRAM budget math](analysis/four-gb-budget-math.md): VRAM accounting for (model × quant × context length). Weights + KV cache + activations + workspace. **The 4 GB ceiling is roughly a 4 B-parameter ceiling at Q4 with compressed KV cache.** Hybrid SSM/attention models get a structural win at long context.
- [Quant-vs-capability frontier](analysis/quant-vs-capability-frontier.md): Per-quant quality drops, with SLM-specific corrections per [SLMQuant](techniques/slmquant.md). Tool-call format conformance is the most quant-sensitive task.
- [Runtime comparison](analysis/runtime-comparison.md): Decision tree across llama.cpp / vLLM / KTransformers / ExLlamaV2 / MLX / Ollama. **For dense 1-4B on 4 GB: llama.cpp default, ExLlamaV2 max throughput. For MoE on 4 GB: KTransformers stack only.**
- [Harness comparison](analysis/harness-comparison.md): Tool-call format taxonomy across Claude Code / aider / Cline / Continue / Goose. Format conformance dominates failure modes at SLM scale.
- [Speculative decoding at 4 GB](analysis/spec-decoding-at-4gb.md): Three viable patterns; EAGLE-style auxiliary head (default), self-speculative (Medusa/lookahead), block-diffusion drafter (DDTree, emerging). Saguaro composes; MoE-Spec required for MoE targets.

### Gap analysis (Phase 4)

- [Missing evals](analysis/missing-evals.md): Eight gaps in the published literature; ranked by leverage and cost. Headline gap: **no published (model × quant × runtime × harness) matrix at 4 GB envelope on agentic-coding metrics.**
- [Open questions](analysis/open-questions.md): Ten unresolved claims and source contradictions, with status and resolution paths.

### Contribution roadmap (Phase 5)

- [Contribution roadmap](analysis/contribution-roadmap.md): Six ranked solo-dev paths. **Top two: (1) agentic-format SFT dataset + recipe, (2) 4 GB-envelope reproducible eval suite.** Both are the "papers Anthropic / DeepSeek would not write themselves"; high community leverage at solo-dev cost.
- [Agentic SFT recipe](analysis/agentic-sft-recipe.md): Concrete plan for Path 1. Replay 5K Claude Code sessions on OSS issues; re-format to Cline / aider; SFT a 1-3B base via Unsloth + LoRA. Cost: $300-500. Time: 4-8 weeks.
- [Harness eval suite design](analysis/harness-eval-suite-design.md): Concrete plan for Path 2. Docker-packaged eval matrix. v1: 45 cells × $15-35 = $700-1,500. The infrastructure piece every other contribution depends on.

## Conventions

See [`../CLAUDE.md`](../CLAUDE.md) for ingest, query, and lint workflows. See [`../log.md`](../log.md) for operation history.
