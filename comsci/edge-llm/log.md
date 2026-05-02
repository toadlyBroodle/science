2026-05-02 INIT: edge-llm wiki scaffolded. Minimal (bpu-style) variant. Subdirs: models, techniques, architectures, runtimes, training, harnesses, benchmarks, analysis. Target ~40-50 pages.
2026-05-02 INGEST: AWQ (arXiv:2306.00978, MLSys 2024). Foundational reference. techniques/awq.md.
2026-05-02 INGEST: GPTQ (arXiv:2210.17323, ICLR 2023). Foundational reference. techniques/gptq.md.
2026-05-02 INGEST: KIVI (arXiv:2402.02750, ICML 2024). Foundational reference. techniques/kivi.md.
2026-05-02 INGEST: AQLM (arXiv:2401.06118, ICML 2024). Foundational reference. techniques/aqlm.md.
2026-05-02 INGEST: DeepSeek-R1 + distill family (arXiv:2501.12948, Nature 645 2025). models/deepseek-r1.md.
2026-05-02 INGEST: LFM2 (arXiv:2511.23404, Nov 2025). Liquid AI hybrid family with HW-in-the-loop NAS. models/lfm2.md.
2026-05-02 INGEST: Phi-4-Mini (arXiv:2503.01743, March 2025). MoLoRAs multimodal. models/phi-4-mini.md.
2026-05-02 INGEST: Gemma 3 (arXiv:2503.19786, March 2025). Local-vs-global attention rebalance for KV-cache. models/gemma-3.md.
2026-05-02 INGEST: Qwen3-Coder-Next (arXiv:2603.00729, Feb 2026). 80B/3B-active agentic-coder MoE. models/qwen3-coder-next.md.
2026-05-02 INGEST: EAGLE-3 (arXiv:2503.01840, NeurIPS 2025). Direct token prediction + multi-layer feature fusion. techniques/eagle-3.md.
2026-05-02 INGEST: SLMQuant (arXiv:2511.13023, Nov 2025). LLM quant doesn't transfer cleanly to SLMs. techniques/slmquant.md.
2026-05-02 INGEST: Saguaro / SSD (arXiv:2603.03251, ICLR 2026). Parallel speculation+verification, 30% over SD baseline. techniques/saguaro-ssd.md.
2026-05-02 INGEST: DDTree (arXiv:2604.12989, April 2026). Block-diffusion draft trees beat EAGLE-3. techniques/ddtree.md.
2026-05-02 INGEST: DASH-KV (arXiv:2604.19351, ACL 2026 Findings). O(N) attention via asymmetric hashing. techniques/dash-kv.md.
2026-05-02 INGEST: StructKV (arXiv:2604.06746, ACL 2026 Findings). Global In-Degree Centrality for KV pruning. techniques/structkv.md.
2026-05-02 INGEST: MoE-Spec (arXiv:2602.16052, Feb 2026). Expert budgeting for MoE spec decoding, +10-30% over EAGLE-3. techniques/moe-spec.md.
2026-05-02 INGEST: DALI (arXiv:2602.03495, Feb 2026). 0-1 integer optimization for MoE offload on local PCs. runtimes/dali-moe.md.
2026-05-02 INGEST: FlashMoE (arXiv:2601.17063, Jan 2026). SSD-based MoE caching with ML replacement, +51% hit rate. runtimes/flashmoe.md.
2026-05-02 INGEST: MoL CoT distillation (arXiv:2604.15701, EMNLP 2025/April 2026). Stepwise-attention transfer for small-model reasoning. training/mol-distillation.md.
2026-05-02 INGEST: Gemma 4/Phi-4/Qwen3 reasoning tradeoffs (arXiv:2604.07035, April 2026). Best: Gemma-4-E4B 0.675 / 14.9 GB. benchmarks/dense-vs-moe-reasoning-tradeoffs.md.
2026-05-02 INGEST: Small models for agentic tool calling (arXiv:2512.15943, AAAI 2026 Workshop). 350M-OPT fine-tuned hits 77.55% ToolBench, beats ToolLLaMA. training/slm-agentic-tool-calling.md.
2026-05-02 INGEST: HybriMoE (arXiv:2504.05897, DAC 2025). +1.33x prefill / +1.70x decode on KTransformers. raw only.
2026-05-02 INGEST: Terminal-Bench 2.0 (arXiv:2601.11868, Jan 2026). 89 hard CLI tasks. benchmarks/terminal-bench.md.
2026-05-02 INGEST: xLAM-2 (Salesforce, 2025). 1B-70B agentic open-weight family. training/xlam-2.md.
2026-05-02 INGEST: ToolACE (arXiv:2409.00920, ICLR 2025). 26,507-API self-evolution pipeline. training/toolace.md.
2026-05-02 INGEST: KTransformers (SOSP 2025). 4.62-19.74x prefill speedup for hybrid CPU-GPU MoE. runtimes/ktransformers.md.
2026-05-02 INGEST: SWE-Bench (arXiv:2310.06770, ICLR 2024). Closest proxy for agentic coding. benchmarks/swe-bench.md.
2026-05-02 INGEST: BFCL v3/v4 (Berkeley, ICML 2025). benchmarks/bfcl.md.
2026-05-02 INGEST: Aider polyglot benchmark. benchmarks/aider-polyglot.md.
2026-05-02 INGEST: LiveCodeBench (arXiv:2403.07974). Contamination-free coding eval. benchmarks/livecodebench.md.
2026-05-02 INGEST: llama.cpp / KTransformers / vLLM / Ollama-and-friends. runtimes/ pages.
2026-05-02 INGEST: Claude Code / aider / Cline-Continue-Goose harness pages. harnesses/.
2026-05-02 INGEST: Mamba-2/3 + Zamba2 hybrid architectures. architectures/mamba-ssm-hybrids.md.
2026-05-02 INGEST: MoE active-parameter architectures synthesis. architectures/moe-active-param.md.
2026-05-02 INGEST: SLM agentic tool calling (arXiv:2512.15943, AAAI 2026 Workshop). 350M-OPT fine-tuned hits 77.55% ToolBench. raw + training/.
2026-05-02 SYNTH: Phase 3 SOTA analyses written. analysis/four-gb-budget-math.md, quant-vs-capability-frontier.md, runtime-comparison.md, harness-comparison.md, spec-decoding-at-4gb.md.
2026-05-02 SYNTH: Phase 4 gap analysis written. analysis/missing-evals.md (8 gaps ranked), open-questions.md (10 contradictions).
2026-05-02 SYNTH: Phase 5 contribution roadmap written. analysis/contribution-roadmap.md (6 ranked paths), agentic-sft-recipe.md (Path 1 concrete plan), harness-eval-suite-design.md (Path 2 concrete plan).
2026-05-02 STATUS: All 5 build phases complete. 47 wiki articles total.
2026-05-02 LINT: 0 broken links, 0 orphans, 0 em-dashes (27 fixed via perl bulk replace), 0 forbidden hedging words. All 47 articles have Summary/Sources/See Also sections. Two raw extracts (hybrimoe, gemma-phi-qwen-tradeoffs) referenced inline rather than getting dedicated wiki pages: acceptable.
