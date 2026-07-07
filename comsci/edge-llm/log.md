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
2026-05-03 INGEST: 13 new 2026 source extracts (long-context-coding-agents 2603.20432, fasttts 2509.00195, mamba-3 2603.15569, deer-diffusion-draft 2512.15176, vsd-variational-spec 2602.05774, expected-attention-kv 2510.00636, polykv 2604.24971, self-indexing-kv 2603.14224, agentic-rl-recipes-2026, swe-bench-evolution-2026, longcodebench, executorch, laptop-energy-thermal-2026).
2026-05-03 INGEST: 13 new wiki pages (analysis/long-context-via-filesystem.md, analysis/laptop-gpu-energy-thermal.md, techniques/fasttts, deer-diffusion-draft, vsd-variational-spec, expected-attention, polykv, self-indexing-kv, training/agentic-rl-coding, benchmarks/swe-rebench-multi-swe, benchmarks/longcodebench, runtimes/executorch). Updates to architectures/mamba-ssm-hybrids (Mamba-3 + DUET detail), runtimes/llama-cpp (April 2026 wave), index.md.
2026-05-03 LINT: 0 broken links, 0 orphans, 0 em-dashes. 60 articles total, 43 raw extracts.
2026-05-03 SYNTH: contribution-roadmap.md updated for May 2026 ingest. Path 1 expanded SFT+RL ($800-1500, 8-12 wk). Path 2 anchored on SWE-rebench v2 + LongCodeBench + Watt Counts energy. Path 4 adds VSD/DEER drafter candidates. New Path 7 (harness-side filesystem-tool optimization, zero compute). New Path 8 (PolyKV vLLM integration). Sequencing recommendation revised. New top-3: Path 2, Path 7-or-4, Path 1.
2026-05-17 CONTRIB: KIVI-in-llama.cpp MVP drafted (contributions/kivi-llamacpp/SPEC.md + TODO.md + issue-proposal.md). Pre-PR discovery surfaced ggml-org/llama.cpp#21089 (TurboQuant CPU KV, OPEN), #21551 (existing-quant asymmetric exploration, OPEN draft), #21591 (NexusQuant K3V2 finding, CLOSED), and recent upstream KV rotation work. Conclusion: drafted KIVI MVP duplicates #21089's structural plumbing and targets the wrong bitcount (Q4 vs Q3). Plan paused; posted comment on #21089 at 2026-05-16T23:01:47Z surfacing #21591's K3 softmax-error-floor curve + layer-position fp16 boundary protection (data, not direction). Decision tree gated on response by 2026-05-31.
2026-05-17 CONTRIB: Engagement on #21089 — @jagmarques (NexusQuant author) replied at 2026-05-17T07:42:05Z (8.5h post-comment) with a new datapoint: per-head fp16 masking of the lowest-2% KV-heads on Qwen2.5-7B-Instruct matches K3V2 boundary-protect on retrieval at 2.28 vs 3.93 K-bits/element. Orthogonal to #21089's quant-type scope. Opens Branch C: per-head fp16 KV-head precision-override as a fresh PR surface in llama.cpp, no in-flight competitor. Status updated to ENGAGED; next moves are (1) grep upstream for existing per-head precision plumbing, (2) read KV-rotation PR for outlier-statistics interaction, (3) decide on a one-question follow-up reply.
2026-06-17 INGEST: 7 new 2026 source extracts (nemotron-3-nano, mellum2, swe-hero 2604.01496, swe-trace 2604.14820, exllamav3 releases, harp 2605.29843, openhands May-2026 update). Together they form a 4 GB agentic-coding stack: ready-to-run model (Nemotron-3-Nano-4B, Q4_K_M 2.9 GB, BFCL v3 61.1), open code MoE (Mellum2 12B/2.5B-active), distillation recipe (SWE-HERO: 32B 62.2% / 7B 52.7% SWE-bench Verified), PRM+TTS recipe (SWE-TRACE: 4B 40.7%, 30B-A3B 71.2%), extreme-low-VRAM runtime (ExLlamaV3/EXL3, 1.6 bpw + 2-8 bit KV quant), 2-bit accuracy rescue (HARP), and a local-friendly harness (OpenHands prompt-serialization tool-call fallback).
2026-06-17 INGEST: 7 new wiki pages (models/nemotron-3-nano, models/mellum2, training/swe-hero, training/swe-trace, runtimes/exllamav3, techniques/harp, harnesses/openhands). Cross-linked into mamba-ssm-hybrids, moe-active-param, agentic-rl-coding, llama-cpp; index.md updated to 67 articles / 50 source extracts.
2026-06-17 LINT: relative-link check on the 7 new pages -- fixed 2 path bugs (training pages -> ../models/qwen3-coder-next.md); 0 broken links and 0 broken raw refs remaining.
2026-06-17 LINT: maintain pass. 66 pages; 0 broken links, 0 em dashes, 0 index drift, 0 orphans. See wiki/LINT-REPORT.md. Clean.
2026-07-06 INGEST: 4 new source extracts covering May-June 2026 window (lfm2-5: LFM2.5-8B-A1B 2026-05-28 + LFM2.5-230M 2026-06-25; gemma-4: family 2026-04-02 + QAT checkpoints 2026-06-05; swe-chain arXiv:2605.14415; runtimes-may-june-2026: llama.cpp MTP PR #22673, Ollama v0.24-0.30.x, vLLM EAGLE 3.1, MLX M5, LM Studio MTP).
2026-07-06 INGEST: 3 new wiki pages (models/lfm2-5.md, models/gemma-4.md, benchmarks/swe-chain.md); updates to runtimes/llama-cpp.md (May 2026 MTP wave), runtimes/ollama-and-friends.md (May-June updates), superseded banners on models/lfm2.md and models/gemma-3.md; index.md to 70 articles / 54 extracts.
2026-07-06 LINT: touched-page check. 0 em dashes, 0 broken relative links across 8 touched pages.
2026-07-07 MAINT: dashboard/ added. Self-contained benchmark-trend dashboard (index.html + data.js seed) with scrape.py refreshing live scores from benchlm.ai (SWE-bench Verified), swe-rebench.com, and BFCL data_overall.csv into live.js; open-dashboard.sh scrapes then opens. Curated seed includes closed-lab frontier (Anthropic/OpenAI/Google/xAI) vs open >35B vs open laptop-class groups.
2026-07-07 MAINT: dashboard v2. SWE-bench Verified seed densified to 63 dated points (adds 2025 open-model wave: DeepSeek R1/V3.1/V3.2, Kimi K2 line, GLM-4.5/4.6, Qwen3-Coder, Devstral, SWE-agent-LM-32B, KAT-Dev-32B, plus GPT-5/o3/Gemini 3 Pro closed fills). Scrollable ranked lists beside both big charts, open/closed filter, list hover highlights the plotted point; undated live leaderboard entries appear grayed in the list.
2026-07-07 MAINT: dashboard v3. BFCL/LCB charts now fed from live leaderboards (gorilla data_overall.csv 109 rows best-variant-deduped; benchlm liveCodeBench + bfclV4 added as scrape sources) dated via a 96-entry release-date map in data.js. All charts: per-family running-best frontier lines plus dashed linear extrapolations 4-5 months out, non-frontier points dimmed for readability, today-divider line. Fixed variant dedupe swallowing Kimi K2 Thinking.
