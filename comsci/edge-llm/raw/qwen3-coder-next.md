# Qwen3-Coder-Next Technical Report

**Source:** arXiv:2603.00729 (https://arxiv.org/abs/2603.00729)
**Fetched:** 2026-05-02 via WebFetch (abstract + page metadata)
**Authors:** Qwen Team (20 authors): Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, Zeyao Ma, Kashun Shum, Xuwu Wang, Jinxi Wei, Jiaxi Yang, Jiajun Zhang, Lei Zhang, Zongmeng Zhang, Wenting Zhao, Fan Zhou
**Submitted:** 2026-02-28

## Abstract / extracted content

Open-weight LLM specialized for coding agents. 80B total parameters / 3B active per token (MoE). Agentic training via large-scale synthesis of verifiable coding tasks paired with executable environments; learns directly from environment feedback in mid-training and RL stages. Both base and instruction-tuned weights released. Competitive on agent-centric benchmarks (SWE-Bench, Terminal-Bench) relative to active parameter count.

## Key claims

- 80B / 3B active competitive with much larger dense coders on agentic benchmarks.
- Mid-training + RL on verifiable coding tasks with executable environments improves agentic competence directly.
- Open weights released (base + instruct).

## Parameter counts

- Total: 80 B
- Active: 3 B per token (MoE active-param)

## Benchmarks referenced

- SWE-Bench
- Terminal-Bench
- Specific scores not extracted from arXiv abstract page; see paper Section X for tables.

## Relevance to 4 GB edge target

The headline 80B-total / 3B-active design is too big for 4 GB VRAM in pure-GPU mode (80B even at Q4 ~ 40 GB), but is the canonical reference for what an "agentic-coder MoE" can be. Hybrid CPU+GPU offload (KTransformers-style) is the only path for this specific model on 4 GB; smaller dense Qwen3-Coder variants are the practical edge candidates.
