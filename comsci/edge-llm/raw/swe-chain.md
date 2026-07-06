# SWE-Chain: Benchmarking Coding Agents on Chained Release-Level Package Upgrades — source extract

Extracted 2026-07-06 from https://arxiv.org/abs/2605.14415 (submitted 2026-05-14).

Authors: Man Ho Lam, Chaozheng Wang, Hange Liu, Jingyu Xiao, Haau-sing Li, Jen-tse Huang, Terry Yue Zhuo, Michael R. Lyu.

## Design

- Evaluates sequential release-level package upgrades instead of isolated issue fixes. Each version transition builds cumulatively on the agent's previous codebase modifications.
- Scale: 12 upgrade chains across 9 Python packages; 155 version transitions; 1,660 grounded upgrade requirements.
- Construction: divide-and-conquer synthesis pipeline aligning release notes with code diffs to produce grounded upgrade specifications.
- Metrics: resolving rate, precision, F1 (Build+Fix regime).

## Results (Build+Fix regime, paper-reported)

- Industry average: 44.8% resolving, 65.4% precision, 50.2% F1.
- Best: Claude-Opus-4.7 + Claude Code: 60.8% resolving, 80.6% precision, 68.5% F1.
- Headline finding: current agents struggle to maintain functionality across chained releases; errors compound as chains lengthen.

## Notes

- No small-model results reported in the extracted material.
- Related June 2026 arXiv work in the same space (not extracted in depth): SWE-Explore (2606.07297, repo-exploration benchmark), "Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering" (2606.17799).
