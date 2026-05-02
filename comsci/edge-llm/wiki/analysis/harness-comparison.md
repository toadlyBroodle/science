# Harness Comparison for Local 4 GB Agentic Coding

> **Summary:** Five harnesses considered: Claude Code (the target), aider, Cline, Continue, Goose. For local 4 GB-class models, **the binding constraint is tool-call format conformance**, not raw IQ. Cline + a model fine-tuned for Cline's exact format is the most direct local Claude-Code analog. aider gives the cleanest published benchmark (Aider polyglot edit-correctness column).

**Sources:** [harnesses/claude-code.md](../harnesses/claude-code.md), [harnesses/aider.md](../harnesses/aider.md), [harnesses/cline-continue-goose.md](../harnesses/cline-continue-goose.md), [training/slm-agentic-tool-calling.md](../training/slm-agentic-tool-calling.md), [benchmarks/aider-polyglot.md](../benchmarks/aider-polyglot.md), [benchmarks/bfcl.md](../benchmarks/bfcl.md)

---

## What a coding harness actually does

Every agentic coding harness is a tool-orchestration loop:

1. Build prompt: system + user request + repo context.
2. Model emits tool call(s) (or edit blocks).
3. Harness parses, validates, executes.
4. Result feeds back into next turn.
5. Loop until goal reached or budget exhausted.

The interesting differences are in steps 2 and 3: what *format* the model must emit, and how strict the parser is.

## Format taxonomy

| Harness | Format | Strictness | Recovery |
|---|---|---|---|
| Claude Code | Anthropic JSON tool calls (XML-tagged in prompt) | strict | retry once |
| aider | SEARCH/REPLACE blocks (markdown-fenced) | strict | second attempt with test feedback |
| Cline | XML-tagged tool calls (similar to Claude Code) | strict | retry on parse error |
| Continue | varies (Claude-style or OpenAI tool calls) | depends on provider | retry |
| Goose | MCP-native (JSON-RPC over stdio) | strict | retry |

Strict + agentic coding loop means: a 99%-format-correct model becomes a ~85% effective agent over a 4-step loop (0.99⁴ = 0.96, but each error usually breaks the loop, not just the step).

## Why format conformance dominates for 4 GB models

The [350M tool-calling paper](../training/slm-agentic-tool-calling.md) showed that *one epoch of SFT on the right format* takes a 350M base from random to 77.55% on ToolBench. The same lever applies at 1-4B: targeted format-conformant SFT closes the gap with much larger general-purpose models.

This is the wiki's central argument: **you don't need a smarter model; you need a model that emits the harness's exact format reliably.**

## Harness choice for the local-Claude-Code goal

Three live options:

### Option A: Cline + local model fine-tuned for Cline format

- **Pros:** Cline's tool format mirrors Claude Code's; minimal cognitive load when switching.
- **Cons:** Cline-specific format SFT data must be assembled (or generated; see [ToolACE](../training/toolace.md) approach).

### Option B: aider + local model fine-tuned for SEARCH/REPLACE format

- **Pros:** aider polyglot benchmark provides reproducible measurement; well-documented format.
- **Cons:** Less feature-rich than Claude Code (no shell, no sub-agents by default).

### Option C: Goose + local model fine-tuned for MCP

- **Pros:** MCP is the protocol Claude Code is migrating toward; longest-term-correct choice.
- **Cons:** MCP tooling around small-model fine-tuning is least mature.

## Harness-format SFT data requirement (rough)

To get a 1-4B model to harness-format-conformant 99%, the recipe (from the [350M paper](../training/slm-agentic-tool-calling.md)) suggests:

- 5-50 K targeted dialog examples in the *exact* harness format.
- One epoch SFT (more risks overfitting).
- Cost: < $200 cloud compute for a Qwen3-Coder-3B / Phi-4-mini-class base via Unsloth + LoRA.

See [contribution roadmap](contribution-roadmap.md) (pending Phase 5) for the concrete recipe.

## Benchmark-to-harness mapping

| Benchmark | Best for measuring |
|---|---|
| [Aider polyglot](../benchmarks/aider-polyglot.md) (edit-correctness column) | Format conformance specifically |
| [BFCL v3 / v4](../benchmarks/bfcl.md) | Multi-turn tool-call correctness |
| [SWE-Bench Verified](../benchmarks/swe-bench.md) | End-to-end agentic capability |
| [Terminal-Bench](../benchmarks/terminal-bench.md) | CLI breadth |
| [LiveCodeBench](../benchmarks/livecodebench.md) | Contamination-free raw coding |

For Phase-5 evaluation of "is the local-fine-tuned model ready," the diagnostic stack is: BFCL v4 → Aider polyglot edit column → SWE-Bench Lite. Pass all three with respectable margins, and Claude-Code-equivalent performance on a 4 GB model becomes plausible.

## See Also

- [Runtime comparison](runtime-comparison.md)
- [Spec decoding at 4 GB](spec-decoding-at-4gb.md)
- [Contribution roadmap](contribution-roadmap.md) (pending)
