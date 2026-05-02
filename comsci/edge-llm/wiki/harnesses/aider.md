# aider

> **Summary:** Open-source CLI agentic-coding harness from Aider-AI. Multi-provider (OpenAI, Anthropic, local via OpenAI-compatible APIs, Ollama). Ships its own [Aider polyglot benchmark](../benchmarks/aider-polyglot.md); making it the harness with the most public local-model data.

**Sources:** aider.chat documentation; [raw/aider-polyglot.md](../../raw/aider-polyglot.md)

---

## Tool model

aider uses a structured-edit format ("edit blocks") rather than the function-call JSON Claude Code uses. The model emits SEARCH/REPLACE blocks; aider applies them. Format conformance is critical: the [Aider polyglot benchmark](../benchmarks/aider-polyglot.md) explicitly measures edit-correctness as a separate column.

## Loop structure

1. User prompt + context (file contents, repo map).
2. Model emits edit blocks.
3. aider applies edits.
4. On test failures, aider retries with feedback.

Less ambitious than Claude Code's multi-tool loop (no shell execution by default, no sub-agents), but predictable and well-instrumented.

## Why aider matters for the 4 GB target

- Public benchmark with edit-conformance metrics → diagnostic for small-model failure modes.
- Multi-provider → can swap the local model behind without changing the user-facing flow.
- Open source → fine-tuning the model to match aider's exact edit format is straightforward (the format is documented and stable).

## Pairs with

- [llama.cpp](../runtimes/llama-cpp.md) via Ollama or llama-cpp-server.
- [vLLM](../runtimes/vllm.md) via OpenAI-compatible endpoint.

## See Also

- [Claude Code](claude-code.md)
- [Cline / Continue / Goose](cline-continue-goose.md)
- [Aider polyglot benchmark](../benchmarks/aider-polyglot.md)
