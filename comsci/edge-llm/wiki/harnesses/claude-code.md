# Claude Code

> **Summary:** Anthropic's official CLI agentic-coding harness. The wiki's stated *target* harness; the goal is to come as close as possible to Claude Haiku 4.5 (or Sonnet 4.6) running inside Claude Code, but with a 4 GB-budget local model. Claude Code's tool protocol and context-management strategy define the bar.

**Sources:** Anthropic public documentation; ecosystem context.

---

## What Claude Code actually does

A loop that orchestrates tool use against a Claude model:
- File read/write (Edit, Write, Read).
- Shell execution (Bash).
- Search (Grep, Glob).
- Sub-agent spawning (Agent tool).
- Plan mode, todo lists, slash commands (skills).

The model emits tool-call JSON; the harness executes the tool; the result feeds back. Multi-turn loops, with auto-context-management (file-system snapshots, prompt caching).

## Why "Claude Code-like on 4 GB" is the wiki's target

For solo developers:
- Claude Code with Sonnet 4.6 or Haiku 4.5 is the production-grade agentic coder benchmark.
- Running locally avoids API spend and keeps source code private.
- The hard part is getting a small local model to match the *tool-call format* and the *multi-turn coherence* Claude Code expects, not raw IQ.

## What's transferable

The Claude Code tool protocol (a specific JSON schema plus a specific set of tools) is the de facto standard the wiki's local-model recipe should target. A model fine-tuned on Claude-Code-formatted tool calls (the [agentic SFT recipe](../training/) thread, see [350M tool-calling paper](../training/slm-agentic-tool-calling.md)) will plug into Claude Code's harness directly via API-compat shims (e.g., LiteLLM, OpenRouter-style proxies).

## Position vs aider / Cline / Continue / Goose

- **Claude Code:** Anthropic-native, official, the target. Closed harness, Anthropic API only.
- **[aider](aider.md):** Open-source CLI, multi-provider, terser tool model.
- **[Cline / Continue / Goose](cline-continue-goose.md):** VS Code / IDE-integrated, BYOM (bring your own model).

For a local-LLM agentic coder on 4 GB, the pragmatic path is: train the model on Claude-Code-style tool calls, then deploy via an open harness like aider or Cline pointed at the local model.

## See Also

- [aider](aider.md)
- [Cline / Continue / Goose](cline-continue-goose.md)
- [Aider polyglot benchmark](../benchmarks/aider-polyglot.md)
- [Agentic SFT thread](../training/slm-agentic-tool-calling.md)
