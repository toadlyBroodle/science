# Cline, Continue, Goose

> **Summary:** Open-source IDE-integrated agentic-coding harnesses. Cline (formerly Claude-Dev) and Continue are VS Code extensions; Goose (Block) is a CLI/desktop agent. All BYOM (bring-your-own-model); directly useful for hosting a 4 GB-class local model.

**Sources:** project documentation; ecosystem context.

---

## Cline

VS Code extension. Implements a Claude-Code-like tool loop (file read/write, shell, browser). Multi-provider; OpenAI-compatible endpoints work out of the box, enabling local llama.cpp/vLLM/Ollama backends.

The tool format is similar to Claude Code's (XML-tagged tool calls). Models fine-tuned for Claude Code's tool format transfer with minimal adaptation.

## Continue

VS Code + JetBrains extension. More inline-completion-focused than autonomous agent loops, but the agent mode is comparable to Cline. Strong "context provider" abstraction (custom context types) makes it useful for teams with bespoke knowledge bases.

## Goose

Block (Square)'s desktop / CLI agentic agent. Uses MCP (Model Context Protocol) for tool access; the same protocol Anthropic's Claude Code adopts. MCP-native makes Goose the closest open-source structural analog to Claude Code.

Multi-provider; supports local models via OpenAI-compatible endpoints.

## Position chart

| Harness | UI | Tool format | Closest analog |
|---|---|---|---|
| Claude Code | CLI | Anthropic JSON | itself (target) |
| aider | CLI | SEARCH/REPLACE blocks | own benchmark |
| Cline | VS Code | XML-tagged | Claude Code |
| Continue | VS Code / JetBrains | varies | Cursor |
| Goose | CLI / desktop | MCP | Claude Code (MCP) |

## Relevance to 4 GB target

For solo dev wanting Claude-Code-like local capability: Cline + a local Qwen3-Coder/Phi-4-mini fine-tuned for Cline's exact format is the most direct path. Goose is the second choice (MCP-native means transfer of tool definitions is cleanest).

## See Also

- [Claude Code](claude-code.md)
- [aider](aider.md)
- [SLM agentic tool calling](../training/slm-agentic-tool-calling.md)
