# Lint report

_Last run: 2026-06-17 (sst-wiki-curator maintain pass)._

| Check | Result |
|---|---|
| Pages scanned | 79 |
| Broken relative links | 90 (to 42 missing tool pages) |
| Em dashes (style rule: none) | 4 |
| Pages missing from index.md | 0 |
| Orphan pages (no inbound link) | 0 |

The 2026-06-17 ingest (Robinhood, Replit Agent, Perplexity Computer, Manus, ESM,
consumer-health-AI) added **zero** new broken links and zero orphans. All new
tool pages exist and are cross-linked from their capability pages and the index.

## [review] Missing tool pages (pre-existing deferred references)

The capability and analysis pages link to Tier-2/Tier-3 tool pages that were
intentionally deferred at curate time (see `log.md`, 2026-05-03 CURATE:
"Tool pages for non-shortlist references intentionally deferred to future
ingest passes; lint will surface gaps"). Each is a missing-page gap, not a
typo. Create on a future ingest pass or downgrade the reference to plain text:

- `tools/aider.md` — referenced by: wiki/capabilities/agentic-coding.md, wiki/tools/claude-code.md
- `tools/browser-use-lib.md` — referenced by: wiki/capabilities/browser-use.md, wiki/tools/playwright-mcp.md
- `tools/cartesia-sonic.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md, wiki/tools/elevenlabs.md
- `tools/claude-computer-use.md` — referenced by: wiki/capabilities/browser-use.md, wiki/tools/chatgpt-agent.md, wiki/tools/playwright-mcp.md
- `tools/claude-research.md` — referenced by: wiki/capabilities/autonomous-research.md
- `tools/cline.md` — referenced by: wiki/capabilities/agentic-coding.md, wiki/tools/claude-code.md
- `tools/composer-trade.md` — referenced by: wiki/capabilities/autonomous-trading.md
- `tools/devin.md` — referenced by: wiki/analysis/what-an-individual-can-now-do.md, wiki/capabilities/agentic-coding.md, wiki/capabilities/code-free-app-building.md
- `tools/duolingo-max.md` — referenced by: wiki/capabilities/personalized-education.md
- `tools/finalround-interview-copilot.md` — referenced by: wiki/analysis/shortlist.md, wiki/capabilities/automated-job-application.md
- `tools/flux.md` — referenced by: wiki/capabilities/generative-image.md, wiki/tools/midjourney.md
- `tools/google-imagen.md` — referenced by: wiki/capabilities/generative-image.md, wiki/tools/midjourney.md
- `tools/k-health.md` — referenced by: wiki/capabilities/personal-medical-ai.md, wiki/tools/openevidence.md
- `tools/kling.md` — referenced by: wiki/capabilities/generative-video.md, wiki/tools/veo.md
- `tools/kokoro-tts.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md
- `tools/lazyapply.md` — referenced by: wiki/analysis/shortlist.md, wiki/capabilities/automated-job-application.md
- `tools/magnifi.md` — referenced by: wiki/capabilities/autonomous-trading.md, wiki/capabilities/personal-tax-and-financial-ai.md
- `tools/notebooklm-audio.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md
- `tools/numerai.md` — referenced by: wiki/capabilities/autonomous-trading.md
- `tools/obsidian-with-ai.md` — referenced by: wiki/capabilities/personal-knowledge-management.md, wiki/tools/karpathy-llm-wiki-pattern.md
- `tools/openai-codex.md` — referenced by: wiki/capabilities/agentic-coding.md
- `tools/openai-deep-research.md` — referenced by: wiki/capabilities/autonomous-research.md, wiki/tools/gemini-deep-research.md, wiki/tools/perplexity.md
- `tools/openai-gpt-image.md` — referenced by: wiki/capabilities/generative-image.md, wiki/tools/midjourney.md
- `tools/openai-realtime.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md, wiki/tools/elevenlabs.md
- `tools/origin-financial.md` — referenced by: wiki/analysis/what-an-individual-can-now-do.md, wiki/capabilities/personal-tax-and-financial-ai.md, wiki/tools/wealthfront.md
- `tools/quantconnect.md` — referenced by: wiki/capabilities/autonomous-trading.md
- `tools/range-financial.md` — referenced by: wiki/capabilities/personal-tax-and-financial-ai.md, wiki/tools/wealthfront.md
- `tools/reclaim-motion-scheduling.md` — referenced by: wiki/capabilities/email-and-inbox-management.md, wiki/tools/superhuman.md
- `tools/runway.md` — referenced by: wiki/capabilities/generative-video.md, wiki/tools/veo.md
- `tools/sanebox.md` — referenced by: wiki/tools/superhuman.md
- `tools/sesame-csm.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md
- `tools/shortwave.md` — referenced by: wiki/capabilities/email-and-inbox-management.md, wiki/tools/superhuman.md
- `tools/skyvern.md` — referenced by: wiki/capabilities/browser-use.md
- `tools/sora.md` — referenced by: wiki/capabilities/generative-video.md, wiki/tools/veo.md
- `tools/synthesia.md` — referenced by: wiki/analysis/what-an-individual-can-now-do.md, wiki/capabilities/ai-avatars-and-dubbing.md, wiki/capabilities/voice-cloning-and-voice-agents.md, wiki/tools/heygen.md
- `tools/synthesis-tutor.md` — referenced by: wiki/capabilities/personalized-education.md, wiki/tools/khanmigo.md, wiki/tools/study-modes-chatgpt-claude-gemini.md
- `tools/turbotax-intuit-assist.md` — referenced by: wiki/analysis/what-an-individual-can-now-do.md, wiki/capabilities/personal-tax-and-financial-ai.md
- `tools/udio.md` — referenced by: wiki/capabilities/generative-music.md, wiki/tools/suno.md
- `tools/v0.md` — referenced by: wiki/capabilities/code-free-app-building.md, wiki/tools/lovable.md
- `tools/voice-agent-platforms.md` — referenced by: wiki/capabilities/voice-cloning-and-voice-agents.md
- `tools/wan-2-1.md` — referenced by: wiki/capabilities/generative-video.md, wiki/tools/veo.md
- `tools/woebot.md` — referenced by: wiki/capabilities/ai-therapy-and-companions.md, wiki/tools/wysa.md

## [review] Em dashes

Files using em dashes against the wiki's no-em-dash style rule:
- `wiki/analysis/shortlist.md`
- `wiki/analysis/specific-unlocks.md`
- `wiki/analysis/unhobbling-thesis.md`
- `wiki/index.md`

Pre-existing and pervasive (the index catalog and analysis pages use em dashes
throughout). Left for a dedicated style-normalization pass; not introduced by
this ingest beyond matching the index's existing line style.

## LLM-judgment checks

No contradictions or stale-claim regressions from the 2026-06 ingest. Shortlist
drift: the new tools (Replit Agent 4, ESM world model, Manus) are candidates for
a shortlist review but were not promoted in this pass.
