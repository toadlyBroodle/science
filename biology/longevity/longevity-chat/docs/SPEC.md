# longevity-chat SPEC

> Canonical project spec for `longevity-chat`: a public web chat interface that answers questions about the longevity research corpus at `~/Dev/science/biology/longevity/` using Gemini 2.5 Flash in an agentic retrieval loop. Deployed on the shared Vultr VPS alongside `csvagent.com`. Read this file and `docs/TODO.md` end-to-end before any change; update both in the same commit as the code change.

## Goal

Stand up a low-cost public chat endpoint that lets anyone ask longevity-research questions and get cited, evidence-tier-aware answers grounded in the 87-paper / 147-page wiki at `biology/longevity/`. Target per-query cost ≤ $0.02 with full citations to `wiki/papers/*` and `wiki/topics/*` pages. The corpus is the user's curated open-source wiki; the chat surface is a thin agentic layer over it, not a fine-tuned model and not a vector-DB RAG system. The wiki already carries evidence tiers, contradictions surfaced on topic pages, and structured front matter — the model's job is retrieval + faithful summarization with citations, not domain reasoning from scratch.

## Architecture / stack (one-liner each)

- **Backend**: FastAPI + SQLAlchemy + PostgreSQL (same instance as csvagent), Python 3.11+, gunicorn + uvicorn workers. Mirrors `~/Dev/csvagent/main.py` shape.
- **LLM**: `litellm` SDK targeting `gemini/gemini-2.5-flash` with streaming. Reuse `gem_llm_info.json` pricing from `~/Dev/csvagent/` for cost-tracking.
- **Wiki retrieval**: existing `wiki/build/tfidf.npz` + `tfidf_vocab.json` (loaded once at startup) plus `wiki/build/graph.json` for wikilink traversal. No embeddings, no vector DB.
- **Agent loop**: 3 tools exposed to the model — `search_wiki(query)`, `read_page(path)`, `follow_wikilinks(path)`. Typical query resolves in 2-4 tool calls.
- **Frontend**: static SPA at `/static/` — vanilla HTML/CSS/JS, no build step (matches csvagent). Chat UI with streaming, citation chips, evidence-tier badges.
- **Auth**: public, no signup. IP-based rate limit (Redis or in-memory sliding-window). Optional Phase-9 login if abuse appears.
- **Deployment**: VPS Linux + systemd unit + nginx reverse proxy + Let's Encrypt cert via certbot. Native install, no Docker.
- **Wiki sync**: cron `git pull` against the science repo (sparse-checkout for `biology/longevity/`); rebuild TF-IDF index on update.
- **Commit shape**: each phase ships `docs/SPEC.md` + `docs/TODO.md` + code in one commit. Commit messages: imperative subject, ≤50 chars, no AI attribution (matches the user's global rules).

## Open decisions (resolve before Phase 10)

- **Domain**. Placeholder: `<chat-domain>` (e.g. `longevity.<top-domain>` or a new domain). Resolve at the start of Phase 9 before requesting the TLS cert.
- **Budget cap**. Default $5/day across all users; configurable via env `DAILY_BUDGET_USD`. Resolve at Phase 5 when cost tracking lands.
- **Wiki sync cadence**. Default cron every 30 minutes. Switch to a GitHub webhook in a later phase if 30 min is too laggy.
- **Rate limit**. Default: 5 requests/minute per IP, 100/day per IP. Tune on first traffic.

## Phases

### Phase 0: scaffold + dev environment

Bootstrap the repo so a developer can `python start_app.py` locally against a sample wiki and see the chat endpoint respond. No real LLM call yet; the chat returns canned fixtures.

- [ ] 0.1 [easy] Create `pyproject.toml` (or `requirements.txt`) pinning `fastapi`, `uvicorn[standard]`, `gunicorn`, `litellm`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`, `pyyaml`, `numpy`, `scipy`, `scikit-learn`, `pytest`. Python `>=3.11`.
- [ ] 0.2 [easy] Mirror the csvagent top-level layout: `main.py` (FastAPI app + routers), `start_app.py` (entry point with `--setup-only` for DB migrations), `logging_config.py`, `static/`, `tests/`, `logs/`, `.gitignore`, `.env.example`, `README.md`.
- [ ] 0.3 [easy] `.env.example` with the required keys: `GEMINI_API_KEY`, `DATABASE_URL` (postgresql://...), `WIKI_ROOT` (default `/home/rob/Dev/science/biology/longevity`), `SECRET_KEY`, `PUBLIC_BASE_URL`, `DAILY_BUDGET_USD`, `RATE_LIMIT_PER_MIN`, `RATE_LIMIT_PER_DAY`, `LOG_DIR`. `.env` itself is gitignored.
- [ ] 0.4 [medium] `start_app.py` parses `--setup-only` → runs SQLAlchemy `Base.metadata.create_all`, exits; otherwise launches `uvicorn main:app --host 127.0.0.1 --port 8002 --reload`. Port 8002 (avoid 8000/8001 collisions with csvagent/sdrai).
- [ ] 0.5 [easy] `main.py` mounts `/api` router, `/static`, and a `/health` endpoint returning `{"status": "ok", "wiki_pages": <int>}`.
- [ ] 0.6 [easy] `tests/test_health.py` hits `/health`, asserts 200 + page count > 0. `pytest` runs green.

### Phase 1: wiki retrieval tools

Three thin Python functions that the LLM tool-use loop will call. All read from `WIKI_ROOT`. No LLM yet; tests assert behavior against the real longevity wiki.

- [ ] 1.1 [medium] `wiki_tools.py` — `load_index(wiki_root)` reads `wiki/build/index.json` + `wiki/build/graph.json` + loads `wiki/build/tfidf.npz` + `tfidf_vocab.json` into memory once at startup. Returns a `WikiIndex` object with O(1) page lookup and O(log n) TF-IDF query.
- [ ] 1.2 [medium] `search_wiki(query: str, top_k: int = 5) -> list[SearchHit]` — vectorize `query` with the loaded TF-IDF vocab, score against the page matrix, return top-k hits with `id`, `title`, `kind`, `path`, `score`, `summary` (first 200 chars of body).
- [ ] 1.3 [easy] `read_page(path: str) -> PageBody` — reads `<wiki_root>/<path>`, returns body text + parsed front matter. Path must be inside `wiki_root` (refuse traversal).
- [ ] 1.4 [easy] `follow_wikilinks(path: str) -> list[str]` — returns outbound wikilink targets for the given page from `graph.json`. Used when the model wants to traverse "related" without doing another search.
- [ ] 1.5 [medium] `tests/test_wiki_tools.py` — load the real longevity wiki, assert `search_wiki("rapamycin")` returns `papers/pearl-rapamycin-2025` in the top 3; assert `read_page("wiki/papers/pearl-rapamycin-2025.md")` parses the `evidence_tier: T6` front matter; assert `follow_wikilinks` on a known topic returns expected neighbors.

### Phase 2: agent loop with Gemini 2.5 Flash

The LLM tool-use loop. Synchronous first; streaming added in Phase 3. System prompt carries the orientation context (index.md "Grok the field" + recommendations.md tier-tagged list + evidence-tiers.md rubric); per-query body carries the user question; tools are invoked agentically.

- [ ] 2.1 [hard] `chat.py` — `class ChatAgent` with `__init__(wiki_index, model="gemini/gemini-2.5-flash")`. `run(user_question: str) -> ChatResponse` does the full tool-use loop synchronously: build system prompt, call `litellm.completion` with tools, parse tool calls, execute against `wiki_tools`, feed results back, repeat until `finish_reason == "stop"`. Cap at 8 tool calls per turn.
- [ ] 2.2 [medium] System prompt builder reads `<wiki_root>/CLAUDE.md` (schema spec), `wiki/index.md`, `recommendations.md`, `wiki/analysis/evidence-tiers.md` and concatenates into a single ~12k-token preamble. Cached via Gemini's context-caching API where supported; falls back to literal repetition.
- [ ] 2.3 [medium] Tool schema: convert `search_wiki` / `read_page` / `follow_wikilinks` signatures into OpenAI/litellm tool-call JSON schema. Wire response handling for each tool name.
- [ ] 2.4 [medium] Citation extraction: post-process the assistant's final answer to find `[[papers/<id>]]` / `[[topics/<slug>]]` patterns and emit a `citations: [{slug, title, kind, url}]` array alongside the text. URL is the GitHub source link for each page.
- [ ] 2.5 [medium] `tests/test_chat_offline.py` — mock `litellm.completion` to return a scripted tool-call sequence; assert the loop terminates, citations are extracted, and the final answer is returned. No real API calls in tests.
- [ ] 2.6 [easy] Smoke script `bin/smoke_chat.py` that takes a real query, runs the loop end-to-end against the live Gemini API (gated on `GEMINI_API_KEY`), prints the answer + citations + token usage. Manual-run only, not in pytest.

### Phase 3: streaming chat endpoint

Wire the agent loop into a FastAPI endpoint that streams to the browser via SSE. The frontend is built in Phase 4.

- [ ] 3.1 [hard] `POST /api/chat` accepting `{"question": str, "session_id": str | null}`. Returns `text/event-stream`. Each SSE event is one of: `{"type": "token", "text": "..."}`, `{"type": "tool_call", "name": "...", "args": {...}}`, `{"type": "tool_result", "name": "...", "hits": [...]}`, `{"type": "citation", "slug": "...", "title": "..."}`, `{"type": "done", "usage": {...}, "cost_usd": 0.0123}`.
- [ ] 3.2 [medium] Convert `ChatAgent.run` to `async def stream(...)` yielding events. Use `litellm.acompletion(stream=True)` and chunk-by-chunk forwarding to the client.
- [ ] 3.3 [medium] Persist a `chat_sessions` row + N `chat_messages` rows per session (id, ip_hash, question, answer, tool_calls_json, citations_json, prompt_tokens, completion_tokens, cost_usd, created_at). Used by Phase 5 cost tracking and Phase 7 logging.
- [ ] 3.4 [medium] `tests/test_chat_endpoint.py` — TestClient hits `/api/chat`, asserts SSE event sequence shape, asserts a `done` event with non-zero cost.

### Phase 4: frontend SPA

Vanilla JS chat UI. No build step. One HTML file + one CSS file + one JS file. Lives in `static/` mounted at `/`. Modeled on `~/Dev/csvagent/static/` but stripped to chat-only.

- [ ] 4.1 [medium] `static/index.html` with a chat transcript area, a textarea + send button, and a sidebar listing the source-papers table from `wiki/index.md` for browsing (optional, collapsible).
- [ ] 4.2 [medium] `static/chat.js` opens an EventSource (or fetch + ReadableStream) against `/api/chat`, appends streaming tokens to the current assistant bubble, renders `tool_call` events as a transient "Searching the wiki for X..." indicator, renders `citation` events as chips below the bubble, renders `done` events as a final cost/usage footnote.
- [ ] 4.3 [easy] `static/style.css` — clean, monospace-leaning, mobile-friendly. Citation chips link to the GitHub source for each cited page. Evidence-tier badges color-coded T0 (gray) through T7 (green).
- [ ] 4.4 [easy] Landing page at `/` shows the chat UI directly (no login wall in MVP). Includes a one-line "This bot answers from <N> peer-reviewed sources; see the wiki at <github link>" header.
- [ ] 4.5 [medium] Browser-side manual smoke: open `/`, ask "is creatine worth taking?", verify the response cites `papers/chilibeck-2017-creatine` and tags the recommendation with its evidence tier.

### Phase 5: cost tracking + budget caps + rate limiting

Guardrails so a public endpoint with no auth doesn't burn the budget on day one.

- [ ] 5.1 [medium] Cost tracking — extract `prompt_tokens` + `completion_tokens` from the Gemini response, multiply by the tier rates in `gem_llm_info.json` (copy from csvagent), record `cost_usd` on the `chat_messages` row. Per-session and per-IP aggregates queryable via SQL.
- [ ] 5.2 [medium] Daily-budget cap — middleware checks `SUM(cost_usd) WHERE created_at >= today` before each chat call; if ≥ `DAILY_BUDGET_USD`, return 429 with `{"error": "daily budget exhausted, try tomorrow"}`. Reset at UTC midnight.
- [ ] 5.3 [medium] Rate limiting — sliding-window per IP, in-memory dict keyed by `hash(ip)` with `(timestamp, count)` rows. Refuse with 429 + `Retry-After` header when over `RATE_LIMIT_PER_MIN` or `RATE_LIMIT_PER_DAY`. Phase 11 (deferred) swaps to Redis if multi-worker.
- [ ] 5.4 [easy] `/api/usage` endpoint (admin-gated by simple bearer token from `ADMIN_TOKEN` env) returning `{"today_usd": ..., "month_usd": ..., "requests_today": ..., "requests_per_ip_today": {...}}`.
- [ ] 5.5 [medium] `tests/test_budget_caps.py` — fake clock, simulate 100 chats summing to $5.01 in one day, assert the 101st chat returns 429.

### Phase 6: wiki sync (VPS pulls latest from github)

The VPS needs the wiki content + the TF-IDF build artifacts. Build artifacts are gitignored (`wiki/build/` per longevity's `.gitignore`), so they must be rebuilt on the VPS.

- [ ] 6.1 [medium] `bin/sync_wiki.sh` — sparse-checkout clone of `github.com/clankwright/science` at `WIKI_REPO_DIR` (default `/srv/longevity-chat/science`), pull main, run `python3 scripts/index.py` from the wiki dir to regenerate `wiki/build/{index,graph,keywords,pages}.json` + `tfidf.npz` + `tfidf_vocab.json`. Exit non-zero if `scripts/lint.py` fails. Re-runnable, idempotent.
- [ ] 6.2 [medium] After successful sync, send `SIGHUP` to the longevity-chat systemd unit so the app reloads the in-memory `WikiIndex` without a full restart. Alternatively: write a sentinel file and have the app watch it.
- [ ] 6.3 [easy] Cron entry `/etc/cron.d/longevity-chat-sync` running `bin/sync_wiki.sh` every 30 minutes as user `rob`, logging to `/var/log/longevity-chat-sync.log`.
- [ ] 6.4 [easy] `tests/test_sync_wiki_dryrun.py` — runs the sync against a local fixture wiki, asserts the build artifacts appear and the lint script passes.

### Phase 7: logging + monitoring + error handling

Mirror the csvagent pattern: daily-rotated logs, LiteLLM noise suppression, structured error responses.

- [ ] 7.1 [easy] `logging_config.py` configures daily-rotated logs at `LOG_DIR/longevity-chat_YYYY-MM-DD.log`, suppresses `litellm` / `httpx` / `httpcore` debug noise (mirror csvagent's pattern).
- [ ] 7.2 [medium] Structured errors — every endpoint returns `{"error": str, "code": str, "request_id": uuid}` on 4xx/5xx; uncaught exceptions logged with stack trace + request_id; client sees only the code + a friendly message.
- [ ] 7.3 [easy] `/api/admin/recent-errors` (admin-token gated) returning the last 50 error rows for triage.
- [ ] 7.4 [medium] Periodic budget alert — daily cron at 23:00 UTC posts a one-line summary to a Telegram chat (reuse the `bin/notify-telegram.sh` pattern from skill-set if convenient, or simply curl Telegram's bot API): "longevity-chat: $X.XX spent across N chats today, top intent: '<sample question>'."

### Phase 8: tests

- [ ] 8.1 [easy] `pytest` config in `pyproject.toml`: `testpaths = ["tests"]`, `addopts = "-q --tb=short"`. CI-friendly defaults.
- [ ] 8.2 [medium] End-to-end test marked `@pytest.mark.live` that hits the real Gemini API with one canned question (skipped by default; opt in with `--run-live`). Asserts non-empty response with at least one citation.
- [ ] 8.3 [easy] `bin/test.sh` running the offline test suite + smoke; the deploy script (Phase 9) refuses to deploy if this exits non-zero.

### Phase 9: VPS deployment (systemd + nginx + TLS)

Land the app on the existing VPS. Pre-resolve the `<chat-domain>` decision before this phase.

- [ ] 9.1 [medium] `systemd/longevity-chat.service` — copy `~/Dev/csvagent/csvagent.service`, change `WorkingDirectory`, `ExecStart` (port 8002), `EnvironmentFile`, log dirs. `ProtectSystem=strict`, `ProtectHome=no`, `User=rob`.
- [ ] 9.2 [medium] `nginx/longevity-chat.conf` — server block for `<chat-domain>`, reverse proxy `/api/chat` with `proxy_buffering off` (SSE), `proxy_read_timeout 600s`. Static `/static` cached 30d. HSTS + the standard csvagent security headers.
- [ ] 9.3 [easy] `bin/deploy.sh` — git pull on VPS, install/update deps in venv, run migrations, restart systemd unit, reload nginx. Refuses to run if `bin/test.sh` exits non-zero. Reuse csvagent's deploy script as a template if one exists.
- [ ] 9.4 [easy] DNS A record for `<chat-domain>` pointing at the VPS IP.
- [ ] 9.5 [easy] `certbot certonly --nginx -d <chat-domain>` to mint the TLS cert; verify auto-renewal cron is active.
- [ ] 9.6 [medium] Production smoke: hit `https://<chat-domain>/health` (200), then `/api/chat` with a real question (streaming works, citations appear, cost recorded).

### Phase 10: launch + monitoring window

- [ ] 10.1 [easy] Wire the launch into the existing daily-backup cron (longevity-chat DB included).
- [ ] 10.2 [easy] Add a `robots.txt` allowing indexing of `/` and disallowing `/api/`. `sitemap.xml` listing only `/`.
- [ ] 10.3 [medium] Two-week monitoring window — daily Telegram digest of cost, request volume, error rate. Tune `DAILY_BUDGET_USD`, `RATE_LIMIT_*` based on real traffic.
- [ ] 10.4 [easy] Public announcement (link from `~/Dev/science/biology/longevity/README.md`, optional Reddit post to r/Biohackers per the prior posting cadence).

## Deferred / out of scope

- **User accounts + login.** MVP is public + rate-limited. Add auth (JWT + email-verify, mirror csvagent) only if abuse becomes structural.
- **Per-user billing / Stripe.** Same. Daily budget cap is a softer guardrail and is adequate until usage justifies billing.
- **Embedding-based retrieval.** Explicitly out of scope. The longevity wiki has TF-IDF + wikilinks already; embeddings would not improve recall on a 147-page corpus and would lose the cited-claim chain.
- **Multi-wiki dispatcher.** Serving aliens/, bpu/, etc. from the same chat surface is a Phase-11+ idea once longevity-chat itself is steady-state.
- **Self-hosted Qwen / DeepSeek alternative.** Gemini 2.5 Flash is the chosen baseline. Self-hosting becomes interesting around ~60k queries/month; revisit if volume materializes.
- **GitHub webhook for instant wiki sync.** Cron at 30-min cadence is the MVP. Switch to webhook in a later phase only if 30-minute lag becomes a real complaint.

## Glossary (project-specific terms)

- **agent loop**: the model's tool-call → execute → result-feedback → repeat cycle (Phase 2). Up to 8 calls per user turn.
- **citation chip**: UI element rendered below an assistant message linking to a `wiki/papers/*` or `wiki/topics/*` page on GitHub.
- **daily budget cap**: a hard cutoff `DAILY_BUDGET_USD` evaluated against today's `SUM(cost_usd)` (Phase 5.2).
- **evidence tier**: T0-T7 per `wiki/analysis/evidence-tiers.md` — used by the model to weight which sources to cite and by the UI to render badges.
- **synthesis page**: a `kind: synthesis` page (per the v1.1.0 wiki-curator schema): `recommendations.md`, `evidence-tiers.md`. Always pulled into the agent's system prompt.
- **wiki sync**: the cron-driven `git pull` + `scripts/index.py` rebuild that keeps the VPS's wiki copy + TF-IDF index current (Phase 6).

---

### How this file evolves

Same contract as the skill-set framework:

- A skill or dev-cycle closes a sub-item by flipping `- [ ]` → `- [x]` in the same commit as the code change.
- Sub-item IDs (`<phase>.<n>`) are stable and never renumbered; gaps from removed items are valid. Inserts use letter suffixes (`<phase>.<n>a`).
- Difficulty labels (`[easy]` / `[medium]` / `[hard]`) route the dev-cycle's model + effort tier per `max(item_tier, skill_floor)`.
- New work surfaced mid-cycle goes to `docs/TODO.md`'s `Next up`, not directly here.
- When all items in a phase are checked, append a 1-paragraph "completed" block + bulleted file citations to that phase. Don't delete the checklist.
- Commit messages follow imperative-mood, ≤50 char subject, no AI/LLM attribution (the user's hard global rule).
