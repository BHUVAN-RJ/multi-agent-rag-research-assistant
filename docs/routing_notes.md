# LLM routing notes (input for the architecture doc)

Code: `src/llm/router.py` (policy + fallback), `src/llm/providers.py` (httpx clients).
Tests: `tests/test_router.py`.

## Providers and verified model IDs (checked 2026-10-25)

| Provider | Tier | Model ID | Why |
|---|---|---|---|
| NVIDIA NIM | fast | `nvidia/nemotron-3.5-lightning-30b-a3b` | 30B MoE / 3B active: cheap, low latency |
| NVIDIA NIM | strong | `nvidia/nemotron-3-ultra-550b-a55b` | 550B MoE / 55B active; replaced nemotron-3-super, which returned 404 for this account on 2026-10-26 |
| Gemini | fast | `gemini-3.5-flash-lite` | stable Flash-Lite with a free tier |
| Gemini | strong | `gemini-3.8-flash` | strongest Flash with a free tier |
| OpenRouter | free | `openrouter/free` | OpenRouter's free-only router (priced $0, routes only to `:free` models, so it survives free models being retired). No pinned backups: if it fails, the router moves to NVIDIA or Gemini |

Sources:
- Gemini: https://ai.google.dev/gemini-api/docs/pricing (free-tier column lists `gemini-3.8-flash`,
  `gemini-3.5-flash-lite`), https://ai.google.dev/gemini-api/docs/models (both marked stable;
  2.5 models are limited to existing users) and https://ai.google.dev/gemini-api/docs/deprecations
  (no shutdown date for either). The free-tier RPM/RPD is not published there; AI Studio shows the
  limits per project.
- NVIDIA: public catalog `GET https://integrate.api.nvidia.com/v1/models` (both IDs listed), plus
  https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b ("free API endpoint", sample
  code uses this exact ID) and https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b.
- OpenRouter: public `GET https://openrouter.ai/api/v1/models`, filtered to ids ending in `:free`
  with `pricing.prompt == pricing.completion == "0"`. Only 17 `:free` ids existed on that date. The
  others were coding-only models (Laguna, North Mini Code, Flint), safety classifiers, or tiny models.

The IDs are in `.env.example` and are also the defaults in `providers.DEFAULT_MODELS` /
`DEFAULT_OPENROUTER_FREE_MODELS`, so the app still works when `.env` leaves them empty (or keeps the
`<verified-...>` placeholder).

## Policy (hybrid: cost by default, capability when the task needs it)

Threshold: `LONG_CONTEXT_CHARS = 20_000` (about 5k tokens).

| Role | Condition | Primary | Dimension | Rule that fires |
|---|---|---|---|---|
| supervisor | any | NVIDIA fast | cost | short classification -> cheapest low-latency model |
| data | simple | NVIDIA fast | cost | simple structured SQL -> cheapest model |
| data | complex | NVIDIA strong | capability | large result set (>20 rows) to explain -> stronger Nemotron |
| research | simple, ctx <= 20k | Gemini fast | hybrid | cheap + fast enough for a short lookup |
| research | complex | Gemini strong | capability | multi-source synthesis -> strongest generator |
| research | ctx > 20k | Gemini strong | capability | long context -> Gemini (1M window) |
| summary | ctx <= 20k | OpenRouter `:free` chain | cost | zero-cost condensation |
| summary | ctx > 20k | Gemini fast | capability | long context; Lite is enough to condense |
| judge | any | Gemini strong | capability | RAGAS judge needs the best instruction-follower |

Every `RouteDecision.reason` starts with its dimension and names the rule, for example
`capability: context 48.0k chars > 20k threshold -> long-context Gemini strong`. The UI shows the
`llm_route` trace, which carries the full decision (including the fallback chain).
`router.routing_table()` returns one decision per (role, sample profile) if the doc wants to print
the table from code.

Why the split works: the supervisor and SQL calls are frequent and short, so they go to NVIDIA
(about 40 RPM, the highest free limit). Generation that needs quality or long context goes to Gemini.
Summaries go to OpenRouter's zero-cost pool, which keeps Gemini's tight RPM free for research.

## Fallback behavior

- Each decision has a fallback chain that always crosses providers (a test enforces this).
  OpenRouter shows up as one entry, `openrouter/free` (the two pinned `:free` backups were
  removed on 2026-10-26: both only ever returned 404 for this account), and every chain also has
  NVIDIA and Gemini entries. So if OpenRouter fails, the call lands on another provider at once.
- Every `complete()` call gets one `call_id` (12 hex chars) that is put on its `llm_route`, each
  `llm_fallback`, the `error` trace on total failure, and a closing `llm_result` trace. The
  `llm_result` says which candidate really answered (`provider`, `model`, `requested_model`,
  `served_model`, `latency_ms`, `ttft_ms`, `attempts`, `fell_back`, `budget_exhausted`, `error`),
  or that none did (`ok: false`).
- No key, or a `replace-me` placeholder key: that provider is skipped, and an `llm_fallback`
  trace with `skipped: true` says so.
- Cooldown skip: before walking the chain, each candidate that is still cooling down and has a
  ready candidate after it gets an `llm_fallback` trace with `skipped: true`, `reason: "cooldown"`
  and `cooldown_remaining_s`. It is then tried last rather than first. It is never dropped, so
  total exhaustion still tries everything.
- HTTP 429: the router moves to the next entry immediately, with no retry. The (provider, model)
  is then put in a cooldown (Retry-After, or 30s, capped at 120s).
- HTTP 402 (account out of credits) and HTTP 404 (model unavailable, e.g. OpenRouter "no
  endpoints" under the account's data policy, or an NVIDIA function not enabled for the account):
  not retried, 120s cooldown. OpenRouter can also report the code inside a stream chunk's `error`
  object; the same rules apply there.
- HTTP 503 ("high demand"): not retried; cooldown of Retry-After or 30s.
- Transient errors (transport error, other 5xx, 408) get one retry after 0.75s, and only if no
  token has been streamed yet. After that the router moves on.
- Empty completions (200 with no content) count as failures.
- If a stream breaks mid-way after tokens were emitted, the router falls back. The `llm_fallback`
  trace then carries `stream_restarted: true` so the UI can discard the partial tokens.
- Time budget: `TOTAL_CALL_BUDGET_SECONDS = 90` for one `complete()` call across the whole
  fallback walk. Each attempt (including its retry) is capped at the budget left. When it is
  spent, the rest of the chain stays untried and the error ends with "stopped after the 90s time
  budget with N candidate(s) untried". Reason: Cloud Run kills a request at 300s and one turn
  makes several LLM calls.
- Total failure: `LLMResult.error = "all N provider attempts failed: ..."`, plus an `error` trace
  and an `llm_result` with `ok: false`. Nothing is raised.
- Logging: each failed attempt is logged with `logger.warning` ("LLM fallback: %s/%s failed
  (%s), trying %s"), giving the HTTP status or error class but never the response body.
- Paid-model guard: config entries for OpenRouter that are neither `:free` ids nor exactly
  `openrouter/free` are dropped, and `call_provider` refuses them before any network call.
  `openrouter/auto` (which picks paid models) is refused; a test covers it.
- Timeouts: 10s to connect, 20s of silence per read, 60s total per provider call
  (`asyncio.wait_for`). A timeout is not retried; the model cools down for 30s and the next
  provider is tried.
- Trace labels use `candidate_label()`, which does not repeat the provider when the model id
  already starts with it (`nvidia/nemotron-...`, not `nvidia/nvidia/nemotron-...`).

## Provider details

- Raw httpx for all three providers (not LangChain wrappers). TTFT and usage are measured the same
  way everywhere, and the clients are easy to test with `httpx.MockTransport`.
- NVIDIA/OpenRouter: `/chat/completions`, with SSE streaming and
  `stream_options.include_usage` so streamed calls report tokens. For Nemotron on NIM the client
  sends `chat_template_kwargs.enable_thinking=false` (Nemotron reasons by default, which adds
  latency). OpenRouter calls send `reasoning.exclude=true` plus the optional
  `HTTP-Referer`/`X-Title` headers.
- Gemini: `models/{id}:generateContent` and `:streamGenerateContent?alt=sse` with the
  `x-goog-api-key` header. System messages become `systemInstruction`. Thought parts are dropped
  from the answer.
- `ttft_ms` is measured from the start of the successful attempt and is `None` for non-streaming
  calls. `latency_ms` covers the whole `complete()` call, including failed attempts.

## Eval hook

`ROUTER_FORCE_PROVIDER=nvidia|gemini|openrouter` moves that provider's models to the front of
every chain, which lets an eval run the same workload on each provider. The rest of the chain stays
as fallback. The reason then reads `override: ROUTER_FORCE_PROVIDER=... (policy would say: ...)`.
