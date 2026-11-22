# Cost Analysis

**Actual spend: $0.** Every model runs on a free tier (NVIDIA NIM free endpoints, Gemini API
free tier, OpenRouter `openrouter/free`), Tavily runs on its free plan,
and Cloud Run stays inside its monthly free allowance at grading traffic. Paid OpenRouter ids
are refused in code before any request is sent (`providers.call_provider`, tested).

The tables below say what the same traffic would cost on paid tiers, so the routing choices can
be compared on cost even though the bill is zero.

## Measured tokens per query (warm Cloud Run, run of 2026-10-27 00:18 UTC)

From `src/eval/results/latency_warm.json` (re-rank build; 18 queries against the deployed
service: 2 passes of 3 RAG, 2 web, 2 data and 2 summary queries). Token counts are summed over
every LLM call in the turn: supervisor plan, worker call(s), and for the data agent the
SQL-writing call(s) too. Averages and model counts were computed directly from the file's
`records` (`input_tokens`, `output_tokens`, `provider`/`model` per record); the per-query cost is
the mean of the per-record costs.

| Query type | n | Avg input tok | Avg output tok | Answering model(s) | List-price equivalent / query |
|---|---|---|---|---|---|
| rag | 6 | 2,281 | 367 | `gemini-3.5-flash-lite` (5), `nemotron-3-ultra-550b-a55b` (1, router fell back) | ~$0.0018 |
| web | 4 | 3,198 | 334 | `gemini-3.5-flash-lite` (4) | ~$0.0018 |
| data | 4 | 1,854 | 440 | `nemotron-3.5-lightning-30b-a3b` (4) | <= ~$0.0022 |
| summary | 4 | 2,242 | 691 | `openrouter/free -> xingchen-agi/xing4.0-29b-a4b:free` (3), `gemini-3.5-flash-lite` (1) | ~$0.0004 (the 3 OpenRouter rows are $0) |

The earlier embeddings-only build (`src/eval/results/baseline_no_rerank/latency_warm.json`, 9
queries, one pass of the same query set, 2026-10-26 22:14 UTC) measured 1,651 input / 243
output tokens per RAG query; the older version of this table was taken from that file.

These runs predate the behavioural changes of 2026-10-26 listed in `docs/evaluation.md` (only
cited sources published, relevance gate, alias pool, data decline); the benchmark was not re-run
after them.

List prices used (USD per 1M tokens, input / output, checked 2026-10-25/26; source and the
same table live in `src/eval/ragas_eval.py`):

| Model | Price | Note |
|---|---|---|
| `gemini-3.5-flash-lite` | $0.30 / $2.50 | Gemini API paid tier |
| `gemini-3.8-flash` | $0.75 / $3.75 | Gemini API paid tier |
| `nvidia/nemotron-3-ultra-550b-a55b` | $0.60 / $2.40 | OpenRouter paid listing of the same weights (NIM publishes no per-token price) |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | no listing | priced at the Ultra rate above as an upper bound |
| `openrouter/free`, `:free` ids | $0 | free by definition |

Each row applies the answering model's price to all of the turn's tokens, which overstates the
cost slightly (the supervisor plan runs on the cheaper Lightning model).

## What routing saves

- **Supervisor on NVIDIA Lightning (cost dimension).** Planning is a short JSON classification;
  it runs on every turn, so it goes to the cheapest fast model, not a strong one.
- **Summary on the OpenRouter free pool (cost).** Condensation of <= 20k chars goes to a $0
  pool that is separate from the Gemini and NVIDIA quotas, keeping Gemini's tight free-tier RPM
  for research.
- **Research on Gemini Flash-Lite unless complex (hybrid).** The 5 RAG queries Flash-Lite
  answered averaged 2,093 input / 324 output tokens, ~$0.0014 at list price; the same tokens on
  `gemini-3.8-flash` would cost ~$0.0028, about 1.9x as much.
- **Strong models only where capability matters**: complex research, long-context summary, and
  data questions returning many rows.

## Other components

| Component | Free allowance | Our usage |
|---|---|---|
| Tavily (search + extract via Remote MCP) | 1,000 credits / month | 1 credit per basic search or extract; a web or URL query uses 1-2 |
| Embeddings (Chroma default all-MiniLM-L6-v2) and re-ranker (ms-marco-MiniLM-L-6-v2), both ONNX | run locally in the container | $0 in API fees. The cost is CPU: re-ranking ~65 passages is 6-9 s of Cloud Run vCPU time per RAG query |
| Cloud Run (2 vCPU, 4 GiB, concurrency 5, min 0 / max 3 instances) | 180,000 vCPU-s and 360,000 GiB-s per month | a warm query bills ~3-20 s of instance time (~10 s typical with re-ranking), so about 9,000 queries a month at 2 vCPU before leaving the free tier |
| Artifact Registry | 0.5 GB free storage | one image of a few hundred MB plus older tags |
| Cloud Build | 2,500 build-minutes / month (e2-standard-2) | ~5 minutes per deploy |

Outside the free allowance, Cloud Run's request-based price (about $0.000024 per vCPU-second
plus $0.0000025 per GiB-second) works out to roughly $0.0001-0.0005 per query, below the
list-price LLM cost of the same query.

## Free-tier limits that actually bite

- **Gemini free tier**: `gemini-3.8-flash` hit both HTTP 503 ("high demand") and HTTP 429
  (daily quota) on 2026-10-26. The router treats both as "move on now" (no retry) and cools the
  model down across requests: `Retry-After` if given, else 30 s (429 capped at 120 s). In the
  embeddings-only benchmark this is why both web queries were answered by NVIDIA Ultra; in the
  18-query run one RAG query fell back to NVIDIA Ultra.
- **OpenRouter free models**: about 50 requests/day without purchased credits. On 2026-10-26
  every free endpoint returned 404 because the account's privacy setting allowed only
  zero-data-retention providers, which no free endpoint is (both summaries in the
  embeddings-only run were answered by Gemini Flash-Lite). In the 18-query run, 3 of 4 summaries
  were answered by `openrouter/free -> xingchen-agi/xing4.0-29b-a4b:free` and 1 by Gemini
  Flash-Lite. A 404, or a 402 when the account is out of credits, puts that model in a 120 s
  cooldown, so the router stops paying for the failed call on every request.
- **NVIDIA NIM**: free endpoints are rate limited per minute; not hit during testing.

## RAGAS evaluation cost

The judge runs on a free tier as well (NVIDIA NIM for the 2026-10-26 run, because the Gemini
strong model was out of daily quota). See `docs/evaluation.md` for the token totals of that run.
