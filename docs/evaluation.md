# Evaluation

> Status: all numbers below are from live runs on 2026-10-26/27 (commands in `src/eval/README.md`).
> Raw results are in `src/eval/results/`; the embeddings-only baseline is kept in
> `src/eval/results/baseline_no_rerank/`. Every result file predates the code changes made
> after the 2026-10-26 grading pass (section 9); RAGAS and the latency benchmark were **not**
> re-run after them.

## 1. Methodology

**What is measured.** The research (RAG) path, the same code that serves corpus questions in
production: `src.agents.research.run_research` with the production prompt, `retrieve()` (k=5,
Chroma + local ONNX all-MiniLM-L6-v2, then a cross-encoder re-rank with
`Xenova/ms-marco-MiniLM-L-6-v2`, max 2 chunks per paper) and `router.complete("research", ...)`.
The eval script wraps `retrieve` and `complete` only to *record* what passes through them. RAGAS
`retrieved_contexts` are the per-source blocks cut out of the exact prompt the LLM received.

**RAGAS setup (ragas 0.4.3, `.venv-eval`).**

| Item | Choice |
|---|---|
| Metrics | `Faithfulness`, `ResponseRelevancy` (strictness 3), `LLMContextPrecisionWithReference`, `LLMContextRecall` |
| Sample schema | `SingleTurnSample(user_input, retrieved_contexts, response, reference)` |
| Judge LLM | **NVIDIA NIM `nvidia/nemotron-3-ultra-550b-a55b`** (`--judge nvidia`, via `langchain-openai`), wrapped in `ragas.llms.base.LangchainLLMWrapper(bypass_n=True)`. The default judge, Gemini strong (`gemini-3.8-flash`), returned 503/429 (free-tier daily quota) on the run date. A side effect: the judge is a different model family from the generator (Gemini Flash-Lite) |
| Embeddings (answer relevancy) | Chroma `DefaultEmbeddingFunction` (ONNX MiniLM, local, no key) behind a LangChain `Embeddings` adapter |
| Run config | `RunConfig(max_workers=1, timeout=180, max_retries=6, max_wait=60)`, client-side rate limit 8 RPM, one question per `evaluate()` call, cached per question |
| Generation provider | Router policy (label `policy`). Provider comparison uses `ROUTER_FORCE_PROVIDER` |

**Latency setup.** `src/eval/latency_bench.py` posts to `/api/chat` and parses the SSE stream.
TTFT is measured client-side at the first `token` frame, and total is measured at the `done` frame.
Retrieval time comes from the `rag_retrieval` trace (`latency_ms`). Worker LLM latency and TTFT
come from the `agent_result` trace. *Other* is server total minus retrieval minus worker LLM, and
covers supervisor planning, web search/SQL tools and graph overhead. The cold request (first after
deploy/idle) is recorded on its own. Warm numbers are representative: median and p90 over 2 passes
of 9 queries (3 RAG, 2 web, 2 data, 2 summary), after one untimed warm-up request.

## 2. Fixed question set

Defined in `src/eval/questions.json` (15 questions). The reference answers are 1-3 sentences
paraphrased from text extracted from each paper's PDF with pypdf, not written from memory.

| ID | Paper (arXiv) | Question (abridged) |
|---|---|---|
| q01_react | ReAct (2210.03629) | How ReAct interleaves reasoning and acting; effect on HotpotQA/Fever |
| q02_reflexion | Reflexion (2303.11366) | Actor / Evaluator / Self-Reflection; learning without weight updates |
| q03_toolformer | Toolformer (2302.04761) | Loss-reduction filter for sampled API calls |
| q04_autogen | AutoGen (2308.08155) | ConversableAgent, AssistantAgent, UserProxyAgent roles |
| q05_metagpt | MetaGPT (2308.00352) | SOPs and the five software-company roles |
| q06_rag_lewis | RAG (2005.11401) | BART + DPR memories; RAG-Sequence vs RAG-Token |
| q07_selfrag | Self-RAG (2310.11511) | Retrieve / ISREL / ISSUP / ISUSE reflection tokens |
| q08_raptor | RAPTOR (2401.18059) | Recursive GMM clustering + summarization tree; traversal vs collapsed |
| q09_hyde | HyDE (2212.10496) | Hypothetical document -> Contriever embedding -> retrieval |
| q10_crag | CRAG (2401.15884) | Retrieval evaluator; Correct / Incorrect / Ambiguous actions |
| q11_graphrag | GraphRAG (2404.16130) | Why RAG fails on global questions; community summaries |
| q12_lost_middle | Lost in the Middle (2307.03172) | U-shaped dependence on the position of relevant info |
| q13_dpr | DPR (2004.04906) | Dual BERT encoders, in-batch negatives; +9-19% over BM25 top-20 |
| q14_tot | Tree of Thoughts (2305.10601) | Search over thoughts; Game of 24: 4% CoT vs 74% ToT |
| q15_ragas | RAGAS (2309.15217) | Faithfulness / answer relevance / context relevance; faithfulness estimate |

All 15 questions route to `research` even under the supervisor's keyword fallback (a test checks
this), so the eval measures the RAG path and not the data or summary workers.

## 3. Results

### 3.1 Retrieval only (offline, embeddings only, generated 2026-10-26T00:37Z)

`.venv-eval/bin/python -m src.eval.ragas_eval --retrieval-only` (no LLM). These numbers come from
`src/eval/results/retrieval_only.json` (`generated_at: 2026-10-26T00:37:48+00:00`), which was
generated **before the cross-encoder re-ranker was added**, and before the paper-alias pool
change. They describe embeddings-only retrieval, not the shipped retriever, and have not been
re-run since.

| Retrieval (k=5) | Value |
|---|---|
| Expected paper in top-k (hit rate) | 1.000 (15/15) |
| MRR of first expected paper | 0.967 |
| Median retrieval latency (ms, local laptop, warm) | 49.9 |

In that run the only question where the expected paper is not ranked first is q14 (Tree of
Thoughts): the agent survey 2308.11432 ranks first, and ToT ranks 2nd and 3rd.

### 3.2 RAGAS (generation: router policy, judge: NVIDIA Nemotron 3 Ultra)

`.venv-eval/bin/python -m src.eval.ragas_eval --judge nvidia --judge-rpm 8 --max-workers 1`

Two runs over the same 15 questions with the same prompt, generator and judge. The only change is
retrieval: embeddings only (baseline), then embeddings + cross-encoder re-rank (shipped).

| Metric | Embeddings only | + Re-rank (shipped) | Change |
|---|---|---|---|
| Faithfulness | 0.915 (15/15 scored) | 0.906 (14/15; q14 judge NaN) | -0.01 |
| Answer relevancy (ResponseRelevancy) | 0.722 (15/15) | **0.824** (15/15) | +0.10 |
| Context precision (LLMContextPrecisionWithReference) | 0.352 (15/15) | **0.545** (15/15) | +0.19 |
| Context recall (LLMContextRecall) | 0.300 (15/15) | **0.433** (15/15) | +0.13 |

Per question (baseline → re-rank):

| Question | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
|---|---|---|---|---|
| q01_react | 0.59 → 0.90 | 0.85 → 0.83 | 0.00 → 0.00 | 0.00 → 0.00 |
| q02_reflexion | 0.91 → 1.00 | 1.00 → 0.97 | 1.00 → 1.00 | 0.50 → 0.50 |
| q03_toolformer | 1.00 → 1.00 | 0.99 → 0.85 | 1.00 → 1.00 | 0.00 → 0.00 |
| q04_autogen | 1.00 → 1.00 | 0.00 → 0.80 | 0.00 → 0.00 | 0.00 → 0.50 |
| q05_metagpt | 0.88 → 0.71 | 0.80 → 0.87 | 0.70 → 0.76 | 0.00 → 0.50 |
| q06_rag_lewis | 0.75 → 0.50 | 0.90 → 0.00 | 0.00 → 0.00 | 0.50 → 0.00 |
| q07_selfrag | 1.00 → 0.83 | 0.81 → 0.90 | 0.00 → 0.00 | 0.00 → 0.00 |
| q08_raptor | 1.00 → 1.00 | 0.96 → 1.00 | 0.00 → 0.00 | 0.00 → 0.50 |
| q09_hyde | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| q10_crag | 1.00 → 0.80 | 0.97 → 0.99 | 0.25 → 1.00 | 0.50 → 0.00 |
| q11_graphrag | 1.00 → 1.00 | 0.00 → 0.99 | 1.00 → 1.00 | 0.00 → 1.00 |
| q12_lost_middle | 0.60 → 1.00 | 0.74 → 0.69 | 0.33 → 1.00 | 1.00 → 1.00 |
| q13_dpr | 1.00 → 1.00 | 0.84 → 0.84 | 0.00 → 0.00 | 0.00 → 0.00 |
| q14_tot | 1.00 → n/a | 0.97 → 0.91 | 0.00 → 1.00 | 0.50 → 1.00 |
| q15_ragas | 1.00 → 0.94 | 0.00 → 0.72 | 0.00 → 0.42 | 0.50 → 0.50 |

**Why the baseline was weak.** Paper-level retrieval was already perfect (hit@5 15/15), but the
chunks it picked from the right paper were often the wrong *section*. For q04 the MiniLM embedding
ranked AutoGen's "2.1 Conversable Agents" (which defines the agent types) 49th inside its own
paper; the top hits were Related Work and Appendix B. The generator then correctly answered "the
sources do not say", which ResponseRelevancy scores 0. That pattern explains all three 0.00
answer-relevancy rows in the baseline (q04, q11, q15).

**What changed.** `retrieve()` now builds a pool of the 50 best global chunks plus the 10 best
chunks inside each of the top 3 papers (~65 chunks). A cross-encoder that reads the query and
the passage together re-scores the pool, and the per-paper cap is applied afterwards
(`src/rag/retriever.py`). The re-ranker fixed q04, q11 and q15, and raised precision on q10, q12
and q14.

**Remaining weak spots.**
- **q06 (RAG, Lewis et al.) regressed to a non-answer.** The re-ranker preferred survey chunks that
  discuss RAG over the original paper's RAG-Sequence/RAG-Token definitions.
- **Zero context recall on q01, q03, q07 and q13.** The method passages (for example Toolformer's
  loss filter in "2 Approach") are in the index but do not reach the pool of 10 per paper.
- **Next steps.** A larger per-paper pool would help, but costs re-rank CPU time (section 4.2).
  HyDE-style query expansion is another option.

## 4. Latency

### 4.1 Cold start (first request after deploy or scale-to-zero)

`.venv/bin/python -m src.eval.latency_bench --cold --url <CLOUD_RUN_URL>`

| Cold request | TTFT | Total | Retrieval | Worker LLM | LLM TTFT | Other | Provider/model |
|---|---|---|---|---|---|---|---|
| rag (embeddings-only build, 1 vCPU, 2026-10-26 22:36 UTC) | 16,788 | 17,440 | 410 | 1,152 | 505 | 995 | gemini/gemini-3.5-flash-lite |
| rag (shipped re-rank build, 2 vCPU, 2026-10-27 00:53 UTC) | 35,641 | 36,453 | 8,396 | 1,197 | 474 | 1,043 | gemini/gemini-3.5-flash-lite |

Cloud Run logs for the first sample show the breakdown. "Starting new instance" was logged at
22:36:20.8, `Started server process` at 22:36:30.8 (about 10 s of container start and image
load), and `Application startup complete` at 22:36:35.4 (about 4.5 s of imports, opening Chroma
and warming the ONNX embedder). That leaves about 2.5 s for the request itself. The shipped build's cold start (second row) breaks down as follows.
"Starting new instance" was logged at 00:53:20.9, `Started server process` at 00:53:32.8 (about
12 s of container start for the larger image), and `Application startup complete` at 00:53:46.2.
That 13.5 s of startup now includes loading and warming the cross-encoder, and the request itself
took about 10.6 s. The request succeeded: SSE heartbeats keep the connection open while the
instance starts. `--min-instances 1`
would remove the cold start, but it bills an idle instance around the clock, so the service keeps
`min 0`.

### 4.2 Warm (representative)

`.venv/bin/python -m src.eval.latency_bench --url <CLOUD_RUN_URL> --warmup 1 --repeat 2` (ms,
median / p90). Shipped configuration: re-rank on, 2 vCPU, 4 GiB, concurrency 5. 18/18 requests ok
(2026-10-27 00:18 UTC).

| Query type | n | TTFT | Total | Retrieval | Worker LLM | LLM TTFT | Other |
|---|---|---|---|---|---|---|---|
| all | 18 | 9,167 / 13,785 | 11,044 / 18,736 | 6,605 / 8,146 | 2,298 / 14,867 | 634 / 6,090 | 904 / 1,252 |
| rag | 6 | 9,835 / 12,558 | 11,171 / 15,228 | 7,881 / 8,685 | 1,703 / 6,138 | 689 / 2,913 | 881 / 1,195 |
| web | 4 | 8,595 / 10,156 | 9,405 / 10,853 | 6,391 / 6,896 | 1,439 / 1,641 | 595 / 797 | 964 / 1,124 |
| data | 4 | 2,375 / 11,247 | 9,535 / 15,136 | n/a | 4,405 / 9,523 | 230 / 246 | 739 / 10,039 |
| summary | 4 | 10,479 / 13,228 | 19,301 / 22,477 | 2,545 / 3,131 | 15,523 / 18,331 | 6,702 / 8,931 | 1,008 / 1,069 |

Same benchmark on the embeddings-only build (1 vCPU, 2 GiB, 9/9 ok,
`results/baseline_no_rerank/latency_warm.json`):

| Query type | n | TTFT | Total | Retrieval |
|---|---|---|---|---|
| all | 9 | 2,353 / 7,680 | 3,312 / 15,829 | 447 / 906 |
| rag | 3 | 1,979 / 1,980 | 2,457 / 2,587 | 418 / 441 |

**The quality/latency trade-off was chosen deliberately.**
- **Cost of re-ranking.** It moves RAG time to first token from about 2 s to about 10 s. Re-scoring
  ~65 passages with the cross-encoder takes 1.0 s on a laptop core but 6-9 s on Cloud Run vCPUs.
- **Why it ships anyway.** It raises context precision by 0.19 and answer relevancy by 0.10, so the
  re-ranker stays on. `RERANK_MODEL=""` turns it off without a code change.

**Three problems found by benchmarking and fixed:**
1. **ONNX thread oversubscription.** Retrieval took 15-27 s on 1 vCPU. ONNX sized its thread pool
   from the host instead of the container, so threads are now pinned with `RERANK_THREADS` and the
   service runs on 2 vCPU.
2. **Out of memory.** The re-ranker's default batch of 64 raised peak memory by ~1.45 GB. The
   2 GiB instance was OOM-killed ("Memory limit of 2048 MiB exceeded"), which caused one failed
   request and a 36 s wait for a replacement instance. Batches of 8 add ~0.3 GB, and the service
   now has 4 GiB with concurrency 5.
3. **Slow failover.** A 45 s read timeout on an overloaded Gemini model made complex RAG queries
   take 51 s to first token. Timeouts now fail over at once, and the model cools down for 30 s
   across requests.

**Other observations.**
- **Summary rows.** The LLM column is high when `openrouter/free` answers (it routed to
  `xingchen-agi/xing4.0-29b-a4b:free`, 9-13 s). Its cost is $0, and Gemini Flash-Lite answers the
  same request in about 1.8 s when OpenRouter is rate limited.
- **Data p90.** The data agent's p90 includes NVIDIA Lightning outliers of 9-10 s on the
  SQL-writing call.

## 5. Cost per query

**Actual cost: $0.** Every provider runs on a free tier (Gemini free tier, NVIDIA NIM free API
endpoints, OpenRouter `:free` models). Embeddings are local and Tavily runs on the free plan.

**List-price equivalent** (prices checked 2026-10-25, USD per 1M tokens, input / output):

| Model | Used for | List price | Source |
|---|---|---|---|
| gemini-3.5-flash-lite | research (simple), summary (long context) | $0.30 / $2.50 | ai.google.dev/gemini-api/docs/pricing (paid tier) |
| gemini-3.8-flash | research (complex), RAGAS judge | $0.75 / $3.75 (until 2026-12-31; $1.50 / $7.50 from 2027-01-01) | same |
| nvidia/nemotron-3-ultra-550b-a55b | data (complex), fallback | $0.60 / $2.40 (OpenRouter paid listing, proxy) | openrouter.ai/api/v1/models |
| nvidia/nemotron-3.5-lightning-30b-a3b | supervisor, data (simple) | no public per-token price | build.nvidia.com (free endpoint) |
| openrouter/free | summary (router) | $0 (routes only to `:free` models) | openrouter.ai/api/v1/models |

**Offline estimate for one RAG query** (made before tokens were measured; the measured values
are in the next table): the research prompt is
about 5.5k characters (median over the 15 questions, system prompt + 5 chunks), which is about
1.4k input tokens. All 15 route as "simple", so research goes to gemini-3.5-flash-lite. With an
assumed ~300 output tokens: 1,400 x $0.30/1M + 300 x $2.50/1M ≈ **$0.0012 per query** at list
price, plus a supervisor call on Nemotron Lightning (~300 tokens, no listed price, well under
$0.0001 at comparable rates).

| Measured (from `generation_stats` in `ragas_scores.json`, re-rank run) | Value |
|---|---|
| Mean input tokens (research call) | 1,278 |
| Mean output tokens (research call) | 188 |
| Mean list-price cost per query | $0.00085 |
| RAGAS judging cost (about 10 judge calls per question) | $0 actual (NVIDIA NIM free endpoint) |

Whole-turn token counts per query type on the deployed service (supervisor plus all worker
calls) are in `docs/cost_analysis.md`.

Formula used by the script: `cost = in_tokens x in_price/1e6 + out_tokens x out_price/1e6`
(`ragas_eval.estimate_cost_usd`). The `:free` suffix is stripped to look up the paid listing.

## 6. Provider comparison

Same 5 questions (`--limit 5`, q01-q05), same retrieval (re-rank on) and prompt, primary forced
with `ROUTER_FORCE_PROVIDER`, judge fixed (NVIDIA Nemotron 3 Ultra). Rows where the router fell
back to another provider are listed in `generation_stats.forced_provider_not_honored`.
Results: `src/eval/results/ragas_scores_<provider>.json` (2026-10-27).

| Provider (model that answered) | Faithfulness | Answer rel. | Ctx precision | Ctx recall | Median LLM latency (ms) | Mean out tokens | List $/query | Fallbacks |
|---|---|---|---|---|---|---|---|---|
| NVIDIA NIM (`nemotron-3.5-lightning-30b-a3b`) | 0.823 | 0.844 (3/5 scored) | 0.551 | 0.460 | 3,106 | 209 | no listed price | 0/5 |
| Gemini (`gemini-3.5-flash-lite`) | 0.976 | 0.856 (4/5 scored) | 0.551 | 0.427 | 1,326 | 188 | $0.00086 | 0/5 |
| OpenRouter forced, all 5 rows as scored: **mixed**, 2 rows `openrouter/free` -> `xingchen-agi/xing4.0-29b-a4b:free` (q01, q02) + 3 rows Gemini `gemini-3.5-flash-lite` fallback (q03-q05) | 0.983 | 0.841 | 0.551 | 0.520 | 1,859 (a Gemini row) | 530 | $0.00073 (`mean_list_price_cost_usd`, which covers only the 3 Gemini rows; the OpenRouter rows have no price and cost $0) | **3/5 fell back to Gemini** |
| OpenRouter's own 2 rows only (q01, q02; computed from `ragas_scores_openrouter.json` / `generations_openrouter.json`) | 0.958 | 0.826 | 0.500 | 0.550 | 17,930 / 41,365 (the two calls) | 1,112 | $0 | 0/2 |

**Reading the table.**
- **Context precision is identical** (0.551) across the three 5-row aggregates, as it should
  be: it depends only on retrieval and the reference. Recall varies by 0.09, which is judge
  noise on the same contexts.
- **The OpenRouter aggregate row is not an OpenRouter measurement.** 3 of its 5 rows were
  answered by Gemini Flash-Lite, and its median latency (1,859 ms) is one of those Gemini rows.
  The two rows OpenRouter actually answered took 17.9 s and 41.4 s (LLM call latency, as
  recorded in `generations_openrouter.json`) and averaged 1,112 output tokens.
- **Gemini Flash-Lite is the best all-rounder.** It has the highest faithfulness among rows that
  were really its own and the lowest latency (1.3 s), which is why the policy routes simple
  research to it.
- **NVIDIA Lightning is less faithful.** It scored 0.82, adding claims the sources do not support,
  and it was 2.3x slower. It stays on the cheap, short tasks (supervisor plan, SQL writing), where
  faithfulness to sources is not at stake.
- **OpenRouter's free pool is not reliable enough to be primary for research.** On q03-q05 all
  three OpenRouter entries tried at the time (`openrouter/free` plus two pinned `:free` ids, since removed) returned HTTP 404 (`"0 endpoints ... available matching your
  guardrail restrictions and data policy"`, the account's zero-data-retention setting), and the
  router fell back to Gemini, so only 2 rows reflect OpenRouter itself. This is why the policy
  uses it for summary condensation, behind a cross-provider fallback.

## 7. Limitations

- **Small, single-domain question set.** 15 questions with one main paper each. This is enough to
  compare configurations, but not to make claims with tight confidence intervals.
- **Single judge run per configuration.** The judge (Nemotron 3 Ultra) is a different family from
  the generator (Gemini Flash-Lite), which avoids self-preference. Each configuration was judged
  once, so run-to-run judge noise was not measured. Individual per-question changes (e.g. q10
  recall 0.50 -> 0.00) may be judge noise; the mean changes are the more reliable signal.
- **Free-tier noise.** Rate limits (~10 RPM on Gemini) cause retries and router fallbacks. Latency
  includes queueing at the provider, and a fallback changes which model answered. Both are
  recorded per row.
- **Generation is non-streaming in the RAGAS run.** Production TTFT comes only from the latency
  bench.
- **RAGAS metrics are LLM-judged.** Faithfulness and both context metrics depend on how the judge
  splits statements. A metric can come back NaN when the judge returns unparseable JSON. Those rows
  are reported as "Failed" and retried on rerun, and they are never counted as 0.
- **References are paraphrases.** Context recall checks reference statements against the
  retrieved chunks, so a reference that paraphrases loosely can lower recall even when retrieval
  was correct.
- **Cold start depends on the platform.** Cloud Run cold start varies with region, image size
  and CPU boost. It is a single sample, not a distribution.

## 8. Gaps found by the eval work (all fixed)

- Token counts now reach the `agent_result` trace for every worker, and the supervisor's
  `agent_route` trace carries `planner_latency_ms` and planner tokens.
- The data agent reports `llm_latency_ms`, `ttft_ms` and tokens summed over its SQL-writing and
  answer calls.

## 9. Changes after the 2026-10-26 grading pass (not yet measured)

The following behavioural fixes went into the code after every result file in
`src/eval/results/` was written (file times: results up to 2026-10-27 00:53 UTC, code changes
after that). **RAGAS was not re-run after them, and neither were the retrieval-only and latency
benchmarks.** The numbers in sections 3-6 therefore describe the code before these changes.
Changes that could move the metrics:

- **Paper-name aliases in the re-rank pool** (`src/rag/paper_aliases.py`,
  `build_rerank_pool`). A paper the query names is expanded first in the pool, before the top 3
  papers from the embedding search. 14 of the 15 eval questions name their paper through an
  alias (all but q06, "RAG (Lewis et al.)"); q14 also matches Chain-of-Thought. This changes which
  chunks reach the re-ranker, so context precision and recall can move either way.
- **Relevance gate in research** (`MIN_RELEVANT_SCORE = 0.05` on the cross-encoder score).
  Chunks below it are dropped, so a RAGAS row can have fewer than 5 `retrieved_contexts`; if no
  chunk passes, research asks the web instead (`web_reason: "no relevant corpus chunk"`).
- **Research prompt.** The "sources do not say" sentence now carries no citation marker, and the
  agent answers only its own sub-task. This can change faithfulness and answer relevancy.
- **Prompt source separator.** Source blocks are joined by `"\n\n\n"` instead of `---`
  (`SOURCE_BLOCK_SEPARATOR`); `ragas_eval.SOURCES_SEPARATOR` splits on the same string.
- **Only cited sources are published** (`select_cited`), and an unknown `[n]` is reported as an
  `error` trace. This changes the citation panel, not the RAGAS contexts, which are cut from the
  prompt.
- **Router.** HTTP 503 is no longer retried (30 s cooldown), 402 gets the 120 s cooldown like
  404, and one call stops after a 90 s budget. This affects latency and which model answers
  under free-tier pressure, not the RAG prompt.
- **Supervisor and other workers** (outside the RAGAS path, inside the latency benchmark): data
  is dropped from multi-worker plans that name no paper metadata, summary is appended when a
  2-worker request asks for a TL;DR, a worker streams only if no summary step comes later, the
  data agent declines non-metadata requests (`NOT_APPLICABLE`) and gets one repair attempt on an
  empty result, summary takes URLs and its retrieval query from the user's own words, and
  `finish` adds a note line for a failed step.
- **OpenRouter chain is `openrouter/free` only.** The two pinned `:free` backups were removed
  from the code defaults, `.env.example` and the live service (revision `multi-agent-rag-00010`).
  An OpenRouter failure now goes straight to NVIDIA or Gemini instead of two calls that 404.

## 10. Live check of the deployed service (2026-10-26, revision `multi-agent-rag-00010`)

A manual smoke run, not a benchmark: one request per question against the live URL, read from the
SSE stream (`done` frame and traces). Every request ended with exactly one `done` frame, no failed
step and no `error` frame.

| Question | Plan | Answered by | TTFT / total (s) | Result |
|---|---|---|---|---|
| What is ColBERT late interaction? | research | gemini-3.5-flash-lite | 12.8 / 13.2 | Correct; cites the ColBERT paper |
| What are the latest AI model releases this week? | research (Tavily search) | gemini-3.5-flash-lite | 11.5 / 12.2 | Current releases, 2 web citations |
| How many papers are about multi-agent systems, by year? | data | nemotron-3.5-lightning | 2.6 / 5.4 | 120 papers, per-year table |
| Summarize the Reflexion paper | summary | gemini-3.5-flash-lite (OpenRouter 404) | 4.9 / 5.8 | Correct paper |
| Summarize a Wikipedia URL, then explain how RAG differs from fine-tuning | summary (Tavily extract) | gemini-3.5-flash-lite | 4.7 / 5.1 | **Half answered**, see below |
| Compare ReAct and Reflexion and give a TL;DR | research -> summary | gemini-3.5-flash-lite | 12.1 / 12.6 | One TL;DR, one section |
| Why is the sky blue? | research (Tavily search) | gemini-3.5-flash-lite | 15.3 / 15.8 | 4 web citations |
| Summarize the Self-RAG paper | summary | `openrouter/free -> dots-studio/dots-3-note-preview:free` | 21.4 / 29.5 | Correct; OpenRouter answered |
| Summarize the HyDE paper | summary | gemini-3.5-flash-lite (OpenRouter empty completion) | 27.9 / 28.9 | Correct, after fallback |

**Findings.**
- **OpenRouter account setting.** Until the account's zero-data-retention setting was turned off,
  `openrouter/free` returned `404 "0 endpoints out of 9 requested are available matching your
  guardrail restrictions and data policy"`, and Gemini answered every summary. After the change it
  answered (Self-RAG row), but its free models were slow (about 20 s to first token) and one
  returned an empty completion, which the router treated as a failure.
- **Known gap: a URL summary plus a follow-up question.** The planner (Nemotron Lightning) planned
  only `summary` for "summarize <URL>, then explain how it differs from fine-tuning". No step
  answered the second part, the summary said the comparison was missing, and it cited a `[3]` that
  no source had (reported as an `error` trace). No plan rule adds research after a URL summary
  yet. This is not fixed in the submitted build.
- **Gemini strong model quota.** `gemini-3.8-flash` returned `429 "You exceeded your current
  quota"` on the live provider test that day (free daily quota). Calls routed to it fell back to
  NVIDIA or Gemini Flash-Lite.
