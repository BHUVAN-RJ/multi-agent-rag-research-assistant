# Multi-Agent RAG Research Assistant (CSCI 599, Assignment 2)

A LangGraph **supervisor** multi-agent system that answers questions about LLM agents,
multi-agent systems and retrieval-augmented generation. It combines:

- **RAG** over 44 arXiv papers: Chroma with local ONNX embeddings, cited answers with arXiv
  page and section.
- **Live web knowledge** through the Tavily Remote MCP server, with a DuckDuckGo fallback.
- **Structured queries** over a SQLite database of 579 arXiv paper records (metadata only).
- **LLM routing** across three free providers (NVIDIA NIM, Google Gemini, OpenRouter `:free`),
  with cross-provider fallback.
- A FastAPI **SSE** backend and a vanilla-JS UI that streams tokens and shows the routing
  decisions, a trace log and citations.

Deployed on Google Cloud Run.

## Architecture in one paragraph

`supervisor` makes one cheap LLM call that returns an ordered plan of workers (with a keyword
classifier as fallback), then routes on the state field `next_agent`. The workers are:

- **research**: Chroma retrieval (k=5), plus Tavily search when the question needs live data.
- **summary**: Tavily extract for URLs, earlier workers' findings, or paper-scoped retrieval.
- **data**: read-only SQL over `papers.db`.

Each worker returns to the supervisor. A failed worker becomes a failed output and triggers a
recovery worker; it never crashes the graph. Only the final worker streams tokens. The same
Tavily MCP client is shared by research and summary, and every call is traced with its `caller`.
The system is stateless: each request is independent.

Full design, with diagrams, state schema, routing table, RAG pipeline and failure handling:
**[docs/architecture.md](docs/architecture.md)**.

## Provider requirements (all free)

The full setup uses four keys, none of which needs a credit card or paid credits. The minimum is
one LLM key: `deploy.sh` and the router skip any provider without a key (with a warning), and
web search falls back to DuckDuckGo without a Tavily key.

| Provider | Used for | How to get a free key |
|---|---|---|
| **NVIDIA NIM** | Supervisor, data agent (Nemotron) | Sign in at [build.nvidia.com](https://build.nvidia.com) and join the free NVIDIA Developer Program. Open any model page, choose "Get API Key", and copy the `nvapi-...` key |
| **Google Gemini** | Research agent, long-context summaries, RAGAS judge | [aistudio.google.com](https://aistudio.google.com) -> "Get API key" -> create a key. Free-tier limits are shown per project in AI Studio |
| **OpenRouter** | Summary agent (`:free` models) | Create an account at [openrouter.ai](https://openrouter.ai) -> Keys -> create a key. **No credits are needed.** The app calls only `openrouter/free` (OpenRouter's free-only router) and refuses any paid id in code before any request is sent. In [privacy settings](https://openrouter.ai/settings/privacy), allow free endpoints that may log or train on prompts, or every free model returns 404 |
| **Tavily** | Live web search / page extraction via MCP | Sign up at [tavily.com](https://tavily.com). The free plan includes about 1,000 API credits per month. Without a key, the app falls back to DuckDuckGo |

Embeddings and re-ranking need no key: both run locally on ONNX (Chroma's all-MiniLM-L6-v2 and the
`Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed, both built into the image).
Re-ranking is on in the deployed build, and warm RAG time-to-first-token on Cloud Run has a median
of about 9.8 s (`src/eval/results/latency_warm.json`). `RERANK_MODEL=""` turns re-ranking off, which
brings that median down to about 2.0 s (`src/eval/results/baseline_no_rerank/latency_warm.json`) at
the cost of the retrieval gains in `docs/evaluation.md`.

If a key is missing, that provider is skipped and the router falls through to the next provider
in the chain. The app still works with any one LLM key, but you then lose the routing diversity.

## Local setup

Requires Python 3.13.

```bash
# 1. Environment (uv or plain venv)
uv venv .venv --python 3.13        # or: python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # with uv: uv pip install -r requirements-dev.txt

# 2. Keys
cp .env.example .env               # then fill in the four keys; leave model ids as shipped

# 3. Data (one-time; the outputs are gitignored and get baked into the Docker image)
python -m src.rag.corpus_fetch     # downloads the 44 arXiv PDFs + data/corpus/metadata.json (~3 s between downloads)
python -m src.rag.ingest           # PDFs -> cleaned chunks -> chroma_index/ (rebuilds from scratch)
python -m src.data.build_db        # arXiv API -> data/papers.db (rebuilds from scratch)

# 4. Run
uvicorn main:app --reload --port 8080
# open http://localhost:8080
```

`python main.py` also works (it binds to `$PORT`, default 8080).

## Tests

```bash
.venv/bin/python -m pytest -q                 # everything; live tests skip when keys are absent
.venv/bin/python -m pytest -q -m "not live"   # offline only (mocked providers/tools)
```

The suite is in `tests/`: `test_agents.py`, `test_data.py`, `test_eval.py`, `test_frontend.py`,
`test_graph.py`, `test_rag.py`, `test_router.py`, `test_server.py`, `test_tools.py`. Tests marked
`live` call real providers and need keys plus network. Run `pytest -m "not live"` for the
offline suite (all pass) and `pytest -m live` for the live provider and Tavily checks.

## Deploy to Cloud Run (fresh GCP project)

### Prerequisites

1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install), then authenticate:
   ```bash
   gcloud auth login
   gcloud projects create <your-project-id>        # or use an existing project
   gcloud config set project <your-project-id>
   ```
2. **Link a billing account** to the project (Console -> Billing; USC/education credit works).
   Cloud Build and Cloud Run refuse to run without billing, even within the free tier.
3. Complete **Local setup** steps 1-3 (the venv from step 1 is needed to build the data).
   `chroma_index/` and `data/papers.db` must exist, because they are baked into the image.
4. In `.env`, set `GCP_PROJECT_ID`. `GCP_REGION` (default `us-west1`) and `CLOUD_RUN_SERVICE`
   (default `multi-agent-rag`) are optional.

### Run

```bash
./deploy.sh
```

### What `deploy.sh` does

1. Loads `.env`. It fails if `GCP_PROJECT_ID` is missing or still a placeholder, or if none of
   the LLM keys (`GOOGLE_API_KEY`, `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`) is set. A missing LLM
   key, a missing `TAVILY_API_KEY` (DuckDuckGo stands in) and missing model ids only produce
   warnings.
2. Checks that `chroma_index/` (with `chroma.sqlite3`), `data/papers.db` and `static/index.html`
   exist.
3. Enables `run.googleapis.com`, `cloudbuild.googleapis.com`, `artifactregistry.googleapis.com`
   and `compute.googleapis.com` before any other gcloud call touches the project. The script
   exports `CLOUDSDK_CORE_DISABLE_PROMPTS=1`, so gcloud fails instead of waiting at a prompt.
4. Refuses to touch an existing Cloud Run service that lacks the `assignment=csci599-a2` label
   (the GCP project is shared with Assignment 1).
5. Creates the Artifact Registry Docker repo `cloud-run-images` in the region, if it is missing.
6. Grants the build service account (`<project-number>-compute@developer.gserviceaccount.com`)
   these roles: `roles/cloudbuild.builds.builder`, `roles/storage.objectViewer`,
   `roles/artifactregistry.writer` and `roles/logging.logWriter`. Set `SKIP_IAM=1` to skip this
   step. A failed grant only produces a warning.
7. Builds the image with `gcloud builds submit`. `.gcloudignore` keeps the index and DB and drops
   `.env`, the venvs and the PDFs.
8. Writes the keys to a private temp env-vars file (deleted on exit), so keys never appear on the
   command line or in the image.
9. Runs `gcloud run deploy` with
   `--allow-unauthenticated --memory 4Gi --cpu 2 --concurrency 5 --timeout 300 --min-instances 0
   --max-instances 3 --cpu-boost --labels assignment=csci599-a2`.
10. Prints the live URL plus `curl` commands for `/health` and a streaming smoke test.

### IAM roles for the person running the script

Project **Owner** covers everything. For least privilege, you need:

- `roles/serviceusage.serviceUsageAdmin` (enable APIs)
- `roles/artifactregistry.admin` (create the repo)
- `roles/cloudbuild.builds.editor` (submit builds)
- `roles/storage.admin` (`gcloud builds submit` uploads the source to the
  `gs://<project-id>_cloudbuild` bucket and creates that bucket on the first build;
  `roles/storage.objectAdmin` is enough once the bucket exists)
- `roles/viewer` (`gcloud projects describe` in step 6 reads the project number, and
  `gcloud builds submit` streams the build log)
- `roles/run.admin` (deploy and allow unauthenticated access)
- `roles/iam.serviceAccountUser` (deploy as the default compute service account)
- `roles/resourcemanager.projectIamAdmin` (only for step 6; otherwise use `SKIP_IAM=1` and have an
  admin grant those roles)

If the project is in an organization that blocks `allUsers` (the domain-restricted sharing policy
`iam.allowedPolicyMemberDomains`), `--allow-unauthenticated` fails. In that case, change it to
`--no-allow-unauthenticated` in `deploy.sh`, grant `roles/run.invoker` to the people who need
access, and call the service with
`-H "Authorization: Bearer $(gcloud auth print-identity-token)"`.

The runtime service account needs no extra roles: the service calls no GCP APIs at runtime.

### Verify

```bash
curl -s https://<service-url>/health          # {"status":"ok"}  (use /health; Cloud Run reserves /healthz)
curl -N -X POST https://<service-url>/api/chat -H 'Content-Type: application/json' \
     -d '{"query":"How does ReAct interleave reasoning and acting?"}'
```

Then open the URL in a browser and run one query from each path below. Check that the routing
card and trace log show the agent, the provider/model, and the reason for each decision.

## Submission zip

Run this from the repository root. It leaves out secrets, virtualenvs, caches, git metadata, the
generated index, the corpus PDFs and the assignment PDF. `data/corpus/metadata.json` stays in, because
the eval question set and its tests check against it:

```bash
zip -r ../AAI-Assignment2.zip . \
    -x '.env' '.venv/*' '.venv-eval/*' '*__pycache__*' '.pytest_cache/*' '.git/*' \
       'chroma_index/*' 'data/corpus/*.pdf' 'Assignment_2_Description.pdf' '*.DS_Store'
```

`.env.example` stays in the zip. `chroma_index/` and the corpus PDFs can be rebuilt with the
**Local setup** step 3 commands.

## Example queries

These are also clickable in the UI.

| Path | Query | Expected route |
|---|---|---|
| RAG (corpus) | How does ReAct interleave reasoning and acting? | supervisor -> research. Corpus citations `[n]` link to arXiv with page and section |
| Live web | What are the latest developments in AI agents this week? | research with `needs_web`. `mcp_tool_call` trace with `caller=research` and `(web)` citations |
| Data (SQL) | How many RAG papers were published per year since 2020? | supervisor -> data. `data_query` trace, markdown table, SQL shown |
| Summary (paper) | Summarize the key ideas of Self-RAG | supervisor -> summary. Paper-scoped retrieval, TL;DR + bullets |
| Summary (URL, shared MCP) | Summarize https://arxiv.org/abs/2310.11511 | summary. `tavily_extract` with `caller=summary` |
| Multi-step | Find what the corpus says about multi-agent debate, then summarize it | Usually research -> summary (the plan comes from the LLM; only summary streams) |

## Cost and access (as of October 2026)

Every component runs on a free tier. Limits are approximate and change often, so check each
provider's page before relying on them.

| Component | Cost | Approximate free limits |
|---|---|---|
| NVIDIA NIM (build.nvidia.com) | $0 | About 40 requests/min per key on the free developer endpoints |
| Google Gemini (AI Studio) | $0 | Per-model RPM/RPD shown in AI Studio. Flash-Lite and Flash have free tiers |
| OpenRouter `:free` models | $0 | Small per-minute and per-day request caps without purchased credits. Paid models are refused in code |
| Tavily | $0 | About 1,000 credits/month on the free plan. Falls back to DuckDuckGo when exhausted |
| Embeddings (local ONNX) | $0 | No limits; runs in-process |
| Cloud Run / Cloud Build / Artifact Registry | Usually $0 at coursework scale | Cloud Run's monthly free tier covers light traffic with `min-instances 0`. Image storage beyond the Artifact Registry free allowance costs cents/month. Billing must be enabled |

The router turns a 429 into an immediate move to the next provider plus a short cooldown. Running
out of one free quota therefore slows a request but does not fail it.

## Evaluation

The RAGAS metrics (faithfulness, answer relevance, context precision, context recall), latency
(TTFT, cold vs warm) and the cross-provider comparison are reported in
[docs/evaluation.md](docs/evaluation.md). The harness lives in `src/eval/`; see
[src/eval/README.md](src/eval/README.md) for how to run it. `ROUTER_FORCE_PROVIDER=nvidia|gemini|openrouter`
pins the primary provider so the same workload can be compared across providers.

## Repository layout

```text
main.py                  FastAPI app: POST /api/chat (SSE), /health, static UI
src/
  config.py              all settings (read from .env)
  trace_events.py        token / trace / citation stream events
  graph.py               LangGraph supervisor graph, AgentState, stream_query()
  agents/                supervisor.py, research.py, summary.py, data_agent.py, common.py
  llm/                   router.py (policy + fallback), providers.py (NVIDIA / Gemini / OpenRouter clients)
  rag/                   corpus_fetch.py, ingest.py (clean + chunk + embed), retriever.py
  tools/web_search.py    Tavily Remote MCP client (shared) + DuckDuckGo / HTTP fallback
  data/                  build_db.py (arXiv metadata -> SQLite), db.py (read-only SQL guard)
  eval/                  evaluation harness (see src/eval/README.md)
static/                  index.html, styles.css, app.js (vanilla JS SSE client)
tests/                   pytest suite (offline + @live)
data/                    corpus/ (PDFs not shipped; metadata.json shipped), papers.db (generated)
chroma_index/            generated Chroma index (baked into the image)
docs/                    architecture.md, evaluation.md, cost_analysis.md, routing_notes.md
Dockerfile, deploy.sh    Cloud Run image and one-command deploy
.env.example             all configuration keys, with placeholders
requirements.txt         pinned runtime deps (requirements-dev.txt adds pytest)
```
