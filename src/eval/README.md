# Evaluation: how to run

Two venvs. RAGAS runs only in `.venv-eval`, and the latency bench runs in the app venv `.venv`.

```bash
# one-time: eval venv
uv venv .venv-eval --python 3.13
VIRTUAL_ENV=.venv-eval uv pip install -r requirements-eval.txt
```

## Offline (no keys)

```bash
.venv/bin/python -m pytest -q                               # full suite, includes tests/test_eval.py
.venv-eval/bin/python -m src.eval.ragas_eval --self-check   # builds the ragas pipeline, no LLM call
.venv-eval/bin/python -m src.eval.ragas_eval --retrieval-only   # hit@k / MRR -> results/retrieval_only.json
```

## RAGAS (needs keys in `.env`)

```bash
# 15 questions, router policy picks the provider, Gemini strong judges
.venv-eval/bin/python -m src.eval.ragas_eval
# NVIDIA NIM as judge instead
.venv-eval/bin/python -m src.eval.ragas_eval --judge nvidia
# provider comparison on a subset (ROUTER_FORCE_PROVIDER pins the primary)
.venv-eval/bin/python -m src.eval.ragas_eval --provider nvidia --limit 5
.venv-eval/bin/python -m src.eval.ragas_eval --provider gemini --limit 5
.venv-eval/bin/python -m src.eval.ragas_eval --provider openrouter --limit 5
```

- Generations are cached in `results/generations_<label>.json`, and judge scores per question in
  `results/ragas_rows_<label>__judge-<judge>.json`. A rerun only redoes failed or missing items.
  Use `--regenerate` or `--rescore` to force a redo.
- The summary goes to `results/ragas_scores.json` (`ragas_scores_<provider>.json` with
  `--provider`), and a markdown table is printed for `docs/evaluation.md`.
- Free tiers: the defaults are `--max-workers 1 --judge-rpm 8 --pause 6`. One question costs about 10
  judge calls (strictness 3), so a full run takes about 20 min on Gemini's free tier. If you hit
  the daily cap, rerun later and it resumes.
- `--generate-only` does the generation step without a judge key.

## Latency (server must be running)

```bash
.venv/bin/python main.py                                                      # local server on :8080
.venv/bin/python -m src.eval.latency_bench --cold --url https://<svc>.run.app  # FIRST request after deploy/idle
.venv/bin/python -m src.eval.latency_bench --url https://<svc>.run.app --repeat 2
.venv/bin/python -m src.eval.latency_bench --report                            # tables from saved JSON
```

For a real cold start, deploy a new revision, or wait until Cloud Run has scaled to zero
(min-instances 0, roughly 15 min idle). Then run `--cold` before anything else hits the service.
The outputs are `results/latency_cold.json` and `results/latency_warm.json`.
