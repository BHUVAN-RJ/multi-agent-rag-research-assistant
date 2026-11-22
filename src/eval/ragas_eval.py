"""RAGAS 0.4.3 evaluation of the research (RAG) path over the fixed question set.

Runs ONLY in `.venv-eval` (ragas conflicts with the app venv):

    .venv-eval/bin/python -m src.eval.ragas_eval --self-check          # offline API wiring check
    .venv-eval/bin/python -m src.eval.ragas_eval --retrieval-only      # offline hit@k / MRR
    .venv-eval/bin/python -m src.eval.ragas_eval                       # generate + judge (needs keys)
    .venv-eval/bin/python -m src.eval.ragas_eval --provider nvidia --limit 5

Two steps, each cached under src/eval/results/ so a rerun resumes instead of starting over:

1. Generation: the real `src.agents.research.run_research` runs for each question (same prompt,
   same router policy as production). Its `retrieve` and `complete` are wrapped for the duration
   of the call so the exact chunks and prompt given to the LLM are recorded. `retrieved_contexts`
   are the per-source blocks cut out of that prompt, i.e. exactly what the LLM read.
   `--provider` sets ROUTER_FORCE_PROVIDER so one provider is pinned to the front of every chain.
2. Judging: ragas.evaluate() with Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
   and LLMContextRecall. The judge is a free LLM (Gemini via langchain-google-genai, or NVIDIA NIM via
   langchain-openai's OpenAI-compatible client). Answer relevancy embeds with Chroma's local ONNX
   all-MiniLM-L6-v2 through a LangChain Embeddings adapter, so no embedding key is needed.

Module import stays ragas-free (ragas is imported inside functions), so the app-venv test suite can
import the pure helpers here without pulling ragas in.
"""

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
METADATA_PATH = PROJECT_ROOT / "data" / "corpus" / "metadata.json"
RESULTS_DIR = EVAL_DIR / "results"

# Column names ragas 0.4.3 writes into EvaluationResult.scores for the four metrics we use.
METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
)
METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy (ResponseRelevancy)",
    "llm_context_precision_with_reference": "Context precision (LLMContextPrecisionWithReference)",
    "context_recall": "Context recall (LLMContextRecall)",
}

# The research agent joins its numbered sources with this separator (src/agents/common.py).
# Must match how build_numbered_sources joins sources (src/agents/common.py).
SOURCES_SEPARATOR = "\n\n\n"
SOURCES_MARKER = "\n\nSources:\n"
PRIOR_FINDINGS_MARKER = "\n\nEarlier agent findings"

VALID_PROVIDERS: tuple[str, ...] = ("nvidia", "gemini", "openrouter")
VALID_JUDGES: tuple[str, ...] = ("gemini", "nvidia")

# List prices in USD per 1M tokens (input, output), checked 2026-10-25. We run on free tiers,
# so actual spend is $0; these say what the same tokens would cost on a paid tier.
# Gemini: https://ai.google.dev/gemini-api/docs/pricing (Standard paid tier, text).
# OpenRouter: GET https://openrouter.ai/api/v1/models, the non-:free variant of the same model.
# NVIDIA NIM build.nvidia.com endpoints publish no per-token price; the OpenRouter price of the
# same Nemotron weights is used as a proxy, and the Lightning model has no listing (None).
PRICES_CHECKED_ON = "2026-10-25"
LIST_PRICES_PER_MILLION: dict[str, tuple[float, float] | None] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 2.40),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 2.40),
    "nvidia/nemotron-3.5-lightning-30b-a3b": None,
    "google/gemma-4-31b-it": (0.10, 0.34),
    "thinkingmachines/inkling-small": (0.45, 1.20),
    "dots-studio/dots-3-note-preview": None,
}


# ---------------------------------------------------------------------------
# Question set and file helpers (pure)
# ---------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    return list(parsed["questions"])


def select_questions(questions: list[dict], limit: int | None, wanted_ids: list[str] | None) -> list[dict]:
    selected = questions
    if wanted_ids:
        wanted_set = set(wanted_ids)
        selected = [entry for entry in selected if entry["id"] in wanted_set]
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_label(provider: str | None) -> str:
    # "policy" means the router's normal hybrid policy picked the provider.
    return provider if provider else "policy"


def generation_cache_path(label: str) -> Path:
    return RESULTS_DIR / f"generations_{label}.json"


def scored_rows_path(label: str, judge: str) -> Path:
    return RESULTS_DIR / f"ragas_rows_{label}__judge-{judge}.json"


def summary_path(label: str) -> Path:
    if label == "policy":
        return RESULTS_DIR / "ragas_scores.json"
    return RESULTS_DIR / f"ragas_scores_{label}.json"


def load_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw_text = path.read_text(encoding="utf-8")
    return json.loads(raw_text)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so an interrupted run never leaves half a cache file.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    temp_path.write_text(encoded, encoding="utf-8")
    temp_path.replace(path)


def utc_now_text() -> str:
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Contexts exactly as the LLM saw them (pure)
# ---------------------------------------------------------------------------


def split_prompt_sources(user_prompt: str, expected_count: int) -> list[str] | None:
    """Cut the research prompt's "Sources:" block back into one string per numbered source.

    Returns None if the prompt does not have exactly `expected_count` source blocks (for example a
    chunk itself contained the separator), so the caller can fall back to raw chunk text.
    """
    if expected_count <= 0 or SOURCES_MARKER not in user_prompt:
        return None
    _, after_marker = user_prompt.split(SOURCES_MARKER, 1)
    # Earlier-agent findings are appended after the sources in multi-step plans; drop them.
    sources_block = after_marker.split(PRIOR_FINDINGS_MARKER, 1)[0]
    segments = sources_block.split(SOURCES_SEPARATOR)
    stripped_segments = [segment.strip() for segment in segments]
    non_empty_segments = [segment for segment in stripped_segments if segment]
    if len(non_empty_segments) != expected_count:
        return None
    return non_empty_segments


def contexts_for_record(user_prompt: str, chunk_texts: list[str]) -> tuple[list[str], str]:
    """Return (contexts, how) where how is "prompt_blocks" or "chunk_text" (fallback)."""
    prompt_blocks = split_prompt_sources(user_prompt, len(chunk_texts))
    if prompt_blocks is not None:
        return prompt_blocks, "prompt_blocks"
    return list(chunk_texts), "chunk_text"


# ---------------------------------------------------------------------------
# RAGAS row building and aggregation (pure)
# ---------------------------------------------------------------------------


def build_sample_row(record: dict) -> dict:
    """Map one generation record to ragas 0.4.3 SingleTurnSample fields.

    user_input <- question, retrieved_contexts <- contexts given to the LLM,
    response <- generated answer, reference <- grounded reference answer.
    Raises ValueError when the record cannot be scored (failed generation).
    """
    response_text = str(record.get("response") or "").strip()
    contexts = [str(context) for context in record.get("retrieved_contexts") or [] if str(context).strip()]
    reference_text = str(record.get("reference") or "").strip()
    question_text = str(record.get("question") or "").strip()
    if not question_text:
        raise ValueError(f"{record.get('id')}: empty question")
    if not response_text:
        raise ValueError(f"{record.get('id')}: empty response (generation failed: {record.get('error')})")
    if not contexts:
        raise ValueError(f"{record.get('id')}: no retrieved contexts")
    if not reference_text:
        raise ValueError(f"{record.get('id')}: empty reference")
    return {
        "user_input": question_text,
        "retrieved_contexts": contexts,
        "response": response_text,
        "reference": reference_text,
    }


def clean_score(value: Any) -> float | None:
    """ragas reports a failed metric as NaN; store it as None so JSON stays valid."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return numeric_value


def aggregate_scores(scored_rows: list[dict], metric_names: tuple[str, ...] = METRIC_NAMES) -> dict[str, dict]:
    """Mean per metric over rows that produced a number, plus how many rows failed for it."""
    aggregates: dict[str, dict] = {}
    for metric_name in metric_names:
        raw_values = [row.get("scores", {}).get(metric_name) for row in scored_rows]
        valid_values = [value for value in raw_values if value is not None]
        mean_value = statistics.fmean(valid_values) if valid_values else None
        aggregates[metric_name] = {
            "mean": mean_value,
            "n": len(valid_values),
            "missing": len(raw_values) - len(valid_values),
        }
    return aggregates


def row_needs_scoring(row: dict | None, rescore: bool) -> bool:
    if rescore or row is None:
        return True
    row_scores = row.get("scores", {})
    # A metric that came back NaN (rate limit, parse failure) is retried on the next run.
    missing_metrics = [name for name in METRIC_NAMES if row_scores.get(name) is None]
    return len(missing_metrics) > 0


def record_needs_generation(record: dict | None, regenerate: bool) -> bool:
    if regenerate or record is None:
        return True
    has_error = bool(record.get("error"))
    has_answer = bool(str(record.get("response") or "").strip())
    return has_error or not has_answer


# ---------------------------------------------------------------------------
# Retrieval-only metrics (pure + offline)
# ---------------------------------------------------------------------------


def source_hit_metrics(expected_ids: list[str], retrieved_ids: list[str]) -> dict:
    """hit = any expected paper in the retrieved list; reciprocal rank of the first expected paper."""
    expected_set = set(expected_ids)
    first_rank: int | None = None
    for position, source_id in enumerate(retrieved_ids, start=1):
        if source_id in expected_set:
            first_rank = position
            break
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    return {"hit": first_rank is not None, "first_rank": first_rank, "reciprocal_rank": reciprocal_rank}


def run_retrieval_only(questions: list[dict], top_k: int) -> dict:
    from src.rag.retriever import retrieve

    per_question: list[dict] = []
    for entry in questions:
        started_at = time.perf_counter()
        chunks = retrieve(entry["question"], top_k)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        retrieved_ids = [chunk.source_id for chunk in chunks]
        hit_stats = source_hit_metrics(entry["source_ids"], retrieved_ids)
        per_question.append(
            {"id": entry["id"], "retrieved_source_ids": retrieved_ids, "latency_ms": round(elapsed_ms, 1), **hit_stats}
        )
    hit_flags = [row["hit"] for row in per_question]
    reciprocal_ranks = [row["reciprocal_rank"] for row in per_question]
    latencies = [row["latency_ms"] for row in per_question]
    return {
        "generated_at": utc_now_text(),
        "k": top_k,
        "n_questions": len(per_question),
        "hit_rate": statistics.fmean(hit_flags) if hit_flags else None,
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Cost estimate (pure)
# ---------------------------------------------------------------------------


def strip_free_suffix(model_id: str) -> str:
    if model_id.endswith(":free"):
        return model_id[: -len(":free")]
    return model_id


def estimate_cost_usd(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """What these tokens would cost at the model's list price. None if tokens or price are unknown."""
    if input_tokens is None or output_tokens is None:
        return None
    price_key = strip_free_suffix(model_id)
    prices = LIST_PRICES_PER_MILLION.get(price_key)
    if prices is None:
        return None
    input_price, output_price = prices
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return input_cost + output_cost


def generation_stats(records: list[dict]) -> dict:
    """Latency, token and cost statistics over successful generations (one research LLM call each)."""
    successful = [record for record in records if not record.get("error")]
    llm_latencies = [record["llm_latency_ms"] for record in successful if record.get("llm_latency_ms") is not None]
    retrieval_latencies = [record["retrieval_ms"] for record in successful if record.get("retrieval_ms") is not None]
    input_counts = [record["input_tokens"] for record in successful if record.get("input_tokens") is not None]
    output_counts = [record["output_tokens"] for record in successful if record.get("output_tokens") is not None]
    costs: list[float] = []
    for record in successful:
        cost = estimate_cost_usd(record.get("model", ""), record.get("input_tokens"), record.get("output_tokens"))
        if cost is not None:
            costs.append(cost)
    provider_counts: dict[str, int] = {}
    for record in successful:
        provider_model = f"{record.get('provider')}/{record.get('model')}"
        provider_counts[provider_model] = provider_counts.get(provider_model, 0) + 1
    return {
        "n_records": len(records),
        "n_ok": len(successful),
        "n_failed": len(records) - len(successful),
        "median_llm_latency_ms": statistics.median(llm_latencies) if llm_latencies else None,
        "median_retrieval_ms": statistics.median(retrieval_latencies) if retrieval_latencies else None,
        "mean_input_tokens": statistics.fmean(input_counts) if input_counts else None,
        "mean_output_tokens": statistics.fmean(output_counts) if output_counts else None,
        "mean_list_price_cost_usd": statistics.fmean(costs) if costs else None,
        "actual_cost_usd": 0.0,
        "prices_checked_on": PRICES_CHECKED_ON,
        "provider_models_used": provider_counts,
        "forced_provider_not_honored": [
            record["id"] for record in successful if record.get("forced_provider") and not record.get("forced_provider_honored")
        ],
    }


# ---------------------------------------------------------------------------
# Markdown rendering (pure)
# ---------------------------------------------------------------------------


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append(f"RAGAS {summary.get('ragas_version')} | generation: {summary.get('label')} | judge: {summary.get('judge', {}).get('provider')}/{summary.get('judge', {}).get('model')} | {summary.get('generated_at')}")
    lines.append("")
    lines.append("| Metric | Mean | Scored | Failed |")
    lines.append("|---|---|---|---|")
    aggregates = summary.get("aggregates", {})
    for metric_name in METRIC_NAMES:
        metric_stats = aggregates.get(metric_name, {})
        label = METRIC_LABELS[metric_name]
        lines.append(f"| {label} | {format_number(metric_stats.get('mean'))} | {metric_stats.get('n', 0)} | {metric_stats.get('missing', 0)} |")
    lines.append("")
    lines.append("| Question | Provider/model | Faith. | Ans. rel. | Ctx prec. | Ctx recall |")
    lines.append("|---|---|---|---|---|---|")
    for row in summary.get("per_question", []):
        row_scores = row.get("scores", {})
        score_cells = [format_number(row_scores.get(metric_name), 2) for metric_name in METRIC_NAMES]
        joined_cells = " | ".join(score_cells)
        lines.append(f"| {row['id']} | {row.get('provider')}/{row.get('model')} | {joined_cells} |")
    stats = summary.get("generation_stats", {})
    lines.append("")
    lines.append(
        f"Generation: {stats.get('n_ok')}/{stats.get('n_records')} ok, median research LLM latency "
        f"{format_number(stats.get('median_llm_latency_ms'), 0)} ms, mean tokens in/out "
        f"{format_number(stats.get('mean_input_tokens'), 0)}/{format_number(stats.get('mean_output_tokens'), 0)}, "
        f"list-price cost/query ${format_number(stats.get('mean_list_price_cost_usd'), 6)} (actual $0, prices {PRICES_CHECKED_ON})"
    )
    return "\n".join(lines)


def render_retrieval_markdown(retrieval_summary: dict) -> str:
    lines = [
        f"| Retrieval (k={retrieval_summary['k']}) | Value |",
        "|---|---|",
        f"| Expected paper in top-k (hit rate) | {format_number(retrieval_summary.get('hit_rate'))} |",
        f"| MRR of first expected paper | {format_number(retrieval_summary.get('mrr'))} |",
        f"| Median retrieval latency (ms, local) | {format_number(retrieval_summary.get('median_latency_ms'), 1)} |",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local embeddings for ResponseRelevancy
# ---------------------------------------------------------------------------


class LocalMiniLMEmbeddings(Embeddings):
    """LangChain Embeddings over Chroma's DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2, 384-d).

    The same model the corpus index uses; runs on CPU with no API key.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.embedding_function(list(texts))
        return [[float(component) for component in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        document_vectors = self.embed_documents([text])
        return document_vectors[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


# ---------------------------------------------------------------------------
# Step 1: generation through the real research agent
# ---------------------------------------------------------------------------


def find_trace(traces: list[dict], kind: str) -> list[dict]:
    matching: list[dict] = []
    for payload in traces:
        if payload.get("type") != "trace":
            continue
        trace_data = payload.get("data", {})
        if trace_data.get("kind") == kind:
            matching.append(trace_data)
    return matching


async def generate_answer(entry: dict, forced_provider: str | None) -> dict:
    """Run run_research once for this question and record what the LLM saw and said."""
    from src.agents import research as research_module
    from src.trace_events import fallback_sink

    recorded_chunks: list = []
    recorded_messages: list[list[dict]] = []
    recorded_llm_results: list = []
    original_retrieve = research_module.retrieve
    original_complete = research_module.complete

    def recording_retrieve(query: str, k: int | None = None):
        chunks = original_retrieve(query, k)
        recorded_chunks.extend(chunks)
        return chunks

    async def recording_complete(role, messages, profile, stream_tokens=False, node="", **kwargs):
        recorded_messages.append(messages)
        llm_result = await original_complete(role, messages, profile, stream_tokens, node, **kwargs)
        recorded_llm_results.append(llm_result)
        return llm_result

    captured_traces: list[dict] = []
    sink_token = fallback_sink.set(captured_traces.append)
    # Wrap the names research.py looks up at call time; restored in `finally`.
    research_module.retrieve = recording_retrieve
    research_module.complete = recording_complete
    started_at = time.perf_counter()
    try:
        agent_state = {"query": entry["question"], "needs_web": False, "agent_outputs": {}, "citations": []}
        output = await research_module.run_research(agent_state, is_final=False)
    finally:
        research_module.retrieve = original_retrieve
        research_module.complete = original_complete
        fallback_sink.reset(sink_token)
    total_ms = (time.perf_counter() - started_at) * 1000

    chunk_texts = [chunk.text for chunk in recorded_chunks]
    user_prompt = ""
    if recorded_messages:
        last_messages = recorded_messages[-1]
        user_messages = [message["content"] for message in last_messages if message.get("role") == "user"]
        user_prompt = user_messages[-1] if user_messages else ""
    contexts, contexts_source = contexts_for_record(user_prompt, chunk_texts)

    llm_result = recorded_llm_results[-1] if recorded_llm_results else None
    retrieval_traces = find_trace(captured_traces, "rag_retrieval")
    retrieval_latencies = [trace["detail"].get("latency_ms") or 0.0 for trace in retrieval_traces]
    fallback_traces = find_trace(captured_traces, "llm_fallback")
    used_provider = output.provider or ""
    return {
        "id": entry["id"],
        "question": entry["question"],
        "reference": entry["reference"],
        "source_ids": entry["source_ids"],
        "response": output.text,
        "error": output.error,
        "retrieved_contexts": contexts,
        "contexts_source": contexts_source,
        "retrieved_source_ids": [chunk.source_id for chunk in recorded_chunks],
        "chunk_ids": [chunk.chunk_id for chunk in recorded_chunks],
        "provider": used_provider,
        "model": output.model,
        "forced_provider": forced_provider,
        "forced_provider_honored": forced_provider is None or used_provider == forced_provider,
        "route_reason": llm_result.decision.reason if llm_result else None,
        "input_tokens": llm_result.input_tokens if llm_result else None,
        "output_tokens": llm_result.output_tokens if llm_result else None,
        "llm_latency_ms": llm_result.latency_ms if llm_result else None,
        "attempts": llm_result.attempts if llm_result else [],
        "fallbacks": len(fallback_traces),
        "retrieval_ms": sum(retrieval_latencies) if retrieval_traces else None,
        "total_ms": total_ms,
        "generated_at": utc_now_text(),
    }


async def generate_all(questions: list[dict], label: str, forced_provider: str | None, regenerate: bool, pause_seconds: float) -> list[dict]:
    cache_path = generation_cache_path(label)
    cache: dict[str, dict] = load_json_or_default(cache_path, {})
    pending = [entry for entry in questions if record_needs_generation(cache.get(entry["id"]), regenerate)]
    print(f"[generate] {label}: {len(questions) - len(pending)} cached, {len(pending)} to generate -> {cache_path}")
    for index, entry in enumerate(pending):
        record = await generate_answer(entry, forced_provider)
        cache[entry["id"]] = record
        save_json(cache_path, cache)
        status_text = "ok" if not record["error"] else f"FAILED: {str(record['error'])[:160]}"
        print(f"[generate] {entry['id']}: {record['provider']}/{record['model']} {status_text}")
        # Free tiers allow ~10 RPM on Gemini; spacing calls avoids burning the fallback chain on 429s.
        is_last = index == len(pending) - 1
        if pause_seconds > 0 and not is_last:
            await asyncio.sleep(pause_seconds)
    ordered_records = [cache[entry["id"]] for entry in questions if entry["id"] in cache]
    return ordered_records


# ---------------------------------------------------------------------------
# Step 2: judging with RAGAS 0.4.3
# ---------------------------------------------------------------------------


def disable_ragas_telemetry() -> None:
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


def resolve_judge_model(judge: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    from src.llm.providers import resolve_model

    # settings.gemini_model_strong / settings.nvidia_model_strong, with the router's verified defaults.
    return resolve_model(judge, "strong")


def judge_api_key(judge: str) -> str:
    from src.config import settings

    if judge == "gemini":
        return settings.google_api_key
    return settings.nvidia_api_key


def build_judge_chat_model(judge: str, model_name: str, api_key: str, requests_per_minute: float, timeout_seconds: float):
    """LangChain chat model for the judge, rate limited client-side to stay under free-tier RPM."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    requests_per_second = requests_per_minute / 60.0
    rate_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second, check_every_n_seconds=0.2, max_bucket_size=1)
    if judge == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=4096,
            timeout=timeout_seconds,
            max_retries=2,
            rate_limiter=rate_limiter,
        )

    from langchain_openai import ChatOpenAI

    from src.config import settings

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=settings.nvidia_base_url,
        temperature=0.0,
        max_tokens=4096,
        timeout=timeout_seconds,
        max_retries=2,
        rate_limiter=rate_limiter,
        # Nemotron reasons by default on NIM; the judge needs plain JSON, fast.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def wrap_judge_for_ragas(chat_model, run_config):
    # ragas.llms.LangchainLLMWrapper is a deprecation shim around this class. We build the class
    # directly so we can pass bypass_n=True: ResponseRelevancy asks for n=strictness completions,
    # which Gemini/NIM do not reliably honor, so we make n separate calls instead.
    from ragas.llms.base import LangchainLLMWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(chat_model, run_config=run_config, bypass_n=True)


def wrap_embeddings_for_ragas(langchain_embeddings, run_config):
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainEmbeddingsWrapper(langchain_embeddings, run_config=run_config)


def build_metrics(ragas_llm, ragas_embeddings, strictness: int) -> list:
    """The four metrics, as the 0.4.3 single-turn Metric classes ragas.evaluate() accepts.

    ragas.metrics.collections holds the newer async-only versions, but evaluate() still requires
    ragas.metrics.base.Metric instances, so we use the classes it ships for that (the same objects
    `from ragas.metrics import Faithfulness` returns, minus the deprecation warning).
    """
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    return [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=strictness),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]


def build_run_config(max_workers: int, timeout_seconds: int, max_retries: int, max_wait_seconds: int):
    from ragas import RunConfig

    return RunConfig(timeout=timeout_seconds, max_retries=max_retries, max_wait=max_wait_seconds, max_workers=max_workers)


def build_evaluation_dataset(rows: list[dict]):
    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(rows)


def score_single_row(row: dict, metrics: list, run_config) -> dict[str, float | None]:
    from ragas import evaluate

    dataset = build_evaluation_dataset([row])
    evaluation_result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=False,
    )
    raw_scores = evaluation_result.scores[0]
    return {metric_name: clean_score(raw_scores.get(metric_name)) for metric_name in METRIC_NAMES}


def score_records(records: list[dict], label: str, judge: str, metrics: list, run_config, rescore: bool) -> list[dict]:
    """Judge one question at a time and save after each, so a 429 storm loses at most one row."""
    rows_path = scored_rows_path(label, judge)
    scored_cache: dict[str, dict] = load_json_or_default(rows_path, {})
    for record in records:
        cached_row = scored_cache.get(record["id"])
        # A regenerated answer invalidates its old score.
        answer_changed = cached_row is not None and cached_row.get("generated_at") != record.get("generated_at")
        if not answer_changed and not row_needs_scoring(cached_row, rescore):
            continue
        try:
            sample_row = build_sample_row(record)
        except ValueError as unscorable:
            print(f"[judge] skip {record['id']}: {unscorable}")
            continue
        started_at = time.perf_counter()
        row_scores = score_single_row(sample_row, metrics, run_config)
        elapsed_seconds = time.perf_counter() - started_at
        scored_cache[record["id"]] = {
            "id": record["id"],
            "provider": record.get("provider"),
            "model": record.get("model"),
            "generated_at": record.get("generated_at"),
            "scored_at": utc_now_text(),
            "judge_seconds": round(elapsed_seconds, 1),
            "scores": row_scores,
        }
        save_json(rows_path, scored_cache)
        score_texts = [f"{name}={format_number(value, 2)}" for name, value in row_scores.items()]
        print(f"[judge] {record['id']}: {', '.join(score_texts)} ({elapsed_seconds:.0f}s)")
    ordered_rows = [scored_cache[record["id"]] for record in records if record["id"] in scored_cache]
    return ordered_rows


def ragas_version() -> str:
    try:
        from importlib.metadata import version

        return version("ragas")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Offline self-check of the ragas wiring (no network)
# ---------------------------------------------------------------------------


def run_self_check(judge: str, strictness: int) -> int:
    """Build judge wrapper, embeddings, metrics, dataset and RunConfig exactly as a live run does,
    then validate the dataset against the metrics' required columns. Makes no network call."""
    disable_ragas_telemetry()
    from ragas.dataset_schema import SingleTurnSample
    from ragas.validation import validate_required_columns, validate_supported_metrics

    questions = load_questions()
    first_entry = questions[0]
    sample_record = {
        "id": first_entry["id"],
        "question": first_entry["question"],
        "reference": first_entry["reference"],
        "response": "ReAct interleaves reasoning traces with actions [1].",
        "retrieved_contexts": ["[1] (corpus) ReAct | arXiv:2210.03629 | p.1\nReAct interleaves reasoning and acting."],
    }
    sample_row = build_sample_row(sample_record)
    sample = SingleTurnSample(**sample_row)
    run_config = build_run_config(max_workers=1, timeout_seconds=180, max_retries=6, max_wait_seconds=60)
    model_name = resolve_judge_model(judge, None)
    chat_model = build_judge_chat_model(judge, model_name, "offline-self-check-key", 8.0, 120.0)
    ragas_llm = wrap_judge_for_ragas(chat_model, run_config)
    local_embeddings = LocalMiniLMEmbeddings()
    ragas_embeddings = wrap_embeddings_for_ragas(local_embeddings, run_config)
    metrics = build_metrics(ragas_llm, ragas_embeddings, strictness)
    dataset = build_evaluation_dataset([sample_row])
    validate_required_columns(dataset, metrics)
    validate_supported_metrics(dataset, metrics)
    for metric in metrics:
        metric.init(run_config)
    query_vector = ragas_embeddings.embed_query("retrieval-augmented generation")
    metric_names = [metric.name for metric in metrics]
    if tuple(metric_names) != METRIC_NAMES:
        print(f"SELF-CHECK FAILED: metric names {metric_names} != {list(METRIC_NAMES)}")
        return 1
    print(f"ragas {ragas_version()} OK: sample fields {sorted(sample.model_dump(exclude_none=True))}")
    print(f"metrics {metric_names}; judge {judge}/{model_name}; embedding dim {len(query_vector)}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 0.4.3 evaluation of the research RAG path (.venv-eval only).")
    parser.add_argument("--provider", choices=VALID_PROVIDERS, help="pin generation to one provider via ROUTER_FORCE_PROVIDER")
    parser.add_argument("--judge", choices=VALID_JUDGES, default="gemini", help="free judge LLM (default gemini strong)")
    parser.add_argument("--judge-model", help="override the judge model id")
    parser.add_argument("--judge-rpm", type=float, default=8.0, help="client-side judge rate limit, requests/minute")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--max-workers", type=int, default=1, help="ragas RunConfig.max_workers (keep 1-2 on free tiers)")
    parser.add_argument("--timeout", type=int, default=180, help="ragas per-operation timeout, seconds")
    parser.add_argument("--max-retries", type=int, default=6, help="ragas retries per operation")
    parser.add_argument("--max-wait", type=int, default=60, help="ragas max backoff between retries, seconds")
    parser.add_argument("--strictness", type=int, default=3, help="ResponseRelevancy questions generated per answer")
    parser.add_argument("--pause", type=float, default=6.0, help="seconds between generation calls")
    parser.add_argument("--regenerate", action="store_true", help="ignore cached generations")
    parser.add_argument("--rescore", action="store_true", help="ignore cached judge scores")
    parser.add_argument("--generate-only", action="store_true", help="stop after generation (no judge key needed)")
    parser.add_argument("--retrieval-only", action="store_true", help="offline: hit@k and MRR of expected papers")
    parser.add_argument("--self-check", action="store_true", help="offline: build the ragas pipeline without calling any LLM")
    parser.add_argument("--out", type=Path, help="summary JSON path (default results/ragas_scores[_<provider>].json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    questions = load_questions()
    selected = select_questions(questions, arguments.limit, arguments.ids)

    if arguments.self_check:
        return run_self_check(arguments.judge, arguments.strictness)

    if arguments.retrieval_only:
        from src.config import settings

        retrieval_summary = run_retrieval_only(selected, settings.retrieval_k)
        save_json(RESULTS_DIR / "retrieval_only.json", retrieval_summary)
        print(render_retrieval_markdown(retrieval_summary))
        return 0

    label = run_label(arguments.provider)
    if arguments.provider:
        # Read by src.llm.router.decide_route at every call.
        os.environ["ROUTER_FORCE_PROVIDER"] = arguments.provider
    from src.llm.providers import has_api_key

    available_providers = [provider for provider in VALID_PROVIDERS if has_api_key(provider)]
    if not available_providers:
        print("No provider API key is set (NVIDIA_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY in .env). Nothing to generate.")
        return 2
    if arguments.provider and arguments.provider not in available_providers:
        print(f"WARNING: --provider {arguments.provider} has no key; the router will fall back to {available_providers}.")

    pending_generation = generate_all(selected, label, arguments.provider, arguments.regenerate, arguments.pause)
    records = asyncio.run(pending_generation)
    if arguments.generate_only:
        print(json.dumps(generation_stats(records), indent=2))
        return 0

    api_key = judge_api_key(arguments.judge)
    if not has_api_key(arguments.judge):
        print(f"No key for judge '{arguments.judge}'. Generations are cached; rerun when the key exists.")
        return 2

    disable_ragas_telemetry()
    run_config = build_run_config(arguments.max_workers, arguments.timeout, arguments.max_retries, arguments.max_wait)
    judge_model = resolve_judge_model(arguments.judge, arguments.judge_model)
    chat_model = build_judge_chat_model(arguments.judge, judge_model, api_key, arguments.judge_rpm, float(arguments.timeout))
    ragas_llm = wrap_judge_for_ragas(chat_model, run_config)
    ragas_embeddings = wrap_embeddings_for_ragas(LocalMiniLMEmbeddings(), run_config)
    metrics = build_metrics(ragas_llm, ragas_embeddings, arguments.strictness)

    scored_rows = score_records(records, label, arguments.judge, metrics, run_config, arguments.rescore)
    summary = {
        "generated_at": utc_now_text(),
        "ragas_version": ragas_version(),
        "label": label,
        "forced_provider": arguments.provider,
        "judge": {"provider": arguments.judge, "model": judge_model, "strictness": arguments.strictness},
        "embeddings": "chromadb DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2, local)",
        "n_questions": len(selected),
        "aggregates": aggregate_scores(scored_rows),
        "generation_stats": generation_stats(records),
        "per_question": scored_rows,
    }
    output_path = arguments.out or summary_path(label)
    save_json(output_path, summary)
    print(f"\nwrote {output_path}\n")
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
