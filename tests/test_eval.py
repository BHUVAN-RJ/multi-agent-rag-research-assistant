"""Offline tests for the evaluation layer (src/eval). No keys, no network.

Runs in the app venv and must never import ragas. Checks that need ragas run in a subprocess
under .venv-eval and skip when that venv is missing.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from src.agents.common import build_numbered_sources
from src.agents.supervisor import keyword_plan
from src.eval import latency_bench, ragas_eval
from src.rag.retriever import RetrievedChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PYTHON = PROJECT_ROOT / ".venv-eval" / "bin" / "python"
METADATA_PATH = PROJECT_ROOT / "data" / "corpus" / "metadata.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_index"

# The papers the assignment brief requires the question set to cover.
REQUIRED_SOURCE_IDS = {
    "2210.03629",  # ReAct
    "2303.11366",  # Reflexion
    "2302.04761",  # Toolformer
    "2308.08155",  # AutoGen
    "2308.00352",  # MetaGPT
    "2005.11401",  # RAG (Lewis)
    "2310.11511",  # Self-RAG
    "2401.18059",  # RAPTOR
    "2212.10496",  # HyDE
    "2401.15884",  # CRAG
    "2404.16130",  # GraphRAG
    "2307.03172",  # Lost in the Middle
    "2004.04906",  # DPR
    "2305.10601",  # Tree of Thoughts
    "2309.15217",  # RAGAS
}


def make_chunk(index: int, source_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_id}v1::chunk-{index:04d}",
        text=text,
        source_id=source_id,
        title=f"Paper {source_id}",
        section="Method",
        page=index + 1,
        url=f"https://arxiv.org/abs/{source_id}",
        score=0.9 - index * 0.1,
    )


# ---------------------------------------------------------------------------
# Question set
# ---------------------------------------------------------------------------


def test_question_set_has_15_grounded_entries_with_known_sources():
    questions = ragas_eval.load_questions()
    metadata_entries = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    known_ids = {entry["id"] for entry in metadata_entries}

    assert len(questions) == 15
    question_ids = [entry["id"] for entry in questions]
    assert len(set(question_ids)) == 15, "question ids must be unique"
    for entry in questions:
        assert entry["question"].strip().endswith("?"), entry["id"]
        reference_text = entry["reference"].strip()
        assert len(reference_text) > 80, f"{entry['id']} reference too short"
        # 1-3 sentences: count sentence-ending periods followed by a space or end of text.
        sentence_count = reference_text.count(". ") + 1
        assert sentence_count <= 3, f"{entry['id']} reference has {sentence_count} sentences"
        assert entry["source_ids"], entry["id"]
        unknown_ids = set(entry["source_ids"]) - known_ids
        assert not unknown_ids, f"{entry['id']} cites ids not in metadata.json: {unknown_ids}"


def test_question_set_covers_every_required_paper():
    questions = ragas_eval.load_questions()
    covered_ids = {source_id for entry in questions for source_id in entry["source_ids"]}
    missing_ids = REQUIRED_SOURCE_IDS - covered_ids
    assert not missing_ids, f"required papers not covered: {missing_ids}"


def test_questions_route_to_research_even_on_keyword_fallback():
    # With no LLM the supervisor uses keyword_plan; a question that trips the data/summary/web
    # keywords would never reach the RAG path the eval measures.
    for entry in ragas_eval.load_questions():
        fallback_plan = keyword_plan(entry["question"], "test")
        assert fallback_plan.plan == ["research"], entry["id"]
        assert fallback_plan.needs_web is False, entry["id"]


def test_select_questions_by_limit_and_ids():
    questions = ragas_eval.load_questions()
    assert len(ragas_eval.select_questions(questions, 5, None)) == 5
    picked = ragas_eval.select_questions(questions, None, ["q15_ragas", "q01_react"])
    picked_ids = [entry["id"] for entry in picked]
    assert picked_ids == ["q01_react", "q15_ragas"]


# ---------------------------------------------------------------------------
# Contexts and RAGAS rows
# ---------------------------------------------------------------------------


def test_split_prompt_sources_recovers_exact_blocks_from_research_prompt():
    chunks = [make_chunk(0, "2210.03629", "ReAct interleaves thoughts and actions."), make_chunk(1, "2303.11366", "Reflexion keeps an episodic memory.")]
    _, context_block = build_numbered_sources(chunks, [], 1)
    # Same layout research.run_research builds, including appended prior findings.
    user_prompt = f"Question: q?\n\nSources:\n{context_block}\n\nEarlier agent findings (may cite their own sources):\nfoo"
    blocks = ragas_eval.split_prompt_sources(user_prompt, 2)
    assert blocks is not None and len(blocks) == 2
    assert blocks[0].startswith("[1] (corpus) Paper 2210.03629 | arXiv:2210.03629")
    assert blocks[0].endswith("ReAct interleaves thoughts and actions.")
    assert blocks[1].endswith("Reflexion keeps an episodic memory.")
    assert "Earlier agent findings" not in blocks[1]


def test_contexts_fall_back_to_chunk_text_when_block_count_mismatches():
    chunk_texts = ["first chunk", "second chunk with\n\n\nseparator inside"]
    chunks = [make_chunk(index, "2210.03629", text) for index, text in enumerate(chunk_texts)]
    _, context_block = build_numbered_sources(chunks, [], 1)
    user_prompt = f"Question: q?\n\nSources:\n{context_block}"
    contexts, how = ragas_eval.contexts_for_record(user_prompt, chunk_texts)
    assert how == "chunk_text"
    assert contexts == chunk_texts


def test_build_sample_row_maps_fields_and_rejects_failed_generations():
    record = {
        "id": "q", "question": "What is X?", "reference": "X is Y.", "response": "X is Y [1].",
        "retrieved_contexts": ["ctx one", "  ", "ctx two"], "error": None,
    }
    row = ragas_eval.build_sample_row(record)
    assert row == {"user_input": "What is X?", "retrieved_contexts": ["ctx one", "ctx two"], "response": "X is Y [1].", "reference": "X is Y."}

    failed_record = {**record, "response": "", "error": "all 3 provider attempts failed"}
    with pytest.raises(ValueError, match="generation failed"):
        ragas_eval.build_sample_row(failed_record)
    no_context_record = {**record, "retrieved_contexts": []}
    with pytest.raises(ValueError, match="no retrieved contexts"):
        ragas_eval.build_sample_row(no_context_record)


@pytest.mark.skipif(not EVAL_PYTHON.exists(), reason=".venv-eval not present")
def test_sample_row_validates_against_ragas_0_4_3_schema_in_eval_venv():
    # Build a row with the pure builder here, then validate it with ragas in the eval venv.
    record = {"id": "q", "question": "What is X?", "reference": "X is Y.", "response": "X is Y.", "retrieved_contexts": ["ctx"]}
    row_json = json.dumps(ragas_eval.build_sample_row(record))
    check_script = (
        "import json, sys\n"
        "from ragas import SingleTurnSample, EvaluationDataset, __version__\n"
        "row = json.loads(sys.argv[1])\n"
        "unknown = set(row) - set(SingleTurnSample.model_fields)\n"
        "assert not unknown, unknown\n"
        "sample = SingleTurnSample(**row)\n"
        "dataset = EvaluationDataset.from_list([row])\n"
        "assert dataset.get_sample_type() is SingleTurnSample\n"
        "assert sample.retrieved_contexts == row['retrieved_contexts']\n"
        "print(__version__)\n"
    )
    completed = subprocess.run([str(EVAL_PYTHON), "-c", check_script, row_json], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=180)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "0.4.3"


@pytest.mark.skipif(not EVAL_PYTHON.exists(), reason=".venv-eval not present")
def test_ragas_pipeline_self_check_in_eval_venv():
    # Builds judge wrapper, local embeddings, the 4 metrics and RunConfig, validates required columns.
    completed = subprocess.run(
        [str(EVAL_PYTHON), "-m", "src.eval.ragas_eval", "--self-check"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "faithfulness" in completed.stdout and "embedding dim 384" in completed.stdout


def test_eval_modules_do_not_import_ragas_in_app_venv():
    check_script = "import sys; import src.eval.ragas_eval, src.eval.latency_bench; print('ragas' in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", check_script], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# Score aggregation, caching decisions, cost
# ---------------------------------------------------------------------------


def test_clean_score_turns_nan_into_none():
    assert ragas_eval.clean_score(float("nan")) is None
    assert ragas_eval.clean_score(None) is None
    assert ragas_eval.clean_score("0.5") == 0.5
    assert ragas_eval.clean_score(1) == 1.0


def test_aggregate_scores_means_ignore_failed_metrics():
    scored_rows = [
        {"scores": {"faithfulness": 1.0, "answer_relevancy": 0.8, "llm_context_precision_with_reference": 0.5, "context_recall": None}},
        {"scores": {"faithfulness": 0.5, "answer_relevancy": None, "llm_context_precision_with_reference": 1.0, "context_recall": 1.0}},
        {"scores": {"faithfulness": 0.0, "answer_relevancy": 0.6, "llm_context_precision_with_reference": 0.0, "context_recall": 0.0}},
    ]
    aggregates = ragas_eval.aggregate_scores(scored_rows)
    assert math.isclose(aggregates["faithfulness"]["mean"], 0.5)
    assert math.isclose(aggregates["answer_relevancy"]["mean"], 0.7)
    assert aggregates["answer_relevancy"]["missing"] == 1
    assert math.isclose(aggregates["context_recall"]["mean"], 0.5)
    assert aggregates["context_recall"]["n"] == 2


def test_cache_decisions_regenerate_failures_and_rescore_missing_metrics():
    good_record = {"response": "answer", "error": None}
    assert ragas_eval.record_needs_generation(None, False)
    assert not ragas_eval.record_needs_generation(good_record, False)
    assert ragas_eval.record_needs_generation(good_record, True)
    assert ragas_eval.record_needs_generation({"response": "", "error": "429"}, False)

    complete_row = {"scores": {name: 0.5 for name in ragas_eval.METRIC_NAMES}}
    partial_row = {"scores": {**complete_row["scores"], "context_recall": None}}
    assert not ragas_eval.row_needs_scoring(complete_row, False)
    assert ragas_eval.row_needs_scoring(partial_row, False)
    assert ragas_eval.row_needs_scoring(complete_row, True)


def test_source_hit_metrics_rank_and_reciprocal():
    first_hit = ragas_eval.source_hit_metrics(["B"], ["A", "B", "B"])
    assert first_hit == {"hit": True, "first_rank": 2, "reciprocal_rank": 0.5}
    miss = ragas_eval.source_hit_metrics(["Z"], ["A", "B"])
    assert miss["hit"] is False and miss["reciprocal_rank"] == 0.0


def test_estimate_cost_uses_list_price_and_strips_free_suffix():
    gemini_cost = ragas_eval.estimate_cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000)
    assert math.isclose(gemini_cost, 0.75 + 3.75)
    free_variant_cost = ragas_eval.estimate_cost_usd("google/gemma-4-31b-it:free", 2_000_000, 0)
    assert math.isclose(free_variant_cost, 0.20)
    assert ragas_eval.estimate_cost_usd("gemini-3.8-flash", None, 10) is None
    assert ragas_eval.estimate_cost_usd("nvidia/nemotron-3.5-lightning-30b-a3b", 10, 10) is None


def test_generation_stats_flags_ignored_provider_override():
    records = [
        {"id": "a", "provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "forced_provider": "nvidia", "forced_provider_honored": True,
         "llm_latency_ms": 100.0, "retrieval_ms": 10.0, "input_tokens": 1000, "output_tokens": 100, "error": None},
        {"id": "b", "provider": "gemini", "model": "gemini-3.5-flash-lite", "forced_provider": "nvidia", "forced_provider_honored": False,
         "llm_latency_ms": 300.0, "retrieval_ms": 30.0, "input_tokens": 2000, "output_tokens": 200, "error": None},
        {"id": "c", "provider": "", "model": "", "error": "all failed"},
    ]
    stats = ragas_eval.generation_stats(records)
    assert stats["n_ok"] == 2 and stats["n_failed"] == 1
    assert stats["median_llm_latency_ms"] == 200.0
    assert stats["forced_provider_not_honored"] == ["b"]
    assert stats["actual_cost_usd"] == 0.0
    assert stats["mean_list_price_cost_usd"] is not None


def test_render_markdown_lists_all_four_metrics():
    summary = {
        "ragas_version": "0.4.3", "label": "policy", "judge": {"provider": "gemini", "model": "m"},
        "aggregates": ragas_eval.aggregate_scores([{"scores": {name: 1.0 for name in ragas_eval.METRIC_NAMES}}]),
        "per_question": [{"id": "q01", "provider": "gemini", "model": "m", "scores": {name: 1.0 for name in ragas_eval.METRIC_NAMES}}],
        "generation_stats": {},
    }
    markdown = ragas_eval.render_markdown(summary)
    for label in ragas_eval.METRIC_LABELS.values():
        assert f"| {label} | 1.000 |" in markdown
    assert "| q01 | gemini/m | 1.00 | 1.00 | 1.00 | 1.00 |" in markdown


# ---------------------------------------------------------------------------
# Generation step through the real research agent, with a fake provider
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CHROMA_DIR.is_dir(), reason="chroma_index not built")
async def test_generate_answer_records_exact_contexts_and_tokens(monkeypatch):
    from src.llm import providers

    prompts_seen: list[str] = []

    async def fake_call_provider(provider, model, messages, stream=False, on_token=None, max_tokens=1024, temperature=0.2, transport=None):
        prompts_seen.append(messages[-1]["content"])
        return providers.ProviderResponse("ReAct interleaves reasoning and acting [1].", 1234, 56, 42.0, None)

    def fake_has_api_key(provider):
        return provider == "nvidia"

    monkeypatch.setattr(providers, "call_provider", fake_call_provider)
    monkeypatch.setattr(providers, "has_api_key", fake_has_api_key)
    entry = ragas_eval.load_questions()[0]
    record = await ragas_eval.generate_answer(entry, "nvidia")

    assert record["error"] is None
    assert record["response"].startswith("ReAct interleaves")
    assert record["provider"] == "nvidia" and record["forced_provider_honored"] is True
    assert record["input_tokens"] == 1234 and record["output_tokens"] == 56
    assert record["contexts_source"] == "prompt_blocks"
    assert len(record["retrieved_contexts"]) == len(record["chunk_ids"]) > 0
    # Every recorded context must appear verbatim in the prompt the LLM was sent.
    for context in record["retrieved_contexts"]:
        assert context in prompts_seen[-1]
    assert "2210.03629" in record["retrieved_source_ids"]
    assert record["retrieval_ms"] is not None and record["retrieval_ms"] > 0

    # The wrappers must be removed afterwards so the app's research agent is untouched.
    from src.agents import research as research_module
    from src.rag.retriever import retrieve as real_retrieve
    assert research_module.retrieve is real_retrieve


# ---------------------------------------------------------------------------
# SSE parsing and per-query records (latency_bench)
# ---------------------------------------------------------------------------

SAMPLE_STREAM = (
    ": ping\n\n"
    'event: trace\ndata: {"kind": "rag_retrieval", "node": "retriever", "summary": "", "detail": {"latency_ms": 40.5}}\n\n'
    'event: trace\ndata: {"kind": "llm_route", "node": "research", "summary": "", "detail": {"role": "research", "provider": "gemini", "model": "gemini-3.5-flash-lite", "dimension": "hybrid"}}\n\n'
    ": ping\n\n"
    'event: token\ndata: {"text": "Hello", "node": "research"}\n\n'
    'event: token\ndata: {"text": " world", "node": "research"}\n\n'
    'event: trace\ndata: {"kind": "agent_result", "node": "research", "summary": "", "detail": {"ok": true, "provider": "gemini", "model": "gemini-3.5-flash-lite", "llm_latency_ms": 900.0, "ttft_ms": 350.0, "input_tokens": 1500, "output_tokens": 80}}\n\n'
    'event: done\ndata: {"answer": "Hello world", "final_agent": "research", "plan": ["research"], "latency_ms": 1200.0, "ttft_ms": 400.0}\n\n'
)


def parse_in_pieces(stream_text: str, piece_size: int) -> list[latency_bench.SSEEvent]:
    parser = latency_bench.SSEParser()
    events: list[latency_bench.SSEEvent] = []
    for start in range(0, len(stream_text), piece_size):
        piece = stream_text[start:start + piece_size]
        events.extend(parser.feed(piece))
    events.extend(parser.finish())
    return events


@pytest.mark.parametrize("piece_size", [1, 7, 64, 10_000])
def test_sse_parser_handles_split_frames_and_pings(piece_size):
    events = parse_in_pieces(SAMPLE_STREAM, piece_size)
    event_names = [sse_event.event for sse_event in events]
    assert event_names == ["trace", "trace", "token", "token", "trace", "done"]
    assert events[2].data == {"text": "Hello", "node": "research"}
    assert events[-1].data["latency_ms"] == 1200.0


def test_sse_parser_handles_crlf_split_between_cr_and_lf_and_multiline_data():
    crlf_stream = "event: token\r\ndata: {\"text\": \"a\"}\r\n\r\n: ping\r\n\r\nevent: note\r\ndata: line one\r\ndata: line two\r\n\r\n"
    # Cut between "\r" and "\n" inside a frame: turning that lone "\r" into "\n" too early would
    # create a fake blank line and split "event: token" from its data.
    split_point = crlf_stream.index("event: token\r") + len("event: token\r")
    parser = latency_bench.SSEParser()
    events = parser.feed(crlf_stream[:split_point])
    events += parser.feed(crlf_stream[split_point:])
    assert [sse_event.event for sse_event in events] == ["token", "note"]
    assert events[0].data == {"text": "a"}
    assert events[1].data == "line one\nline two"


def test_sse_parser_finish_flushes_unterminated_frame():
    parser = latency_bench.SSEParser()
    assert parser.feed('event: done\ndata: {"latency_ms": 5}') == []
    flushed = parser.finish()
    assert len(flushed) == 1 and flushed[0].data == {"latency_ms": 5}


def test_build_query_record_extracts_latency_breakdown():
    events = parse_in_pieces(SAMPLE_STREAM, 13)
    record = latency_bench.build_query_record(events, "rag", "q", client_ttft_ms=410.0, client_total_ms=1250.0)
    assert record["ok"] is True
    assert record["retrieval_ms"] == 40.5
    assert record["llm_ms"] == 900.0 and record["llm_ttft_ms"] == 350.0
    assert record["provider"] == "gemini" and record["model"] == "gemini-3.5-flash-lite"
    assert record["server_total_ms"] == 1200.0 and record["server_ttft_ms"] == 400.0
    assert math.isclose(record["other_ms"], 1200.0 - 40.5 - 900.0)
    assert record["input_tokens"] == 1500 and record["output_tokens"] == 80
    assert record["routes"][0]["dimension"] == "hybrid"


def test_build_query_record_marks_error_streams_failed():
    error_stream = 'event: error\ndata: {"message": "boom"}\n\nevent: done\ndata: {"answer": "", "latency_ms": 10}\n\n'
    events = parse_in_pieces(error_stream, 5)
    record = latency_bench.build_query_record(events, "rag", "q", None, 12.0)
    assert record["ok"] is False
    assert record["errors"] == [{"message": "boom"}]
    assert record["llm_ms"] is None and record["other_ms"] is None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_percentile_and_median_match_known_values():
    values = [float(number) for number in range(1, 11)]  # 1..10
    assert math.isclose(latency_bench.percentile(values, 0.9), 9.1)
    assert math.isclose(latency_bench.percentile(values, 0.5), 5.5)
    assert latency_bench.percentile([7.0], 0.9) == 7.0
    assert latency_bench.percentile([], 0.9) is None
    stats = latency_bench.summarize_values([3, None, 1, "x", 2])
    assert stats == {"n": 3, "median": 2.0, "p90": 2.8, "min": 1.0, "max": 3.0}


def test_summarize_records_groups_by_type_and_skips_failures():
    records = [
        {"query_type": "rag", "ok": True, "client_total_ms": 100.0, "client_ttft_ms": 10.0, "provider": "gemini", "model": "m"},
        {"query_type": "rag", "ok": True, "client_total_ms": 300.0, "client_ttft_ms": 30.0, "provider": "gemini", "model": "m"},
        {"query_type": "data", "ok": True, "client_total_ms": 50.0, "client_ttft_ms": 5.0, "provider": "nvidia", "model": "n"},
        {"query_type": "data", "ok": False, "client_total_ms": 99999.0, "client_ttft_ms": None, "provider": None, "model": None},
    ]
    summary = latency_bench.summarize_records(records)
    assert summary["n_ok"] == 3 and summary["n_failed"] == 1
    assert summary["groups"]["rag"]["client_total_ms"]["median"] == 200.0
    assert summary["groups"]["data"]["client_total_ms"]["n"] == 1
    assert summary["groups"]["all"]["client_total_ms"]["max"] == 300.0
    assert summary["answering_provider_models"] == {"gemini/m": 2, "nvidia/n": 1}
    table = latency_bench.render_warm_table(summary)
    assert "| rag | 2 | 20 / 28 | 200 / 280 |" in table


# ---------------------------------------------------------------------------
# End to end against the real FastAPI app (in-process, no keys -> failure path)
# ---------------------------------------------------------------------------


async def test_run_query_parses_real_server_stream(monkeypatch):
    import main
    from src.llm import providers

    def no_keys(provider):
        return False

    # Force the keyless path even if a developer has keys in .env: no network calls are made.
    monkeypatch.setattr(providers, "has_api_key", no_keys)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench.test", timeout=120) as client:
        record = await latency_bench.run_query(client, "http://bench.test", "rag", "What are the reflection tokens in Self-RAG?")
    assert record["http_status"] == 200
    assert record["server_total_ms"] is not None
    assert record["client_total_ms"] >= record["server_total_ms"] * 0.5
    # No provider key: the supervisor uses the keyword plan, research retrieves, then the LLM fails.
    assert record["plan"] == ["research"]
    assert record["retrieval_calls"] >= 1
    assert record["fallbacks"] >= 1
    assert record["client_ttft_ms"] is not None  # the failure message is streamed as a token
    # ...but an apology with no successful worker must not count as an ok sample.
    assert record["final_agent"] == ""
    assert record["ok"] is False


def test_build_query_record_treats_all_workers_failed_as_not_ok():
    failed_stream = (
        'event: token\ndata: {"text": "Sorry, I could not produce a grounded answer", "node": "finish"}\n\n'
        'event: done\ndata: {"answer": "Sorry", "final_agent": "", "plan": ["research"], "latency_ms": 30.0}\n\n'
    )
    events = parse_in_pieces(failed_stream, 9)
    record = latency_bench.build_query_record(events, "rag", "q", 5.0, 31.0)
    assert record["ok"] is False


async def test_run_query_records_http_errors(monkeypatch):
    import main

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench.test", timeout=30) as client:
        record = await latency_bench.run_query(client, "http://bench.test/wrong-prefix", "rag", "q")
    assert record["http_status"] == 404
    assert record["ok"] is False
    assert record["errors"] and "HTTP 404" in record["errors"][0]["message"]
