"""Frontend checks: run the real static/app.js under node and test its pure helpers.

app.js is a browser script; it skips DOM wiring when `document` is undefined and
exports its pure functions through a guarded `module.exports`. Each test sends a
scenario name plus JSON input to a small node harness and asserts on the JSON reply.
Skipped when `node` is not on PATH.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"
NODE_BINARY = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE_BINARY is None, reason="node is not on PATH")

# Node side: load app.js, run one named scenario on the JSON from stdin, print JSON.
NODE_HARNESS = r"""
"use strict";
const fs = require("fs");
const app = require(process.argv[1]);

function parseSseText(sseText) {
  const frames = [];
  function collectFrame(frame) {
    frames.push(frame);
  }
  const parser = app.createSseParser(collectFrame);
  parser.push(sseText);
  parser.flush();
  return frames;
}

function routingFromSse(sseText) {
  const routing = app.createRoutingState();
  const frames = parseSseText(sseText);
  for (const frame of frames) {
    if (frame.event === "trace") {
      const traceData = JSON.parse(frame.data);
      app.applyTraceToRouting(routing, traceData);
    }
  }
  return routing;
}

function describeAllCalls(routing) {
  const descriptions = [];
  for (const call of routing.llmCalls) {
    const description = app.describeLlmCall(call);
    descriptions.push({ node: call.node, provider: call.provider, model: call.model,
      reason: call.reason, failures: call.failures.length, ...description });
  }
  return descriptions;
}

function runMarkdown(input) {
  return { html: app.renderMarkdown(input.text, 1) };
}

function runRouting(input) {
  const routing = routingFromSse(input.sse);
  return {
    agentSteps: routing.agentSteps,
    finalAgent: routing.finalAgent,
    fallbackCount: routing.fallbackCount,
    answeringModel: app.answeringModelOfTurn(routing),
    calls: describeAllCalls(routing),
  };
}

function runCitation(input) {
  return app.describeCitationSource(input.citation, "");
}

function runSegments(input) {
  const segments = [];
  for (const token of input.tokens) {
    app.appendTokenSegment(segments, token.node, token.text);
  }
  const keptSegments = app.segmentsWithoutNode(segments, input.restartNode);
  return { before: app.joinSegmentTexts(segments), after: app.joinSegmentTexts(keptSegments) };
}

function runErrorDetail(input) {
  return { text: app.formatErrorDetail(input.body, input.status) };
}

const SCENARIOS = {
  markdown: runMarkdown,
  routing: runRouting,
  citation: runCitation,
  segments: runSegments,
  errorDetail: runErrorDetail,
};

const request = JSON.parse(fs.readFileSync(0, "utf8"));
const runScenario = SCENARIOS[request.scenario];
const result = runScenario(request.input);
process.stdout.write(JSON.stringify(result));
"""


def run_app_scenario(scenario: str, scenario_input: dict) -> dict:
    """Run one harness scenario against the real static/app.js and return its JSON."""
    request_text = json.dumps({"scenario": scenario, "input": scenario_input})
    completed = subprocess.run(
        [NODE_BINARY, "-e", NODE_HARNESS, "--", str(APP_JS)],
        input=request_text,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def render_markdown(markdown_text: str) -> str:
    return run_app_scenario("markdown", {"text": markdown_text})["html"]


# Trace frames copied verbatim from a grader capture of POST /api/chat (r1.sse):
# research planned gemini/gemini-3.8-flash, got HTTP 429, fell back to nvidia.
R1_FALLBACK_TRACE_SSE = 'event: trace\ndata: {"kind": "llm_route", "node": "supervisor", "summary": "supervisor -> nvidia/nvidia/nemotron-3.5-lightning-30b-a3b [cost]", "detail": {"role": "supervisor", "provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "dimension": "cost", "reason": "cost: supervisor routing is short classification -> cheapest low-latency NVIDIA fast model", "fallback_chain": [["gemini", "gemini-3.5-flash-lite"], ["openrouter", "openrouter/free"], ["openrouter", "google/gemma-4-31b-it:free"], ["openrouter", "nvidia/nemotron-3-super-120b-a12b:free"], ["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"]]}, "ts": 1793062630.5737023}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "supervisor", "summary": "supervisor -> research", "detail": {"next_agent": "research", "reason": "plan [\'research\'] (llm): The question asks for a technical comparison of vector search indexes (HNSW vs IVF) and guidance on when to choose each, which requires up-to-date conceptual explanations from the corpus or web.", "plan_queue": [], "tried_agents": [], "failed_agents": [], "visit": 1, "planner_latency_ms": 1024.0845590000022, "planner_input_tokens": 346, "planner_output_tokens": 60}, "ts": 1793062631.5977747}\n\nevent: trace\ndata: {"kind": "mcp_tool_call", "node": "research", "summary": "research -> web_search via tavily_mcp (4 results)", "detail": {"caller": "research", "server": "tavily", "tool": "tavily_search", "backend": "tavily_mcp", "args": "{\\"query\\": \\"What is the difference between HNSW and IVF indexes for vector search, and when would you choose each?\\", \\"max_results\\": 4}", "result_count": 4, "latency_ms": 4346.2, "fallback_reason": null, "error": null}, "ts": 1793062635.9470072}\n\nevent: trace\ndata: {"kind": "rag_retrieval", "node": "retriever", "summary": "retrieved 5 chunks from 4 papers", "detail": {"query": "What is the difference between HNSW and IVF indexes for vector search, and when would you choose each?", "k": 5, "chunk_ids": ["2310.08560v2::chunk-0039", "2004.12832v2::chunk-0035", "2312.10997v5::chunk-0041", "2004.04906v3::chunk-0037", "2004.12832v2::chunk-0043"], "source_ids": ["2310.08560", "2004.12832", "2312.10997", "2004.04906", "2004.12832"], "scores": [0.1617, 0.006, 0.0042, 0.0017, 0.0014], "latency_ms": 7772.1, "reranked": true, "error": null}, "ts": 1793062639.3732283}\n\nevent: trace\ndata: {"kind": "llm_route", "node": "research", "summary": "research -> gemini/gemini-3.8-flash [capability]", "detail": {"role": "research", "provider": "gemini", "model": "gemini-3.8-flash", "dimension": "capability", "reason": "capability: complex multi-source synthesis -> strongest generator, Gemini strong", "fallback_chain": [["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"], ["gemini", "gemini-3.5-flash-lite"], ["openrouter", "openrouter/free"], ["openrouter", "google/gemma-4-31b-it:free"], ["openrouter", "nvidia/nemotron-3-super-120b-a12b:free"]]}, "ts": 1793062639.373982}\n\nevent: trace\ndata: {"kind": "llm_fallback", "node": "research", "summary": "gemini/gemini-3.8-flash failed (gemini/gemini-3.8-flash: HTTP 429: {\\n  \\"error\\": {\\n    \\"code\\": 429,\\n    \\"message\\": \\"You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \\\\n* Quota ex); next: nvidia/nvidia/nemotron-3-ultra-550b-a55b", "detail": {"provider": "gemini", "model": "gemini-3.8-flash", "ok": false, "error": "gemini/gemini-3.8-flash: HTTP 429: {\\n  \\"error\\": {\\n    \\"code\\": 429,\\n    \\"message\\": \\"You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \\\\n* Quota ex", "status_code": 429, "rate_limited": true, "latency_ms": 1244.4, "next": ["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"], "stream_restarted": false}, "ts": 1793062640.618439}\n\nevent: trace\ndata: {"kind": "agent_result", "node": "research", "summary": "research ok", "detail": {"ok": true, "provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b", "citations": 9, "error": null, "corpus_chunks": 5, "web_results": 4, "web_error": null, "llm_latency_ms": 5588.899762000001, "ttft_ms": 533.6321789999943, "input_tokens": 2200, "output_tokens": 614}, "ts": 1793062644.962881}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "supervisor", "summary": "supervisor -> finish", "detail": {"next_agent": "finish", "reason": "research done; plan complete", "plan_queue": [], "tried_agents": ["research"], "failed_agents": [], "visit": 2}, "ts": 1793062644.9655817}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "finish", "summary": "answer from research", "detail": {"final_agent": "research", "tried_agents": ["research"], "failed_agents": []}, "ts": 1793062644.9677503}\n\n'

# Trace frames from a grader capture (r2.sse): data agent with two llm_route calls.
R2_DATA_TRACE_SSE = 'event: trace\ndata: {"kind": "llm_route", "node": "supervisor", "summary": "supervisor -> nvidia/nvidia/nemotron-3.5-lightning-30b-a3b [cost]", "detail": {"role": "supervisor", "provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "dimension": "cost", "reason": "cost: supervisor routing is short classification -> cheapest low-latency NVIDIA fast model", "fallback_chain": [["gemini", "gemini-3.5-flash-lite"], ["openrouter", "openrouter/free"], ["openrouter", "google/gemma-4-31b-it:free"], ["openrouter", "nvidia/nemotron-3-super-120b-a12b:free"], ["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"]]}, "ts": 1793062664.2989235}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "supervisor", "summary": "supervisor -> data", "detail": {"next_agent": "data", "reason": "plan [\'data\'] (llm): The user wants to know which authors appear most often in the paper database, which requires counting author frequencies from the arXiv metadata.", "plan_queue": [], "tried_agents": [], "failed_agents": [], "visit": 1, "planner_latency_ms": 633.4243630000174, "planner_input_tokens": 333, "planner_output_tokens": 46}, "ts": 1793062664.9323485}\n\nevent: trace\ndata: {"kind": "llm_route", "node": "data", "summary": "data -> nvidia/nvidia/nemotron-3.5-lightning-30b-a3b [cost]", "detail": {"role": "data", "provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "dimension": "cost", "reason": "cost: simple structured SQL generation -> cheapest NVIDIA fast model", "fallback_chain": [["gemini", "gemini-3.5-flash-lite"], ["openrouter", "openrouter/free"], ["openrouter", "google/gemma-4-31b-it:free"], ["openrouter", "nvidia/nemotron-3-super-120b-a12b:free"], ["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"]]}, "ts": 1793062664.9349349}\n\nevent: trace\ndata: {"kind": "data_query", "node": "data", "summary": "SQL -> 50 rows", "detail": {"sql": "SELECT a.name, COUNT(*) AS paper_count\\nFROM authors a\\nJOIN paper_authors pa ON a.author_id = pa.author_id\\nGROUP BY a.author_id\\nORDER BY paper_count DESC\\nLIMIT 50", "row_count": 50, "columns": ["name", "paper_count"], "rows_preview": [["Fei Huang", 10], ["Jinjie Gu", 6], ["Hao Wang", 6], ["Jun Wang", 6], ["Pengjun Xie", 6]], "truncated": false, "latency_ms": 6.0, "error": null}, "ts": 1793062665.6462884}\n\nevent: trace\ndata: {"kind": "llm_route", "node": "data", "summary": "data -> nvidia/nvidia/nemotron-3-ultra-550b-a55b [capability]", "detail": {"role": "data", "provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b", "dimension": "capability", "reason": "capability: complex SQL (joins/aggregates) -> stronger NVIDIA Nemotron model", "fallback_chain": [["gemini", "gemini-3.8-flash"], ["nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"], ["openrouter", "openrouter/free"], ["openrouter", "google/gemma-4-31b-it:free"], ["openrouter", "nvidia/nemotron-3-super-120b-a12b:free"]]}, "ts": 1793062665.6468508}\n\nevent: trace\ndata: {"kind": "agent_result", "node": "data", "summary": "data ok", "detail": {"ok": true, "provider": "nvidia", "model": "nvidia/nemotron-3.5-lightning-30b-a3b", "citations": 0, "error": null, "sql": "SELECT a.name, COUNT(*) AS paper_count\\nFROM authors a\\nJOIN paper_authors pa ON a.author_id = pa.author_id\\nGROUP BY a.author_id\\nORDER BY paper_count DESC\\nLIMIT 50", "rows": 50, "sql_attempts": 1, "llm_latency_ms": 15062.700307999989, "ttft_ms": 247.70329500000798, "input_tokens": 1967, "output_tokens": 775}, "ts": 1793062680.0045185}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "supervisor", "summary": "supervisor -> finish", "detail": {"next_agent": "finish", "reason": "data done; plan complete", "plan_queue": [], "tried_agents": ["data"], "failed_agents": [], "visit": 2}, "ts": 1793062680.0090191}\n\nevent: trace\ndata: {"kind": "agent_route", "node": "finish", "summary": "answer from data", "detail": {"final_agent": "data", "tried_agents": ["data"], "failed_agents": []}, "ts": 1793062680.0113401}\n\n'

# Head of the real data-agent answer from r2.sse (a GFM table).
R2_DATA_ANSWER = 'The top 50 authors in the database, ordered by paper count, are shown below. The author with the most papers is **Fei Huang** with 10 papers.\n\n| Rank | Author | Paper Count |\n|------|--------|-------------|\n| 1 | Fei Huang | 10 |\n| 2 | Jinjie Gu | 6 |\n| 3 | Hao Wang | 6 |\n| 4 | Jun Wang | 6 |\n| 5 | Pengjun Xie | 6 |'


# --------------------------------------------------------------- markdown tables


def test_real_data_answer_table_renders_as_html_table():
    html = render_markdown(R2_DATA_ANSWER)
    assert "<table>" in html
    assert "<th>Rank</th><th>Author</th><th>Paper Count</th>" in html
    assert "<td>1</td><td>Fei Huang</td><td>10</td>" in html
    # The pipes must not survive as paragraph text.
    assert "| Rank |" not in html
    assert "<strong>Fei Huang</strong>" in html


def test_table_alignment_and_inline_formatting_in_cells():
    html = render_markdown("| L | C | R |\n|:---|:---:|---:|\n| `a` | **b** | [1] |")
    assert '<th class="align-left">L</th>' in html
    assert '<th class="align-center">C</th>' in html
    assert '<th class="align-right">R</th>' in html
    assert '<td class="align-left"><code>a</code></td>' in html
    assert "<strong>b</strong>" in html
    assert 'class="cite-ref"' in html


def test_table_cell_html_is_escaped():
    html = render_markdown('| Name |\n|---|\n| <script>alert(1)</script> |\n| <img src=x onerror="alert(1)"> |')
    assert "<td>&lt;script&gt;alert(1)&lt;/script&gt;</td>" in html
    assert "<script" not in html
    assert "<img" not in html


def test_pipe_line_without_delimiter_stays_a_paragraph():
    html = render_markdown("a | b\nnot a delimiter")
    assert "<table" not in html
    assert html.startswith("<p>")


def test_lone_dash_rule_is_still_a_horizontal_rule():
    assert render_markdown("above\n\n---\n\nbelow") == "<p>above</p><hr><p>below</p>"


# --------------------------------------------------------- inline code in links


def test_inline_code_inside_link_text_restores_fully():
    html = render_markdown("see [`x`](https://a.com) now")
    assert html == (
        '<p>see <a href="https://a.com" target="_blank" rel="noopener noreferrer">'
        "<code>x</code></a> now</p>"
    )
    assert "\u0000" not in html


def test_link_url_cannot_capture_a_code_placeholder():
    html = render_markdown("[t](https://e.com/`x`)")
    assert "\u0000" not in html
    assert 'href="https://e.com/<code>' not in html


# ------------------------------------------------------------- routing summary


def test_fallback_capture_shows_answering_provider_differs_from_planned():
    summary = run_app_scenario("routing", {"sse": R1_FALLBACK_TRACE_SSE})
    research_calls = [call for call in summary["calls"] if call["node"] == "research"]
    assert len(research_calls) == 1
    research_call = research_calls[0]
    # Planned by llm_route, with its reason...
    assert research_call["plannedLabel"] == "gemini / gemini-3.8-flash"
    assert research_call["reason"].startswith("capability:")
    # ...failed with the 429 from llm_fallback...
    assert research_call["failures"] == 1
    assert "HTTP 429" in research_call["failureLines"][0]
    assert "next nvidia / nvidia/nemotron-3-ultra-550b-a55b" in research_call["failureLines"][0]
    # ...and agent_result names the model that actually answered.
    assert research_call["status"] == "replaced"
    assert research_call["answeredLabel"] == "nvidia / nvidia/nemotron-3-ultra-550b-a55b"
    assert summary["fallbackCount"] == 1
    assert summary["finalAgent"] == "research"
    assert summary["answeringModel"] == "nvidia / nvidia/nemotron-3-ultra-550b-a55b"


def test_no_fallback_capture_marks_call_as_planned():
    summary = run_app_scenario("routing", {"sse": R2_DATA_TRACE_SSE})
    data_calls = [call for call in summary["calls"] if call["node"] == "data"]
    assert len(data_calls) == 2
    statuses = [call["status"] for call in data_calls]
    # agent_result reports the lightning model, which the first (cost) call planned.
    assert statuses == ["as-planned", "pending"]
    assert summary["fallbackCount"] == 0
    assert summary["answeringModel"] == "nvidia / nvidia/nemotron-3.5-lightning-30b-a3b"


def test_openrouter_free_router_resolution_counts_as_planned():
    sse = (
        'event: trace\ndata: {"kind": "llm_route", "node": "summary", "summary": "s", '
        '"detail": {"provider": "openrouter", "model": "openrouter/free", "reason": "cost: cheap"}}\n\n'
        'event: trace\ndata: {"kind": "agent_result", "node": "summary", "summary": "summary ok", '
        '"detail": {"ok": true, "provider": "openrouter", "model": "openrouter/free -> some/model:free"}}\n\n'
    )
    summary = run_app_scenario("routing", {"sse": sse})
    assert summary["calls"][0]["status"] == "as-planned"
    assert summary["calls"][0]["answeredLabel"] == "openrouter / openrouter/free -> some/model:free"


def test_agent_steps_exclude_finish():
    summary = run_app_scenario("routing", {"sse": R1_FALLBACK_TRACE_SSE})
    step_names = [step["name"] for step in summary["agentSteps"]]
    assert step_names == ["research"]


# ------------------------------------------------------------------ citations


def test_database_citation_has_no_arxiv_prefix_and_sql_snippet():
    citation = {
        "id": "[3]", "source_type": "database", "title": "papers.db query result (12 rows)",
        "source_id": "papers.db", "url": "", "section": "", "page": None,
        "snippet": "SELECT COUNT(*) FROM papers",
    }
    description = run_app_scenario("citation", {"citation": citation})
    assert description == {"kind": "database", "tag": "SQL", "parts": ["papers.db"], "snippetIsCode": True}


def test_corpus_citation_keeps_arxiv_location():
    citation = {"id": "[1]", "source_type": "corpus", "source_id": "2210.03629", "page": 3, "section": "Intro"}
    description = run_app_scenario("citation", {"citation": citation})
    assert description["kind"] == "corpus"
    assert description["parts"] == ["arXiv:2210.03629", "p. 3", "§ Intro"]
    assert description["snippetIsCode"] is False


# ------------------------------------------------------- answer token segments


def test_separator_tokens_survive_a_later_worker_restart():
    tokens = [
        {"node": "research", "text": "First answer."},
        {"node": "separator", "text": "\n\n---\n\n"},
        {"node": "summary", "text": "partial that will be dropped"},
    ]
    result = run_app_scenario("segments", {"tokens": tokens, "restartNode": "summary"})
    assert result["before"] == "First answer.\n\n---\n\npartial that will be dropped"
    assert result["after"] == "First answer.\n\n---\n\n"


def test_restart_of_first_worker_keeps_separator_and_second_worker():
    tokens = [
        {"node": "research", "text": "broken"},
        {"node": "separator", "text": "\n\n---\n\n"},
        {"node": "summary", "text": "Second."},
    ]
    result = run_app_scenario("segments", {"tokens": tokens, "restartNode": "research"})
    assert result["after"] == "\n\n---\n\nSecond."


# ------------------------------------------------- llm_result call resolution


def trace_frame(kind: str, node: str, detail: dict, summary: str = "") -> str:
    payload = {"kind": kind, "node": node, "summary": summary or kind, "detail": detail, "ts": 1.0}
    return f"event: trace\ndata: {json.dumps(payload)}\n\n"


NEMOTRON_FAST = "nvidia/nemotron-3.5-lightning-30b-a3b"
NEMOTRON_STRONG = "nvidia/nemotron-3-ultra-550b-a55b"


def route_frame(node: str, call_id: str, provider: str, model: str) -> str:
    detail = {"role": node, "provider": provider, "model": model, "dimension": "cost", "reason": "cost: x", "call_id": call_id}
    return trace_frame("llm_route", node, detail)


def result_frame(node: str, call_id: str | None, provider: str, model: str, ok: bool = True, error: str | None = None) -> str:
    detail = {"role": node, "provider": provider, "model": model, "ok": ok, "attempts": 1, "latency_ms": 10.0, "error": error}
    if call_id is not None:
        detail["call_id"] = call_id
    return trace_frame("llm_result", node, detail)


def test_llm_result_resolves_supervisor_and_both_data_calls():
    # Grader finding: the supervisor call and the data agent's SQL-writing call stayed "pending".
    sse = "".join([
        route_frame("supervisor", "c1", "nvidia", NEMOTRON_FAST),
        result_frame("supervisor", "c1", "nvidia", NEMOTRON_FAST),
        route_frame("data", "c2", "nvidia", NEMOTRON_FAST),
        result_frame("data", "c2", "nvidia", NEMOTRON_FAST),
        route_frame("data", "c3", "nvidia", NEMOTRON_STRONG),
        result_frame("data", "c3", "nvidia", NEMOTRON_STRONG),
        trace_frame("agent_result", "data", {"ok": True, "provider": "nvidia", "model": NEMOTRON_FAST}),
    ])
    summary = run_app_scenario("routing", {"sse": sse})
    statuses = [(call["node"], call["status"], call["answeredLabel"]) for call in summary["calls"]]
    assert statuses == [
        ("supervisor", "as-planned", f"nvidia / {NEMOTRON_FAST}"),
        ("data", "as-planned", f"nvidia / {NEMOTRON_FAST}"),
        # agent_result names the first model; it must not overwrite the second call's own result.
        ("data", "as-planned", f"nvidia / {NEMOTRON_STRONG}"),
    ]


def test_llm_result_without_call_id_pairs_calls_in_order():
    sse = "".join([
        route_frame("data", "", "nvidia", NEMOTRON_FAST),
        route_frame("data", "", "nvidia", NEMOTRON_STRONG),
        result_frame("data", None, "nvidia", NEMOTRON_FAST),
        result_frame("data", None, "gemini", "gemini-3.8-flash"),
    ])
    summary = run_app_scenario("routing", {"sse": sse})
    assert [call["status"] for call in summary["calls"]] == ["as-planned", "replaced"]


def test_fallback_and_result_pair_by_call_id_not_latest_call():
    # The research fallback arrives after a later llm_route from another node.
    fallback_detail = {
        "provider": "gemini", "model": "gemini-3.8-flash", "ok": False, "error": "gemini/gemini-3.8-flash: HTTP 429: quota",
        "status_code": 429, "next": ["nvidia", NEMOTRON_STRONG], "call_id": "r1", "stream_restarted": False,
    }
    sse = "".join([
        route_frame("research", "r1", "gemini", "gemini-3.8-flash"),
        route_frame("research", "r2", "gemini", "gemini-3.5-flash-lite"),
        trace_frame("llm_fallback", "research", fallback_detail),
        result_frame("research", "r1", "nvidia", NEMOTRON_STRONG),
    ])
    summary = run_app_scenario("routing", {"sse": sse})
    first_call, second_call = summary["calls"]
    assert first_call["failures"] == 1
    assert first_call["status"] == "replaced"
    assert second_call["failures"] == 0
    assert second_call["status"] == "pending"


def test_cooldown_skip_is_shown_as_skipped_not_failed():
    skip_detail = {
        "provider": "openrouter", "model": "openrouter/free", "ok": False,
        "error": "skipped: cooling down (42s left), tried last", "next": ["gemini", "gemini-3.5-flash-lite"],
        "skipped": True, "reason": "cooldown", "cooldown_remaining_s": 42.0, "call_id": "s1", "stream_restarted": False,
    }
    sse = "".join([
        route_frame("summary", "s1", "openrouter", "openrouter/free"),
        trace_frame("llm_fallback", "summary", skip_detail),
        result_frame("summary", "s1", "gemini", "gemini-3.5-flash-lite"),
    ])
    summary = run_app_scenario("routing", {"sse": sse})
    call = summary["calls"][0]
    assert call["failureLines"][0].startswith("skipped: openrouter / openrouter/free: skipped: cooling down (42s left)")
    assert "next gemini / gemini-3.5-flash-lite" in call["failureLines"][0]
    assert call["status"] == "replaced"
    assert call["answeredLabel"] == "gemini / gemini-3.5-flash-lite"


def test_llm_result_failure_marks_call_all_failed():
    sse = "".join([
        route_frame("supervisor", "c9", "nvidia", NEMOTRON_FAST),
        result_frame("supervisor", "c9", "", "", ok=False, error="all 6 provider attempts failed: x"),
    ])
    summary = run_app_scenario("routing", {"sse": sse})
    assert summary["calls"][0]["status"] == "all-failed"


# ------------------------------------------------------------- error details


def test_validation_422_shows_field_and_message_not_raw_json():
    body = {"detail": [{"field": "body.query", "message": "String should have at most 4000 characters"}]}
    text = run_app_scenario("errorDetail", {"body": body, "status": 422})["text"]
    assert text == "body.query: String should have at most 4000 characters"
    assert "{" not in text


def test_error_detail_still_reads_fastapi_msg_and_plain_strings():
    fastapi_body = {"detail": [{"loc": ["body", "query"], "msg": "Field required", "type": "missing"}]}
    fastapi_text = run_app_scenario("errorDetail", {"body": fastapi_body, "status": 422})["text"]
    assert fastapi_text == "body.query: Field required"
    plain_text = run_app_scenario("errorDetail", {"body": {"detail": "busy"}, "status": 503})["text"]
    assert plain_text == "busy"
    empty_text = run_app_scenario("errorDetail", {"body": {}, "status": 500})["text"]
    assert empty_text == "HTTP 500"
