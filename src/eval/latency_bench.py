"""Latency benchmark against a running server's POST /api/chat SSE endpoint (runs in `.venv`).

    .venv/bin/python -m src.eval.latency_bench --cold --url https://<service>.run.app   # first request after deploy/idle
    .venv/bin/python -m src.eval.latency_bench --url https://<service>.run.app           # warm runs
    .venv/bin/python -m src.eval.latency_bench --report                                  # table from saved JSON

Per query it records, client side: TTFT (first `token` frame) and total time (until `done`); and
from the server's own trace frames: retrieval latency_ms (`rag_retrieval`), worker LLM latency and
TTFT plus provider/model (`agent_result` detail), routing decisions (`llm_route`), fallbacks
(`llm_fallback`), and token counts when any trace carries them. Warm runs are summarized with
median and p90 per query type; the cold request is stored and reported separately.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
WARM_PATH = RESULTS_DIR / "latency_warm.json"
COLD_PATH = RESULTS_DIR / "latency_cold.json"
DEFAULT_URL = "http://localhost:8080"

# Fixed workload covering every worker type. The RAG queries are from questions.json.
DEFAULT_QUERIES: list[tuple[str, str]] = [
    ("rag", "How does ReAct combine reasoning and acting in language models, and what does the interleaving buy on HotpotQA and Fever?"),
    ("rag", "What are the reflection tokens in Self-RAG and what does each one evaluate?"),
    ("rag", "How do Hypothetical Document Embeddings (HyDE) enable zero-shot dense retrieval without relevance labels?"),
    ("web", "What are the latest news about LLM agent frameworks this week?"),
    ("web", "What recent open-source RAG framework releases happened in 2026?"),
    ("data", "How many papers in the database were published per year?"),
    ("data", "Which authors have the most papers in the database?"),
    ("summary", "Summarize the Reflexion paper."),
    ("summary", "Summarize the key ideas of RAPTOR."),
]

# Fields summarized with median/p90 in the tables.
LATENCY_FIELDS: tuple[str, ...] = ("client_ttft_ms", "client_total_ms", "retrieval_ms", "llm_ms", "llm_ttft_ms", "other_ms")
FIELD_LABELS: dict[str, str] = {
    "client_ttft_ms": "TTFT",
    "client_total_ms": "Total",
    "retrieval_ms": "Retrieval",
    "llm_ms": "Worker LLM",
    "llm_ttft_ms": "LLM TTFT",
    "other_ms": "Other",
}


# ---------------------------------------------------------------------------
# SSE parsing (pure)
# ---------------------------------------------------------------------------


@dataclass
class SSEEvent:
    event: str
    data: Any
    raw_data: str = ""


@dataclass
class SSEParser:
    """Incremental SSE parser: feed arbitrary text chunks, get complete events back.

    Handles frames split across network chunks, `: ping` comment lines, CRLF line endings and
    multi-line `data:` fields. `data` is JSON-decoded when possible, else left as a string.
    """

    buffer: str = ""
    pending_carriage_return: bool = False
    events_seen: int = field(default=0)

    def normalize_newlines(self, text_chunk: str) -> str:
        if self.pending_carriage_return:
            text_chunk = "\r" + text_chunk
            self.pending_carriage_return = False
        # A trailing "\r" may be the first half of a "\r\n" split across chunks; hold it back.
        if text_chunk.endswith("\r"):
            text_chunk = text_chunk[:-1]
            self.pending_carriage_return = True
        crlf_normalized = text_chunk.replace("\r\n", "\n")
        return crlf_normalized.replace("\r", "\n")

    def feed(self, text_chunk: str) -> list[SSEEvent]:
        self.buffer += self.normalize_newlines(text_chunk)
        completed_events: list[SSEEvent] = []
        while "\n\n" in self.buffer:
            frame_text, self.buffer = self.buffer.split("\n\n", 1)
            parsed_event = parse_frame(frame_text)
            if parsed_event is not None:
                completed_events.append(parsed_event)
        self.events_seen += len(completed_events)
        return completed_events

    def finish(self) -> list[SSEEvent]:
        """Flush a final frame that was not followed by a blank line."""
        if self.pending_carriage_return:
            self.pending_carriage_return = False
        remaining_text = self.buffer
        self.buffer = ""
        parsed_event = parse_frame(remaining_text)
        if parsed_event is None:
            return []
        self.events_seen += 1
        return [parsed_event]


def parse_frame(frame_text: str) -> SSEEvent | None:
    event_name = "message"
    data_lines: list[str] = []
    for line in frame_text.split("\n"):
        if not line or line.startswith(":"):
            # Blank line or comment (the server's ": ping" heartbeat).
            continue
        field_name, separator, field_value = line.partition(":")
        if not separator:
            continue
        # Per the SSE spec a single space after the colon is not part of the value.
        if field_value.startswith(" "):
            field_value = field_value[1:]
        if field_name == "event":
            event_name = field_value
        elif field_name == "data":
            data_lines.append(field_value)
    if not data_lines:
        return None
    raw_data = "\n".join(data_lines)
    try:
        decoded_data = json.loads(raw_data)
    except json.JSONDecodeError:
        decoded_data = raw_data
    return SSEEvent(event=event_name, data=decoded_data, raw_data=raw_data)


# ---------------------------------------------------------------------------
# Per-query record from parsed events (pure)
# ---------------------------------------------------------------------------


def trace_events_of_kind(events: list[SSEEvent], kind: str) -> list[dict]:
    matching: list[dict] = []
    for sse_event in events:
        if sse_event.event != "trace" or not isinstance(sse_event.data, dict):
            continue
        if sse_event.data.get("kind") == kind:
            matching.append(sse_event.data)
    return matching


def find_done_data(events: list[SSEEvent]) -> dict:
    for sse_event in reversed(events):
        if sse_event.event == "done" and isinstance(sse_event.data, dict):
            return sse_event.data
    return {}


def collect_token_counts(events: list[SSEEvent]) -> tuple[int | None, int | None]:
    """Sum input/output token counts from any trace detail that carries them (None if none do)."""
    input_total: int | None = None
    output_total: int | None = None
    for sse_event in events:
        if sse_event.event != "trace" or not isinstance(sse_event.data, dict):
            continue
        detail = sse_event.data.get("detail") or {}
        # Workers report input_tokens/output_tokens; the supervisor's planning call uses a planner_ prefix.
        for key_prefix in ("", "planner_"):
            input_tokens = detail.get(f"{key_prefix}input_tokens")
            output_tokens = detail.get(f"{key_prefix}output_tokens")
            if isinstance(input_tokens, int):
                input_total = (input_total or 0) + input_tokens
            if isinstance(output_tokens, int):
                output_total = (output_total or 0) + output_tokens
    return input_total, output_total


def build_query_record(
    events: list[SSEEvent],
    query_type: str,
    query: str,
    client_ttft_ms: float | None,
    client_total_ms: float,
    http_status: int | None = 200,
) -> dict:
    retrieval_traces = trace_events_of_kind(events, "rag_retrieval")
    retrieval_latencies = [float((trace.get("detail") or {}).get("latency_ms") or 0.0) for trace in retrieval_traces]
    retrieval_ms = sum(retrieval_latencies) if retrieval_traces else None

    agent_results: list[dict] = []
    for trace in trace_events_of_kind(events, "agent_result"):
        detail = trace.get("detail") or {}
        agent_results.append(
            {
                "agent": trace.get("node"),
                "ok": detail.get("ok"),
                "provider": detail.get("provider"),
                "model": detail.get("model"),
                "llm_latency_ms": detail.get("llm_latency_ms"),
                "ttft_ms": detail.get("ttft_ms"),
            }
        )
    worker_llm_latencies = [result["llm_latency_ms"] for result in agent_results if isinstance(result["llm_latency_ms"], (int, float))]
    llm_ms = sum(worker_llm_latencies) if worker_llm_latencies else None

    done_data = find_done_data(events)
    final_agent = done_data.get("final_agent") or ""
    final_results = [result for result in agent_results if result["agent"] == final_agent]
    answering_result = final_results[-1] if final_results else (agent_results[-1] if agent_results else {})
    llm_ttft_ms = answering_result.get("ttft_ms") if answering_result else None

    routes: list[dict] = []
    for trace in trace_events_of_kind(events, "llm_route"):
        detail = trace.get("detail") or {}
        routes.append({"node": trace.get("node"), "role": detail.get("role"), "provider": detail.get("provider"), "model": detail.get("model"), "dimension": detail.get("dimension")})
    fallback_count = len(trace_events_of_kind(events, "llm_fallback"))
    error_frames = [sse_event.data for sse_event in events if sse_event.event == "error"]
    input_tokens, output_tokens = collect_token_counts(events)
    token_frames = [sse_event for sse_event in events if sse_event.event == "token"]

    server_total_ms = done_data.get("latency_ms")
    # Everything that is neither retrieval nor the worker's answer call: supervisor planning,
    # web search, SQL, graph overhead and network. Only defined when the parts are known.
    other_ms = None
    if isinstance(server_total_ms, (int, float)) and llm_ms is not None:
        other_ms = max(0.0, float(server_total_ms) - (retrieval_ms or 0.0) - llm_ms)

    return {
        "query_type": query_type,
        "query": query,
        "http_status": http_status,
        # finish_node leaves final_agent empty when every worker failed and it streamed an apology.
        "ok": bool(done_data) and bool(final_agent) and not error_frames and bool(token_frames),
        "client_ttft_ms": client_ttft_ms,
        "client_total_ms": client_total_ms,
        "server_ttft_ms": done_data.get("ttft_ms"),
        "server_total_ms": server_total_ms,
        "retrieval_ms": retrieval_ms,
        "retrieval_calls": len(retrieval_traces),
        "llm_ms": llm_ms,
        "llm_ttft_ms": llm_ttft_ms,
        "other_ms": other_ms,
        "final_agent": final_agent,
        "plan": done_data.get("plan"),
        "provider": answering_result.get("provider") if answering_result else None,
        "model": answering_result.get("model") if answering_result else None,
        "agent_results": agent_results,
        "routes": routes,
        "fallbacks": fallback_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "answer_chars": len(str(done_data.get("answer") or "")),
        "errors": error_frames,
    }


# ---------------------------------------------------------------------------
# Statistics (pure)
# ---------------------------------------------------------------------------


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolation percentile (numpy's default method). None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def summarize_values(raw_values: list[Any]) -> dict:
    numeric_values = [float(value) for value in raw_values if isinstance(value, (int, float))]
    if not numeric_values:
        return {"n": 0, "median": None, "p90": None, "min": None, "max": None}
    return {
        "n": len(numeric_values),
        "median": statistics.median(numeric_values),
        "p90": percentile(numeric_values, 0.9),
        "min": min(numeric_values),
        "max": max(numeric_values),
    }


def summarize_records(records: list[dict]) -> dict:
    """Median/p90 per latency field, overall and per query type. Failed queries are excluded."""
    ok_records = [record for record in records if record.get("ok")]
    query_types = sorted({record["query_type"] for record in ok_records})
    groups: dict[str, list[dict]] = {"all": ok_records}
    for query_type in query_types:
        groups[query_type] = [record for record in ok_records if record["query_type"] == query_type]
    summary: dict[str, dict] = {}
    for group_name, group_records in groups.items():
        field_stats: dict[str, dict] = {}
        for field_name in LATENCY_FIELDS:
            field_values = [record.get(field_name) for record in group_records]
            field_stats[field_name] = summarize_values(field_values)
        summary[group_name] = field_stats
    provider_counts: dict[str, int] = {}
    for record in ok_records:
        provider_model = f"{record.get('provider')}/{record.get('model')}"
        provider_counts[provider_model] = provider_counts.get(provider_model, 0) + 1
    return {
        "n_records": len(records),
        "n_ok": len(ok_records),
        "n_failed": len(records) - len(ok_records),
        "groups": summary,
        "answering_provider_models": provider_counts,
    }


# ---------------------------------------------------------------------------
# Markdown (pure)
# ---------------------------------------------------------------------------


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def render_warm_table(warm_summary: dict) -> str:
    header_cells = [f"{FIELD_LABELS[field_name]} med / p90" for field_name in LATENCY_FIELDS]
    header_line = "| Query type | n | " + " | ".join(header_cells) + " |"
    divider_line = "|---|---|" + "---|" * len(LATENCY_FIELDS)
    lines = [header_line, divider_line]
    for group_name, field_stats in warm_summary.get("groups", {}).items():
        row_cells: list[str] = []
        for field_name in LATENCY_FIELDS:
            stats = field_stats[field_name]
            row_cells.append(f"{format_ms(stats['median'])} / {format_ms(stats['p90'])}")
        group_count = field_stats["client_total_ms"]["n"]
        lines.append(f"| {group_name} | {group_count} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def render_cold_table(cold_record: dict) -> str:
    lines = [
        "| Cold request | TTFT | Total | Retrieval | Worker LLM | LLM TTFT | Other | Provider/model |",
        "|---|---|---|---|---|---|---|---|",
    ]
    provider_model = f"{cold_record.get('provider')}/{cold_record.get('model')}"
    cells = [format_ms(cold_record.get(field_name)) for field_name in LATENCY_FIELDS]
    lines.append(f"| {cold_record.get('query_type')} | " + " | ".join(cells) + f" | {provider_model} |")
    return "\n".join(lines)


def render_report(warm_payload: dict | None, cold_payload: dict | None) -> str:
    sections: list[str] = ["All times in ms. Other = server total - retrieval - worker LLM (supervisor plan, web/SQL tools, overhead)."]
    if cold_payload:
        cold_records = cold_payload.get("records", [])
        sections.append(f"\nCold start ({cold_payload.get('url')}, {cold_payload.get('run_at')}):\n")
        for cold_record in cold_records:
            sections.append(render_cold_table(cold_record))
    if warm_payload:
        warm_summary = warm_payload.get("summary", {})
        sections.append(
            f"\nWarm ({warm_payload.get('url')}, {warm_payload.get('run_at')}, {warm_summary.get('n_ok')}/{warm_summary.get('n_records')} ok):\n"
        )
        sections.append(render_warm_table(warm_summary))
        sections.append(f"\nAnswering provider/model counts: {json.dumps(warm_summary.get('answering_provider_models', {}))}")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


async def run_query(client: httpx.AsyncClient, base_url: str, query_type: str, query: str) -> dict:
    chat_url = f"{base_url.rstrip('/')}/api/chat"
    request_body = {"query": query, "session_id": f"bench-{uuid.uuid4().hex[:12]}"}
    parser = SSEParser()
    events: list[SSEEvent] = []
    client_ttft_ms: float | None = None
    http_status: int | None = None
    started_at = time.perf_counter()
    try:
        async with client.stream("POST", chat_url, json=request_body) as response:
            http_status = response.status_code
            if http_status != 200:
                # Wrong URL, 422 validation, 5xx: record the body as an error, not a fast success.
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:300]
                events.append(SSEEvent(event="error", data={"message": f"HTTP {http_status}: {error_text}"}))
                client_total_ms = (time.perf_counter() - started_at) * 1000
                return build_query_record(events, query_type, query, client_ttft_ms, client_total_ms, http_status)
            async for text_chunk in response.aiter_text():
                new_events = parser.feed(text_chunk)
                for sse_event in new_events:
                    if sse_event.event == "token" and client_ttft_ms is None:
                        client_ttft_ms = (time.perf_counter() - started_at) * 1000
                events.extend(new_events)
                saw_done = any(sse_event.event == "done" for sse_event in new_events)
                if saw_done:
                    break
        events.extend(parser.finish())
    except httpx.HTTPError as request_error:
        events.append(SSEEvent(event="error", data={"message": f"{type(request_error).__name__}: {request_error}"}))
    client_total_ms = (time.perf_counter() - started_at) * 1000
    return build_query_record(events, query_type, query, client_ttft_ms, client_total_ms, http_status)


def select_queries(type_filter: list[str] | None, limit: int | None) -> list[tuple[str, str]]:
    selected = DEFAULT_QUERIES
    if type_filter:
        wanted_types = set(type_filter)
        selected = [pair for pair in selected if pair[0] in wanted_types]
    if limit is not None:
        selected = selected[:limit]
    return selected


def describe_record(record: dict) -> str:
    status_word = "ok" if record["ok"] else "FAILED"
    return (
        f"[{record['query_type']}] {status_word} ttft={format_ms(record['client_ttft_ms'])} total={format_ms(record['client_total_ms'])} "
        f"retrieval={format_ms(record['retrieval_ms'])} llm={format_ms(record['llm_ms'])} via {record['provider']}/{record['model']}"
    )


async def run_benchmark(arguments: argparse.Namespace) -> int:
    timeout = httpx.Timeout(arguments.timeout, connect=30.0)
    queries = select_queries(arguments.types, arguments.limit)
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    async with httpx.AsyncClient(timeout=timeout) as client:
        if arguments.cold:
            # The cold sample must be the first request the instance sees, so no warm-up here.
            query_type, query = queries[0]
            cold_record = await run_query(client, arguments.url, query_type, query)
            print(describe_record(cold_record))
            payload = {"url": arguments.url, "run_at": run_at, "mode": "cold", "records": [cold_record]}
            output_path = arguments.out or COLD_PATH
            write_payload(output_path, payload)
            print(f"wrote {output_path}\n")
            print(render_report(None, payload))
            return 0

        for warmup_index in range(arguments.warmup):
            warmup_record = await run_query(client, arguments.url, "warmup", queries[0][1])
            print(f"warm-up {warmup_index + 1}: {describe_record(warmup_record)}")
            await asyncio.sleep(arguments.pause)

        records: list[dict] = []
        for pass_index in range(arguments.repeat):
            for query_type, query in queries:
                record = await run_query(client, arguments.url, query_type, query)
                record["pass"] = pass_index + 1
                records.append(record)
                print(describe_record(record))
                # Free-tier RPM limits: spacing requests keeps fallbacks out of the warm numbers.
                await asyncio.sleep(arguments.pause)

    payload = {"url": arguments.url, "run_at": run_at, "mode": "warm", "repeat": arguments.repeat, "summary": summarize_records(records), "records": records}
    output_path = arguments.out or WARM_PATH
    write_payload(output_path, payload)
    print(f"wrote {output_path}\n")
    print(render_report(payload, None))
    return 0


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    path.write_text(encoded, encoding="utf-8")


def read_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    raw_text = path.read_text(encoding="utf-8")
    return json.loads(raw_text)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSE latency benchmark for POST /api/chat.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default {DEFAULT_URL})")
    parser.add_argument("--cold", action="store_true", help="send one request and store it as the cold-start sample")
    parser.add_argument("--repeat", type=int, default=2, help="warm passes over the query set")
    parser.add_argument("--warmup", type=int, default=1, help="untimed requests before warm runs")
    parser.add_argument("--types", nargs="*", choices=["rag", "web", "data", "summary"], help="only these query types")
    parser.add_argument("--limit", type=int, help="only the first N queries of the set")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request read timeout, seconds")
    parser.add_argument("--out", type=Path, help="output JSON path")
    parser.add_argument("--report", action="store_true", help="print tables from saved latency_cold/latency_warm JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.report:
        warm_payload = read_payload(WARM_PATH)
        cold_payload = read_payload(COLD_PATH)
        if warm_payload is None and cold_payload is None:
            print(f"no results yet in {RESULTS_DIR}")
            return 1
        print(render_report(warm_payload, cold_payload))
        return 0
    return asyncio.run(run_benchmark(arguments))


if __name__ == "__main__":
    sys.exit(main())
