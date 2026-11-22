"""Stream events shared by every layer.

Graph nodes, the LLM router and tools call `emit_trace`, `emit_token` and `emit_citation`.
Inside a LangGraph run these go to the graph's custom stream (`get_stream_writer`); the
FastAPI layer turns each payload into one SSE frame whose event name is payload["type"].
Outside a graph run (unit tests, scripts) events go to an optional fallback sink.
"""

import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

TraceKind = Literal[
    "agent_route",  # supervisor picked a worker (or finished), with reason
    "llm_route",  # router picked provider/model, with reason and routing dimension
    "llm_fallback",  # a provider/model failed and the router moved down the chain
    "rag_retrieval",  # retriever returned chunks
    "mcp_tool_call",  # web search/extract through Tavily MCP (or DuckDuckGo fallback)
    "data_query",  # data agent ran SQL
    "agent_result",  # worker finished (ok or failed)
    "llm_result",  # one routed LLM call finished: who answered, or that nobody did
    "error",  # recoverable failure inside a node
]

StreamPayload = dict[str, Any]
StreamSink = Callable[[StreamPayload], None]

# Set by tests/scripts that run components outside a graph and want to capture events.
fallback_sink: ContextVar[StreamSink | None] = ContextVar("fallback_sink", default=None)


@dataclass
class TraceEvent:
    kind: TraceKind
    node: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


def resolve_sink() -> StreamSink | None:
    # get_stream_writer raises outside a LangGraph runnable context; fall back quietly.
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return fallback_sink.get()


def write_payload(payload: StreamPayload) -> None:
    sink = resolve_sink()
    if sink is not None:
        sink(payload)


def emit_trace(event: TraceEvent) -> None:
    write_payload({"type": "trace", "data": asdict(event)})


# A token held back until a node streams its first real token: the separator between two
# workers' answers. Sending it up front left a dangling "---" when the second worker failed
# before writing anything. The holder is a dict so tasks spawned inside the node (which copy
# the context) clear the same object.
pending_prefix: ContextVar[dict | None] = ContextVar("pending_prefix", default=None)


def hold_prefix(text: str, node: str) -> dict:
    holder = {"text": text, "node": node}
    pending_prefix.set(holder)
    return holder


def emit_token(text: str, node: str) -> None:
    holder = pending_prefix.get()
    if holder and holder.get("text") and text:
        prefix_text = holder["text"]
        holder["text"] = None
        write_payload({"type": "token", "data": {"text": prefix_text, "node": holder["node"]}})
    write_payload({"type": "token", "data": {"text": text, "node": node}})


def emit_citation(citation: dict[str, Any]) -> None:
    """citation keys: id (e.g. "[1]"), source_type ("corpus"|"web"), title, source_id, url, section, page, snippet."""
    write_payload({"type": "citation", "data": citation})
