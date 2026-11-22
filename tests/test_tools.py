"""Tests for src/tools/web_search.py (Tavily MCP + DuckDuckGo fallback)."""

import asyncio
import dataclasses
import json

import pytest

from src.config import settings
from src.tools import web_search
from src.trace_events import fallback_sink

FAKE_KEY = "tvly-fake-key-for-tests"

# Shape returned by the live Tavily remote MCP: one text content block holding JSON.
TAVILY_JSON_PAYLOAD = [
    {
        "type": "text",
        "text": json.dumps(
            {
                "query": "multi-agent RAG",
                "answer": None,
                "results": [
                    {
                        "title": "AutoGen: Enabling Next-Gen LLM Applications",
                        "url": "https://arxiv.org/abs/2308.08155",
                        "content": "AutoGen is an open-source framework for multi-agent conversation.",
                        "score": 0.91,
                        "raw_content": None,
                    },
                    {
                        "title": "Retrieval-Augmented Generation for LLMs: A Survey",
                        "url": "https://arxiv.org/abs/2312.10997",
                        "content": "RAG incorporates knowledge from external databases.",
                        "score": 0.88,
                    },
                    {"title": "no url entry, must be dropped", "content": "x"},
                ],
                "response_time": 1.2,
            }
        ),
        "id": "lc_123",
    }
]

# Older tavily-mcp text format: labelled Title/URL/Content blocks with multi-line content.
TAVILY_TEXT_PAYLOAD = """Detailed Results:

Title: LangGraph Supervisor
URL: https://github.com/langchain-ai/langgraph-supervisor-py
Content: A Python library for hierarchical
multi-agent systems.

Title: ReAct: Synergizing Reasoning and Acting
URL: https://arxiv.org/abs/2210.03629
Content: Interleaves reasoning traces and actions.
"""


class FakeTavilyTool:
    """Stands in for a langchain StructuredTool from the MCP adapter."""

    def __init__(self, name, output=None, error=None, delay_seconds=0.0):
        self.name = name
        self.output = output
        self.error = error
        self.delay_seconds = delay_seconds
        self.received_args = []

    async def ainvoke(self, tool_args):
        self.received_args.append(tool_args)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture
def captured_events():
    event_list = []
    sink_token = fallback_sink.set(event_list.append)
    yield event_list
    fallback_sink.reset(sink_token)


@pytest.fixture
def fake_key(monkeypatch):
    keyed_settings = dataclasses.replace(settings, tavily_api_key=FAKE_KEY)
    monkeypatch.setattr(web_search, "settings", keyed_settings)
    web_search.reset_tavily_cache()
    yield
    web_search.reset_tavily_cache()


@pytest.fixture
def no_key(monkeypatch):
    keyless_settings = dataclasses.replace(settings, tavily_api_key="")
    monkeypatch.setattr(web_search, "settings", keyless_settings)


def install_fake_tools(monkeypatch, fake_tools):
    tools_by_name = {tool.name: tool for tool in fake_tools}

    async def fake_get_tavily_tools():
        return tools_by_name

    monkeypatch.setattr(web_search, "get_tavily_tools", fake_get_tavily_tools)


def install_fake_ddg(monkeypatch, hits=None, error=None):
    calls = []

    def fake_ddgs_text_search(query, max_results):
        calls.append((query, max_results))
        if error is not None:
            raise error
        return hits if hits is not None else []

    monkeypatch.setattr(web_search, "run_ddgs_text_search", fake_ddgs_text_search)
    return calls


def mcp_traces(event_list):
    trace_payloads = [event["data"] for event in event_list if event["type"] == "trace"]
    return [payload for payload in trace_payloads if payload["kind"] == "mcp_tool_call"]


DDG_HITS = [
    {"title": "DDG hit one", "href": "https://example.org/one", "body": "first body"},
    {"title": "DDG hit two", "href": "https://example.org/two", "body": "second body"},
]


# --- parsing --------------------------------------------------------------------------------


def test_parser_handles_json_content_blocks():
    parsed_results = web_search.parse_tavily_output(TAVILY_JSON_PAYLOAD)
    parsed_urls = [result.url for result in parsed_results]
    assert parsed_urls == ["https://arxiv.org/abs/2308.08155", "https://arxiv.org/abs/2312.10997"]
    assert parsed_results[0].title.startswith("AutoGen")
    assert "multi-agent conversation" in parsed_results[0].content


def test_parser_handles_labelled_text_format():
    parsed_results = web_search.parse_tavily_output(TAVILY_TEXT_PAYLOAD)
    assert len(parsed_results) == 2
    assert parsed_results[0].url == "https://github.com/langchain-ai/langgraph-supervisor-py"
    assert "multi-agent systems." in parsed_results[0].content
    assert parsed_results[1].title == "ReAct: Synergizing Reasoning and Acting"


def test_find_tool_name_accepts_both_separators():
    assert web_search.find_tool_name(["tavily_search", "tavily_extract"], "search") == "tavily_search"
    assert web_search.find_tool_name(["tavily-search", "tavily-extract"], "extract") == "tavily-extract"
    assert web_search.find_tool_name(["tavily_map"], "search") is None


def test_html_to_text_strips_tags_scripts_and_truncates():
    page_html = "<html><script>var bad=1;</script><p>Hello &amp; <b>world</b></p>" + "a" * 20000
    page_text = web_search.html_to_text(page_html)
    assert page_text.startswith("Hello & world")
    assert "bad" not in page_text
    assert len(page_text) == web_search.EXTRACT_MAX_CHARS


# --- Tavily path and fallbacks (mocked) ---------------------------------------------------


async def test_tavily_success_uses_mcp_and_traces_tool(monkeypatch, fake_key, captured_events):
    search_tool = FakeTavilyTool("tavily_search", output=TAVILY_JSON_PAYLOAD)
    install_fake_tools(monkeypatch, [search_tool, FakeTavilyTool("tavily_extract")])
    ddg_calls = install_fake_ddg(monkeypatch, hits=DDG_HITS)

    result = await web_search.web_search("multi-agent RAG", caller="research", max_results=5)

    assert result.backend == "tavily_mcp"
    assert result.error is None
    assert len(result.results) == 2
    assert search_tool.received_args == [{"query": "multi-agent RAG", "max_results": 5}]
    assert ddg_calls == []
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["tool"] == "tavily_search"
    assert trace["detail"]["server"] == "tavily"
    assert trace["detail"]["backend"] == "tavily_mcp"
    assert trace["detail"]["result_count"] == 2
    assert trace["detail"]["fallback_reason"] is None


async def test_tavily_failure_falls_back_to_ddg_with_reason_and_caller(monkeypatch, fake_key, captured_events):
    failing_tool = FakeTavilyTool("tavily_search", error=RuntimeError("upstream 502 bad gateway"))
    install_fake_tools(monkeypatch, [failing_tool])
    install_fake_ddg(monkeypatch, hits=DDG_HITS)

    result = await web_search.web_search("LLM agents", caller="summary", max_results=2)

    assert result.backend == "duckduckgo"
    assert result.caller == "summary"
    assert [hit.url for hit in result.results] == ["https://example.org/one", "https://example.org/two"]
    [trace] = mcp_traces(captured_events)
    assert trace["node"] == "summary"
    assert trace["detail"]["caller"] == "summary"
    assert trace["detail"]["backend"] == "duckduckgo"
    assert trace["detail"]["server"] == "duckduckgo"
    assert "upstream 502 bad gateway" in trace["detail"]["fallback_reason"]
    assert "fallback" in trace["summary"].lower()


async def test_tavily_quota_reply_falls_back(monkeypatch, fake_key, captured_events):
    # Tavily reports quota problems as a JSON error body, not as an MCP isError.
    quota_payload = [{"type": "text", "text": json.dumps({"error": "Usage limit exceeded", "detail": "plan"})}]
    install_fake_tools(monkeypatch, [FakeTavilyTool("tavily_search", output=quota_payload)])
    install_fake_ddg(monkeypatch, hits=DDG_HITS)

    result = await web_search.web_search("RAG", caller="research")

    assert result.backend == "duckduckgo"
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["fallback_reason"].startswith("quota error")


async def test_tavily_timeout_falls_back(monkeypatch, fake_key, captured_events):
    monkeypatch.setattr(web_search, "TOOL_TIMEOUT_SECONDS", 0.05)
    install_fake_tools(monkeypatch, [FakeTavilyTool("tavily_search", output=TAVILY_JSON_PAYLOAD, delay_seconds=1.0)])
    install_fake_ddg(monkeypatch, hits=DDG_HITS)

    result = await web_search.web_search("RAG", caller="research")

    assert result.backend == "duckduckgo"
    [trace] = mcp_traces(captured_events)
    assert "timeout" in trace["detail"]["fallback_reason"]


async def test_missing_key_falls_back_with_reason(monkeypatch, no_key, captured_events):
    install_fake_ddg(monkeypatch, hits=DDG_HITS)

    result = await web_search.web_search("RAG", caller="research")

    assert result.backend == "duckduckgo"
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["fallback_reason"] == "missing TAVILY_API_KEY"


async def test_both_backends_failing_returns_none_without_raising(monkeypatch, no_key, captured_events):
    monkeypatch.setattr(web_search, "DDG_RETRY_DELAY_SECONDS", 0.0)
    install_fake_ddg(monkeypatch, error=ConnectionError("network down"))

    result = await web_search.web_search("RAG", caller="research")

    assert result.backend == "none"
    assert result.results == []
    assert "network down" in result.error
    assert "missing TAVILY_API_KEY" in result.error
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["backend"] == "none"


async def test_caller_propagates_for_two_agents(monkeypatch, fake_key, captured_events):
    install_fake_tools(monkeypatch, [FakeTavilyTool("tavily_search", output=TAVILY_JSON_PAYLOAD)])

    research_result = await web_search.web_search("q1", caller="research")
    summary_result = await web_search.web_search("q2", caller="summary")

    assert research_result.caller == "research"
    assert summary_result.caller == "summary"
    trace_callers = [trace["detail"]["caller"] for trace in mcp_traces(captured_events)]
    trace_nodes = [trace["node"] for trace in mcp_traces(captured_events)]
    assert trace_callers == ["research", "summary"]
    assert trace_nodes == ["research", "summary"]


async def test_tool_list_is_cached_and_client_reused(monkeypatch, fake_key):
    list_calls = []
    search_tool = FakeTavilyTool("tavily_search", output=TAVILY_JSON_PAYLOAD)

    class FakeMcpClient:
        async def get_tools(self, server_name=None):
            list_calls.append(server_name)
            return [search_tool]

    built_clients = []

    def fake_build_mcp_client(api_key):
        built_clients.append(api_key)
        return FakeMcpClient()

    monkeypatch.setattr(web_search, "build_mcp_client", fake_build_mcp_client)

    await web_search.web_search("q1", caller="research")
    await web_search.web_search("q2", caller="summary")

    assert built_clients == [FAKE_KEY]
    assert list_calls == ["tavily"]
    assert len(search_tool.received_args) == 2


async def test_extract_falls_back_to_http(monkeypatch, fake_key, captured_events):
    install_fake_tools(monkeypatch, [FakeTavilyTool("tavily_extract", error=RuntimeError("extract broke"))])

    async def fake_http_extract(urls):
        return [web_search.WebResult("Page", url, "page text") for url in urls]

    monkeypatch.setattr(web_search, "http_extract", fake_http_extract)

    result = await web_search.web_extract(["https://example.org/a"], caller="summary")

    assert result.backend == "http_fetch"
    assert result.results[0].content == "page text"
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["tool"] == "httpx.get"
    # Not served by the MCP server, so the trace must not claim it was.
    assert trace["detail"]["server"] == "http_fetch"
    assert trace["detail"]["caller"] == "summary"
    assert "extract broke" in trace["detail"]["fallback_reason"]


def test_mcp_client_config_uses_streamable_http():
    mcp_client = web_search.build_mcp_client(FAKE_KEY)
    connection = mcp_client.connections["tavily"]
    assert connection["transport"] == "streamable_http"
    assert connection["url"] == f"https://mcp.tavily.com/mcp/?tavilyApiKey={FAKE_KEY}"


# --- live -----------------------------------------------------------------------------------


@pytest.mark.live
async def test_live_duckduckgo_search(captured_events):
    result_hits = await web_search.duckduckgo_search("retrieval augmented generation survey arxiv", 3)
    assert len(result_hits) >= 1
    assert all(hit.url.startswith("http") for hit in result_hits)


@pytest.mark.live
async def test_live_web_search_without_key_uses_duckduckgo(no_key, captured_events):
    result = await web_search.web_search("LangGraph supervisor multi-agent", caller="research", max_results=3)
    assert result.backend == "duckduckgo"
    assert len(result.results) >= 1
    assert result.results[0].url.startswith("http")


@pytest.mark.live
@pytest.mark.skipif(not settings.tavily_api_key, reason="TAVILY_API_KEY not set")
async def test_live_tavily_search(captured_events):
    web_search.reset_tavily_cache()
    result = await web_search.web_search("AutoGen multi-agent conversation framework", caller="research")
    assert result.backend == "tavily_mcp", result.error
    assert len(result.results) >= 1
    [trace] = mcp_traces(captured_events)
    assert trace["detail"]["tool"] in ("tavily_search", "tavily-search")


@pytest.mark.live
@pytest.mark.skipif(not settings.tavily_api_key, reason="TAVILY_API_KEY not set")
async def test_live_tavily_extract(captured_events):
    web_search.reset_tavily_cache()
    result = await web_search.web_extract(["https://arxiv.org/abs/2308.08155"], caller="summary")
    assert result.backend == "tavily_mcp", result.error
    assert len(result.results[0].content) > 100


async def test_ddg_retries_transient_no_results(monkeypatch, no_key, captured_events):
    monkeypatch.setattr(web_search, "DDG_RETRY_DELAY_SECONDS", 0.0)
    attempt_log = []

    def flaky_ddgs_text_search(query, max_results):
        attempt_log.append(query)
        if len(attempt_log) == 1:
            raise RuntimeError("No results found.")
        return DDG_HITS

    monkeypatch.setattr(web_search, "run_ddgs_text_search", flaky_ddgs_text_search)

    result = await web_search.web_search("RAG", caller="research")

    assert result.backend == "duckduckgo"
    assert len(attempt_log) == 2


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://169.254.169.254/computeMetadata/v1/",
        "http://localhost:8080/healthz",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "file:///etc/passwd",
        "ftp://example.com/x",
    ],
)
async def test_http_extract_refuses_internal_and_non_http_urls(blocked_url):
    from src.tools import web_search as web_search_module

    assert await web_search_module.is_public_http_url(blocked_url) is False
    fetched = await web_search_module.http_extract([blocked_url])
    assert fetched == []


# --- secret redaction -----------------------------------------------------------------------


async def test_tavily_http_error_never_puts_api_key_in_trace(monkeypatch, fake_key, captured_events):
    # httpx errors quote the request URL, and the MCP URL carries the key as a query parameter.
    leaking_message = (
        f"Client error '432 ' for url '{web_search.build_tavily_url(FAKE_KEY)}'"
    )
    install_fake_tools(monkeypatch, [FakeTavilyTool("tavily_search", error=RuntimeError(leaking_message))])
    install_fake_ddg(monkeypatch, error=RuntimeError("ddg down"))

    result = await web_search.web_search("RAG", caller="research")

    [trace] = mcp_traces(captured_events)
    serialized_trace = json.dumps(trace)
    assert FAKE_KEY not in serialized_trace
    assert FAKE_KEY not in (result.error or "")
    assert "tavilyApiKey=[redacted]" in trace["detail"]["fallback_reason"]


def test_redact_secrets_scrubs_bare_tavily_keys():
    assert web_search.redact_secrets("bad key tvly-abc_123-XYZ here") == "bad key tvly-[redacted] here"
