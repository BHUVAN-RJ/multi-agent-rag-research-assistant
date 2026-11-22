"""Shared web tool: Tavily Remote MCP first, DuckDuckGo / plain HTTP as fallback.

Both the research agent and the summary agent call `web_search` / `web_extract`, so every call
records its `caller` and emits one `mcp_tool_call` trace (MCP tool sharing evidence). Nothing in
this module raises for an expected failure: the result carries `error` instead.
"""

import asyncio
import html
import ipaddress
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from src.config import settings
from src.trace_events import TraceEvent, emit_trace

TAVILY_MCP_BASE_URL = "https://mcp.tavily.com/mcp/"
TAVILY_SERVER_NAME = "tavily"
TOOL_TIMEOUT_SECONDS = 20.0
EXTRACT_MAX_CHARS = 8000
ARGS_SUMMARY_MAX_CHARS = 200
DDG_ATTEMPTS = 3
DDG_RETRY_DELAY_SECONDS = 0.7

# Words in a Tavily error/text reply that mean the key is out of credits or rate limited.
QUOTA_ERROR_MARKERS = ("quota", "usage limit", "rate limit", "exceeded", "429", "insufficient credits")

Backend = Literal["tavily_mcp", "duckduckgo", "http_fetch", "none"]


@dataclass
class WebResult:
    title: str
    url: str
    content: str


@dataclass
class WebSearchResult:
    results: list[WebResult]
    backend: Backend
    caller: str
    latency_ms: float
    error: str | None = None


@dataclass
class TavilyToolCache:
    """One MCP client and its discovered tools, reused across calls and callers."""

    client: Any = None
    tools_by_name: dict[str, Any] = field(default_factory=dict)
    lock: asyncio.Lock | None = None


tavily_cache = TavilyToolCache()


class TavilyUnavailableError(Exception):
    """Tavily could not serve this call; the message is the fallback reason shown in traces."""


# ---------------------------------------------------------------------------
# Tavily MCP plumbing
# ---------------------------------------------------------------------------


def build_tavily_url(api_key: str) -> str:
    return f"{TAVILY_MCP_BASE_URL}?tavilyApiKey={api_key}"


def build_mcp_client(api_key: str) -> Any:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connection = {
        "transport": "streamable_http",
        "url": build_tavily_url(api_key),
        "timeout": TOOL_TIMEOUT_SECONDS,
    }
    # handle_tool_errors=False makes an MCP isError reply raise, so quota errors trigger fallback
    # instead of coming back as an error ToolMessage that looks like a normal result.
    return MultiServerMCPClient({TAVILY_SERVER_NAME: connection}, handle_tool_errors=False)


async def get_tavily_tools() -> dict[str, Any]:
    """List the Tavily MCP tools once and cache them; later calls reuse the same client."""
    if tavily_cache.tools_by_name:
        return tavily_cache.tools_by_name
    if tavily_cache.lock is None:
        tavily_cache.lock = asyncio.Lock()
    async with tavily_cache.lock:
        # Another coroutine may have filled the cache while this one waited on the lock.
        if tavily_cache.tools_by_name:
            return tavily_cache.tools_by_name
        if tavily_cache.client is None:
            tavily_cache.client = build_mcp_client(settings.tavily_api_key)
        tool_list = await tavily_cache.client.get_tools(server_name=TAVILY_SERVER_NAME)
        tavily_cache.tools_by_name = {tool.name: tool for tool in tool_list}
    return tavily_cache.tools_by_name


def reset_tavily_cache() -> None:
    tavily_cache.client = None
    tavily_cache.tools_by_name = {}


def find_tool_name(tool_names: list[str], action: str) -> str | None:
    """Pick the tool for `action` ("search" / "extract"); Tavily has used both - and _ separators."""
    preferred_names = [f"tavily_{action}", f"tavily-{action}"]
    for preferred_name in preferred_names:
        if preferred_name in tool_names:
            return preferred_name
    for tool_name in tool_names:
        if action in tool_name.lower():
            return tool_name
    return None


async def call_tavily_tool(action: str, tool_args: dict[str, Any]) -> tuple[str, Any]:
    """Run one Tavily MCP tool. Returns (tool_name, raw_output); raises TavilyUnavailableError."""
    if not settings.tavily_api_key:
        raise TavilyUnavailableError("missing TAVILY_API_KEY")
    try:
        tools_by_name = await asyncio.wait_for(get_tavily_tools(), TOOL_TIMEOUT_SECONDS)
    except TimeoutError as timeout_error:
        raise TavilyUnavailableError(f"timeout listing tools after {TOOL_TIMEOUT_SECONDS:.0f}s") from timeout_error
    except Exception as list_error:
        raise TavilyUnavailableError(f"tool listing failed: {describe_exception(list_error)}") from list_error

    tool_name = find_tool_name(list(tools_by_name), action)
    if tool_name is None:
        raise TavilyUnavailableError(f"no {action} tool among {sorted(tools_by_name)}")
    tool = tools_by_name[tool_name]
    try:
        raw_output = await asyncio.wait_for(tool.ainvoke(tool_args), TOOL_TIMEOUT_SECONDS)
    except TimeoutError as timeout_error:
        raise TavilyUnavailableError(f"timeout after {TOOL_TIMEOUT_SECONDS:.0f}s") from timeout_error
    except Exception as call_error:
        reason = describe_exception(call_error)
        if looks_like_quota_error(reason):
            raise TavilyUnavailableError(f"quota error: {reason}") from call_error
        raise TavilyUnavailableError(f"tool call failed: {reason}") from call_error
    return tool_name, raw_output


# The Tavily key travels in the MCP URL's query string, and httpx puts that URL into its
# error messages. Anything that reaches a trace goes to every browser, so scrub it first.
TAVILY_KEY_IN_URL_PATTERN = re.compile(r"(tavilyApiKey=)[^&'\"\s]+", re.IGNORECASE)
TAVILY_KEY_PATTERN = re.compile(r"tvly-[A-Za-z0-9_-]+")


def redact_secrets(text: str) -> str:
    without_url_key = TAVILY_KEY_IN_URL_PATTERN.sub(r"\1[redacted]", text)
    return TAVILY_KEY_PATTERN.sub("tvly-[redacted]", without_url_key)


def describe_exception(error: BaseException) -> str:
    # anyio TaskGroups wrap transport errors in ExceptionGroups; surface the first real cause.
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    message = redact_secrets(str(error) or type(error).__name__)
    return f"{type(error).__name__}: {message}"[:300]


def looks_like_quota_error(text: str) -> bool:
    lowered_text = text.lower()
    return any(marker in lowered_text for marker in QUOTA_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# Tavily output parsing
# ---------------------------------------------------------------------------


def collect_output_text(raw_output: Any) -> str:
    """Flatten whatever the adapter returned (str, content blocks, ToolMessage) into text."""
    if raw_output is None:
        return ""
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, tuple):
        # content_and_artifact tools return (content, artifact) when invoked with a ToolCall.
        return collect_output_text(raw_output[0])
    if isinstance(raw_output, dict):
        if raw_output.get("type") == "text":
            return str(raw_output.get("text", ""))
        return json.dumps(raw_output)
    if isinstance(raw_output, list):
        block_texts = [collect_output_text(block) for block in raw_output]
        return "\n".join(block_text for block_text in block_texts if block_text)
    message_content = getattr(raw_output, "content", None)
    if message_content is not None:
        return collect_output_text(message_content)
    return str(raw_output)


def web_result_from_mapping(item: dict[str, Any]) -> WebResult | None:
    url = str(item.get("url") or item.get("href") or "").strip()
    if not url:
        return None
    title = str(item.get("title") or url).strip()
    content_value = item.get("content") or item.get("raw_content") or item.get("body") or ""
    return WebResult(title=title, url=url, content=str(content_value).strip())


def parse_json_results(text: str) -> list[WebResult] | None:
    """Parse a JSON payload ({"results": [...]} or a bare list). None if the text is not JSON."""
    stripped_text = text.strip()
    if not stripped_text or stripped_text[0] not in "[{":
        return None
    try:
        payload = json.loads(stripped_text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        items = payload.get("results") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        return None
    parsed_results = [web_result_from_mapping(item) for item in items if isinstance(item, dict)]
    return [result for result in parsed_results if result is not None]


# "Title: ...", "URL: ...", "Content: ...", "Raw Content: ..." lines in Tavily's text format.
FIELD_LINE_PATTERN = re.compile(r"^\s*(title|url|content|raw content)\s*:\s*(.*)$", re.IGNORECASE)


def parse_labeled_text_results(text: str) -> list[WebResult]:
    """Parse Tavily MCP's human-readable format of Title/URL/Content blocks."""
    records: list[dict[str, str]] = []
    current_record: dict[str, str] = {}
    current_field = ""
    for line in text.splitlines():
        field_match = FIELD_LINE_PATTERN.match(line)
        if field_match is None:
            # Continuation lines (multi-line content) belong to the last field seen.
            if current_field and current_record:
                current_record[current_field] += "\n" + line
            continue
        field_name = field_match.group(1).lower().replace(" ", "_")
        field_value = field_match.group(2)
        # A Title always starts a new record; a URL does too if this record already has one.
        starts_new_record = field_name == "title" or (field_name == "url" and "url" in current_record)
        if starts_new_record and current_record:
            records.append(current_record)
            current_record = {}
        current_record[field_name] = field_value
        current_field = field_name
    if current_record:
        records.append(current_record)
    parsed_results = [web_result_from_mapping(record) for record in records]
    return [result for result in parsed_results if result is not None]


def extract_tavily_error(output_text: str) -> str | None:
    """Tavily replies to bad keys / quota with JSON like {"error": ..., "detail": ...}, not isError."""
    stripped_text = output_text.strip()
    if not stripped_text.startswith("{"):
        return None
    try:
        payload = json.loads(stripped_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    error_parts = [str(payload.get("error")), str(payload.get("detail") or "")]
    return ": ".join(part for part in error_parts if part)[:300]


def parse_tavily_output(raw_output: Any) -> list[WebResult]:
    output_text = collect_output_text(raw_output)
    json_results = parse_json_results(output_text)
    if json_results is not None:
        return json_results
    return parse_labeled_text_results(output_text)


# ---------------------------------------------------------------------------
# Fallback backends
# ---------------------------------------------------------------------------


def run_ddgs_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    search_client = DDGS()
    return search_client.text(query, max_results=max_results)


async def run_ddgs_with_retries(query: str, max_results: int) -> list[dict[str, Any]]:
    """ddgs rotates scraped backends and intermittently raises "No results found"; retry briefly."""
    last_error: Exception | None = None
    for attempt_index in range(DDG_ATTEMPTS):
        if attempt_index:
            await asyncio.sleep(DDG_RETRY_DELAY_SECONDS * attempt_index)
        try:
            # ddgs is synchronous; run it in a worker thread so the event loop stays free.
            search_call = asyncio.to_thread(run_ddgs_text_search, query, max_results)
            raw_hits = await asyncio.wait_for(search_call, TOOL_TIMEOUT_SECONDS)
        except TimeoutError:
            raise
        except Exception as search_error:
            last_error = search_error
            continue
        if raw_hits:
            return raw_hits
    if last_error is not None:
        raise last_error
    return []


async def duckduckgo_search(query: str, max_results: int) -> list[WebResult]:
    raw_hits = await run_ddgs_with_retries(query, max_results)
    parsed_hits = [web_result_from_mapping(hit) for hit in raw_hits or []]
    valid_hits = [hit for hit in parsed_hits if hit is not None]
    return valid_hits[:max_results]


# Script/style bodies carry no readable text, so they are dropped before stripping tags.
SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
TITLE_PATTERN = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")


def html_to_text(page_html: str) -> str:
    without_scripts = SCRIPT_STYLE_PATTERN.sub(" ", page_html)
    without_tags = TAG_PATTERN.sub(" ", without_scripts)
    unescaped_text = html.unescape(without_tags)
    collapsed_text = WHITESPACE_PATTERN.sub(" ", unescaped_text)
    return collapsed_text.strip()[:EXTRACT_MAX_CHARS]


def extract_page_title(page_html: str, url: str) -> str:
    title_match = TITLE_PATTERN.search(page_html)
    if title_match is None:
        return url
    title_text = html.unescape(title_match.group(1))
    return WHITESPACE_PATTERN.sub(" ", title_text).strip() or url


async def is_public_http_url(url: str) -> bool:
    """Reject non-http(s) schemes and hosts that resolve to private, loopback or link-local IPs.

    The extract fallback fetches user-supplied URLs from inside Cloud Run, so without this a
    query could make the service request internal addresses (SSRF).
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        return False
    event_loop = asyncio.get_running_loop()
    try:
        address_infos = await event_loop.getaddrinfo(parsed_url.hostname, parsed_url.port or 443)
    except OSError:
        return False
    for address_info in address_infos:
        ip_text = address_info[4][0]
        ip_address = ipaddress.ip_address(ip_text)
        if not ip_address.is_global:
            return False
    return True


async def fetch_page_as_result(http_client: httpx.AsyncClient, url: str) -> WebResult | None:
    if not await is_public_http_url(url):
        return None
    try:
        response = await http_client.get(url)
        response.raise_for_status()
    except Exception:
        return None
    page_html = response.text
    return WebResult(title=extract_page_title(page_html, url), url=url, content=html_to_text(page_html))


async def http_extract(urls: list[str]) -> list[WebResult]:
    request_headers = {"User-Agent": "Mozilla/5.0 (compatible; csci599-research-bot)"}
    # Redirects are not followed: one could bounce to an internal address after the SSRF check.
    async with httpx.AsyncClient(
        timeout=TOOL_TIMEOUT_SECONDS, follow_redirects=False, headers=request_headers
    ) as http_client:
        fetch_tasks = [fetch_page_as_result(http_client, url) for url in urls]
        fetched_pages = await asyncio.gather(*fetch_tasks)
    return [page for page in fetched_pages if page is not None]


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def summarize_args(tool_args: dict[str, Any]) -> str:
    return json.dumps(tool_args, default=str)[:ARGS_SUMMARY_MAX_CHARS]


def emit_tool_trace(
    action: str,
    result: WebSearchResult,
    tool_name: str,
    tool_args: dict[str, Any],
    fallback_reason: str | None,
) -> None:
    if fallback_reason:
        fallback_reason = redact_secrets(fallback_reason)
    summary = f"{result.caller} -> web_{action} via {result.backend} ({len(result.results)} results)"
    if fallback_reason:
        summary += f"; Tavily fallback: {fallback_reason}"
    detail = {
        "caller": result.caller,
        # Only a Tavily answer came from the MCP server; fallbacks are named for what ran.
        "server": TAVILY_SERVER_NAME if result.backend == "tavily_mcp" else result.backend,
        "tool": tool_name,
        "backend": result.backend,
        "args": summarize_args(tool_args),
        "result_count": len(result.results),
        "latency_ms": round(result.latency_ms, 1),
        "fallback_reason": fallback_reason,
        "error": redact_secrets(result.error) if result.error else None,
    }
    emit_trace(TraceEvent(kind="mcp_tool_call", node=result.caller, summary=summary, detail=detail))


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def try_tavily(action: str, tool_args: dict[str, Any]) -> tuple[str, list[WebResult]]:
    """Tavily attempt shared by search and extract; raises TavilyUnavailableError to trigger fallback."""
    tool_name, raw_output = await call_tavily_tool(action, tool_args)
    parsed_results = parse_tavily_output(raw_output)
    if parsed_results:
        return tool_name, parsed_results
    output_text = collect_output_text(raw_output)
    tavily_error = extract_tavily_error(output_text)
    if tavily_error and looks_like_quota_error(tavily_error):
        raise TavilyUnavailableError(f"quota error: {tavily_error}")
    if tavily_error:
        raise TavilyUnavailableError(f"Tavily error reply: {tavily_error}")
    if looks_like_quota_error(output_text):
        raise TavilyUnavailableError(f"quota error: {output_text[:200]}")
    raise TavilyUnavailableError("Tavily returned no parseable results")


async def web_search(query: str, caller: str, max_results: int = 5) -> WebSearchResult:
    start_time = time.perf_counter()
    tool_args = {"query": query, "max_results": max_results}
    try:
        tool_name, tavily_results = await try_tavily("search", tool_args)
        result = WebSearchResult(tavily_results[:max_results], "tavily_mcp", caller, elapsed_ms(start_time))
        emit_tool_trace("search", result, tool_name, tool_args, fallback_reason=None)
        return result
    except TavilyUnavailableError as tavily_error:
        fallback_reason = str(tavily_error)
    except Exception as unexpected_error:
        fallback_reason = f"unexpected: {describe_exception(unexpected_error)}"

    try:
        ddg_results = await duckduckgo_search(query, max_results)
        result = WebSearchResult(ddg_results, "duckduckgo", caller, elapsed_ms(start_time))
        if not ddg_results:
            result.error = "DuckDuckGo returned no results"
        emit_tool_trace("search", result, "ddgs.text", tool_args, fallback_reason)
        return result
    except Exception as ddg_error:
        error_text = redact_secrets(f"Tavily: {fallback_reason}; DuckDuckGo: {describe_exception(ddg_error)}")
        result = WebSearchResult([], "none", caller, elapsed_ms(start_time), error=error_text)
        emit_tool_trace("search", result, "none", tool_args, fallback_reason)
        return result


async def web_extract(urls: list[str], caller: str) -> WebSearchResult:
    start_time = time.perf_counter()
    tool_args = {"urls": urls}
    if not urls:
        result = WebSearchResult([], "none", caller, elapsed_ms(start_time), error="no urls given")
        emit_tool_trace("extract", result, "none", tool_args, fallback_reason=None)
        return result
    try:
        tool_name, tavily_results = await try_tavily("extract", tool_args)
        truncated_results = [
            WebResult(item.title, item.url, item.content[:EXTRACT_MAX_CHARS]) for item in tavily_results
        ]
        result = WebSearchResult(truncated_results, "tavily_mcp", caller, elapsed_ms(start_time))
        emit_tool_trace("extract", result, tool_name, tool_args, fallback_reason=None)
        return result
    except TavilyUnavailableError as tavily_error:
        fallback_reason = str(tavily_error)
    except Exception as unexpected_error:
        fallback_reason = f"unexpected: {describe_exception(unexpected_error)}"

    try:
        fetched_pages = await http_extract(urls)
    except Exception as http_error:
        fetched_pages = []
        fallback_reason += f"; http fetch crashed: {describe_exception(http_error)}"
    if fetched_pages:
        result = WebSearchResult(fetched_pages, "http_fetch", caller, elapsed_ms(start_time))
        emit_tool_trace("extract", result, "httpx.get", tool_args, fallback_reason)
        return result
    error_text = redact_secrets(f"Tavily: {fallback_reason}; HTTP fetch returned nothing")
    result = WebSearchResult([], "none", caller, elapsed_ms(start_time), error=error_text)
    emit_tool_trace("extract", result, "none", tool_args, fallback_reason)
    return result
