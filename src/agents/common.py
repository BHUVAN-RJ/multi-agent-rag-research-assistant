"""Shared types and helpers for worker agents: outputs, numbered citations, context blocks."""

import re
from dataclasses import dataclass, field
from typing import Any

from src.trace_events import TraceEvent, emit_citation, emit_trace

AGENT_NAMES = ("research", "summary", "data")


@dataclass
class AgentOutput:
    agent: str
    ok: bool
    text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def citation_from_chunk(index: int, chunk: Any) -> dict[str, Any]:
    snippet = chunk.text[:280]
    return {
        "id": f"[{index}]",
        "source_type": "corpus",
        "title": chunk.title,
        "source_id": chunk.source_id,
        "url": chunk.url,
        "section": chunk.section,
        "page": chunk.page,
        "snippet": snippet,
        "chunk_id": chunk.chunk_id,
    }


def citation_from_web(index: int, web_result: Any) -> dict[str, Any]:
    snippet = web_result.content[:280]
    return {
        "id": f"[{index}]",
        "source_type": "web",
        "title": web_result.title,
        "source_id": web_result.url,
        "url": web_result.url,
        "section": "",
        "page": None,
        "snippet": snippet,
    }


# Between numbered sources in a prompt. Blank lines, not a markdown "---": models copied that
# rule into their answers, which the UI then showed as an extra section break.
SOURCE_BLOCK_SEPARATOR = "\n\n\n"

# Characters of each web result that reach the prompt, unless the caller asks for more.
WEB_RESULT_PROMPT_CHARS = 2000


def citation_from_sql(index: int, sql_text: str, row_count: int) -> dict[str, Any]:
    """The query result is the data agent's source: the panel shows the SQL that produced it."""
    return {
        "id": f"[{index}]",
        "source_type": "database",
        "title": f"papers.db query result ({row_count} rows)",
        "source_id": "papers.db",
        "url": "",
        "section": "",
        "page": None,
        "snippet": sql_text[:280],
    }


def build_numbered_sources(
    chunks: list[Any], web_results: list[Any], start_index: int = 1, web_chars: int | None = None
) -> tuple[list[dict], str]:
    """Number corpus chunks first, then web results, and render the context block the LLM sees.

    The same numbering is used in the prompt and in the citation events, so an [n] in the
    answer always resolves to the citation panel entry with id [n]. `start_index` continues
    numbering after citations earlier workers in the same turn already published.
    """
    citations: list[dict[str, Any]] = []
    context_lines: list[str] = []
    next_index = start_index
    for chunk in chunks:
        citation = citation_from_chunk(next_index, chunk)
        citations.append(citation)
        header = f"[{next_index}] (corpus) {chunk.title} | arXiv:{chunk.source_id} | p.{chunk.page} {chunk.section}"
        context_lines.append(f"{header}\n{chunk.text}")
        next_index += 1
    for web_result in web_results:
        citation = citation_from_web(next_index, web_result)
        citations.append(citation)
        web_body = web_result.content[: web_chars or WEB_RESULT_PROMPT_CHARS]
        header = f"[{next_index}] (web) {web_result.title} | {web_result.url}"
        context_lines.append(f"{header}\n{web_body}")
        next_index += 1
    context_block = SOURCE_BLOCK_SEPARATOR.join(context_lines)
    return citations, context_block


# "[2]", "[1][3]", "[1, 2, 3]" and "[2-4]" all cite sources; the numbers are what count.
CITATION_MARKER_PATTERN = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")
CITATION_RANGE_PATTERN = re.compile(r"(\d+)\s*[\u2013-]\s*(\d+)")
# A range wider than this is not a citation ("[1-100]" in prose), so it is read as its ends only.
MAX_CITATION_RANGE = 10


def cited_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for marker_match in CITATION_MARKER_PATTERN.finditer(text):
        marker_body = marker_match.group(1)
        for part in marker_body.split(","):
            range_match = CITATION_RANGE_PATTERN.fullmatch(part.strip())
            if range_match:
                first, last = int(range_match.group(1)), int(range_match.group(2))
                if 0 <= last - first <= MAX_CITATION_RANGE:
                    numbers.update(range(first, last + 1))
                else:
                    numbers.update({first, last})
            elif part.strip().isdigit():
                numbers.add(int(part.strip()))
    return numbers


def citation_number(citation: dict[str, Any]) -> int:
    return int(citation["id"].strip("[]"))


def select_cited(citations: list[dict[str, Any]], answer_text: str, known_numbers: set[int]) -> tuple[list[dict], list[int]]:
    """The citations the answer actually uses, and any [n] it cites that no source has.

    Unused sources are not published, so a "the sources do not cover this" answer shows no
    unrelated papers. `known_numbers` are ids earlier workers already published this turn.
    """
    used_numbers = cited_numbers(answer_text)
    used_citations = [citation for citation in citations if citation_number(citation) in used_numbers]
    own_numbers = {citation_number(citation) for citation in citations}
    unknown_numbers = sorted(used_numbers - own_numbers - known_numbers)
    return used_citations, unknown_numbers


def next_citation_index(state: dict) -> int:
    # Ids continue after the highest one published; unused ids leave gaps, never collisions.
    published = state.get("citations", [])
    if not published:
        return 1
    return max(citation_number(citation) for citation in published) + 1


def published_numbers(state: dict) -> set[int]:
    return {citation_number(citation) for citation in state.get("citations", [])}


def report_unknown_citations(agent_name: str, unknown_numbers: list[int]) -> None:
    if not unknown_numbers:
        return
    marker_list = ", ".join(f"[{number}]" for number in unknown_numbers)
    emit_trace(
        TraceEvent(
            kind="error",
            node=agent_name,
            summary=f"{agent_name} cited {marker_list}, which no source has",
            detail={"unknown_citations": unknown_numbers},
        )
    )


def publish_citations(citations: list[dict[str, Any]]) -> None:
    for citation in citations:
        emit_citation(citation)


def report_agent_result(output: AgentOutput) -> None:
    status_word = "ok" if output.ok else "failed"
    summary = f"{output.agent} {status_word}"
    if output.error:
        summary = f"{summary}: {output.error[:160]}"
    emit_trace(
        TraceEvent(
            kind="agent_result",
            node=output.agent,
            summary=summary,
            detail={
                "ok": output.ok,
                "provider": output.provider,
                "model": output.model,
                "citations": len(output.citations),
                "error": output.error,
                **output.detail,
            },
        )
    )


def task_for(state: dict, agent_name: str) -> str:
    """The supervisor's sub-question for this worker, or the whole query when there is none."""
    # A one-worker plan has no other part to split off; the planner's rewrite of the whole
    # request only loses words (a paper name, a URL) that retrieval needs.
    if len(state.get("plan") or []) <= 1:
        return state["query"]
    agent_tasks = state.get("agent_tasks") or {}
    return agent_tasks.get(agent_name) or state["query"]


def question_block(state: dict, agent_name: str) -> str:
    task_text = task_for(state, agent_name)
    if task_text == state["query"]:
        return f"Question: {task_text}"
    return f"Question: {task_text}\n(This is your part of the user's request: \"{state['query']}\". Other agents answer the rest.)"


def format_prior_outputs(prior_outputs: dict[str, dict]) -> str:
    """Render earlier workers' answers so a later worker can build on them."""
    rendered_sections: list[str] = []
    for agent_name, output in prior_outputs.items():
        if not output.get("ok"):
            continue
        rendered_sections.append(f"### Output from {agent_name} agent\n{output.get('text', '')}")
    return "\n\n".join(rendered_sections)
