"""Summary agent: condenses papers, web pages or earlier agents' findings while preserving citations.

Unique tool access: page extraction through the shared Tavily MCP server (web_extract), web search
for named sources, and paper-scoped corpus retrieval.
"""

import asyncio
import re

from src.agents.common import (
    AgentOutput,
    build_numbered_sources,
    format_prior_outputs,
    next_citation_index,
    publish_citations,
    published_numbers,
    report_unknown_citations,
    select_cited,
    task_for,
)
from src.llm.router import TaskProfile, complete
from src.rag.paper_aliases import find_aliased_paper
from src.rag.retriever import retrieve
from src.tools.web_search import web_extract, web_search

AGENT_NAME = "summary"

SUMMARY_SYSTEM_PROMPT = """You are the Summary Agent. You write faithful, compact summaries.
Rules:
- Summarize ONLY what the provided material says. Do not add outside knowledge.
- Preserve citations: every sentence that carries a fact ends with the [n] of the source it came from. Keep citation numbers exactly as given; never renumber or invent them.
- When summarizing earlier agents' findings, keep their [n] markers attached to the claims they support.
- Structure: a one-sentence TL;DR, then 3-6 bullet key points, then (if relevant) limitations or open questions.
- If the material is empty or irrelevant to the request, say what is missing."""

URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

# Prose punctuation after a URL ("see https://x.org/page, then...") is not part of the URL.
URL_TRAILING_PUNCTUATION = ".,;:!?'\""


# A page fetched for "summarize this URL" is the whole point of the request, so far more of it
# reaches the prompt than of a search snippet (web_extract itself keeps up to 8,000 chars).
EXTRACTED_PAGE_CHARS = 8000

# Summaries of more than this much material are treated as complex condensation.
COMPLEX_CONTEXT_CHARS = 8000

# A paper's short name ("Self-RAG" in "Self-RAG: Learning to ...") must be at least this long
# to count as named in the request; shorter names match too much ordinary text.
MIN_PAPER_NAME_CHARS = 4


def extract_urls(text: str) -> list[str]:
    found_urls = URL_PATTERN.findall(text)
    cleaned_urls = [url.rstrip(URL_TRAILING_PUNCTUATION) for url in found_urls]
    return cleaned_urls[:3]



def paper_short_name(title: str) -> str:
    return title.split(":", 1)[0].strip()


def find_named_paper(query: str, chunks: list) -> str | None:
    """The source_id of a retrieved paper whose title the request names, if any.

    "Summarize Self-RAG" should summarize the Self-RAG paper itself, not the top chunks of every
    paper that mentions it, which is what plain diversified retrieval returns.
    """
    lowered_query = query.lower()
    for chunk in chunks:
        short_name = paper_short_name(chunk.title)
        full_title = chunk.title.strip().lower()
        names_short_title = len(short_name) >= MIN_PAPER_NAME_CHARS and short_name.lower() in lowered_query
        names_full_title = bool(full_title) and full_title in lowered_query
        if names_short_title or names_full_title:
            return chunk.source_id
    return None


def retrieve_for_summary(query: str, k: int) -> tuple[list, str | None]:
    named_source_id = find_aliased_paper(query)
    candidate_chunks: list = []
    if named_source_id is None:
        # Embedding order is enough to spot a named paper; re-ranking is spent on the final set only.
        candidate_chunks = retrieve(query, k, rerank=False)
        named_source_id = find_named_paper(query, candidate_chunks)
    if named_source_id is None:
        return retrieve(query, k), None
    paper_chunks = retrieve(query, k, source_id=named_source_id)
    if paper_chunks:
        return paper_chunks, named_source_id
    # Keep topic results if the scoped search came back empty for any reason.
    return (candidate_chunks or retrieve(query, k)), None


async def gather_material(
    query: str, has_prior_outputs: bool, needs_web: bool, urls: list[str]
) -> tuple[list, list, dict]:
    """Pick sources by what the request points at: explicit URLs, prior findings, or a paper/topic."""
    if urls:
        extracted = await web_extract(urls, caller=AGENT_NAME)
        return [], extracted.results, {"mode": "url_extract", "urls": urls, "web_error": extracted.error}

    if has_prior_outputs:
        # Earlier agents already gathered and cited sources; summarize their output.
        return [], [], {"mode": "prior_outputs"}

    # Paper/topic summary: pull more chunks than a QA lookup so the summary covers the paper.
    corpus_task = asyncio.to_thread(retrieve_for_summary, query, 8)
    if not needs_web:
        chunks, named_source_id = await corpus_task
        return chunks, [], {"mode": "corpus", "paper": named_source_id}
    web_task = web_search(query, caller=AGENT_NAME, max_results=3)
    (chunks, named_source_id), web_result = await asyncio.gather(corpus_task, web_task)
    return chunks, web_result.results, {"mode": "corpus+web", "paper": named_source_id, "web_error": web_result.error}


async def run_summary(state: dict, is_final: bool) -> AgentOutput:
    query = task_for(state, AGENT_NAME)
    # URLs come from the user's own words first: the planner's rewrite can drop or mangle them.
    urls = extract_urls(state["query"]) or extract_urls(query)
    prior_outputs = state.get("agent_outputs", {})
    prior_context = format_prior_outputs(prior_outputs)
    # With no earlier findings to condense, a sub-task like "condense the research findings"
    # is not a search query; retrieve on what the user actually asked.
    retrieval_query = query if prior_context else state["query"]
    chunks, web_results, gather_detail = await gather_material(
        retrieval_query, bool(prior_context), bool(state.get("needs_web")), urls
    )

    start_index = next_citation_index(state)
    web_chars = EXTRACTED_PAGE_CHARS if gather_detail.get("mode") == "url_extract" else None
    citations, context_block = build_numbered_sources(chunks, web_results, start_index, web_chars)

    material_parts: list[str] = []
    if context_block:
        material_parts.append(f"Sources:\n{context_block}")
    if prior_context:
        material_parts.append(f"Earlier agent findings to summarize:\n{prior_context}")
    if not material_parts:
        return AgentOutput(
            agent=AGENT_NAME, ok=False, error="nothing to summarize: no sources found", detail=gather_detail
        )

    material = "\n\n".join(material_parts)
    user_prompt = f"Request: {query}\n\n{material}"
    complexity = "complex" if len(material) > COMPLEX_CONTEXT_CHARS else "simple"
    profile = TaskProfile(complexity=complexity, context_chars=len(user_prompt))
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    llm_result = await complete("summary", messages, profile, stream_tokens=is_final, node=AGENT_NAME)

    # Citations reach the UI only for an answer that exists, so a failed worker leaves no orphans,
    # and only the ones the summary cites. A cited number no source has is reported in the trace.
    used_citations: list = []
    if llm_result.error is None:
        used_citations, unknown_numbers = select_cited(citations, llm_result.text, published_numbers(state))
        publish_citations(used_citations)
        report_unknown_citations(AGENT_NAME, unknown_numbers)

    return AgentOutput(
        agent=AGENT_NAME,
        ok=llm_result.error is None,
        text=llm_result.text,
        citations=used_citations,
        provider=llm_result.provider,
        model=llm_result.model,
        error=llm_result.error,
        detail={
            **gather_detail,
            "corpus_chunks": len(chunks),
            "web_results": len(web_results),
            "llm_latency_ms": llm_result.latency_ms,
            "ttft_ms": llm_result.ttft_ms,
            "input_tokens": llm_result.input_tokens,
            "output_tokens": llm_result.output_tokens,
        },
    )
