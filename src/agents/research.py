"""Research agent: grounded answers from the arXiv corpus (RAG) plus live web search (Tavily MCP).

Unique tool access: vector retrieval over the Chroma corpus + the shared web_search MCP tool.
"""

import asyncio

from src.agents.common import (
    AgentOutput,
    build_numbered_sources,
    format_prior_outputs,
    next_citation_index,
    publish_citations,
    published_numbers,
    question_block,
    report_unknown_citations,
    select_cited,
    task_for,
)
from src.config import settings
from src.llm.router import TaskProfile, complete
from src.rag.retriever import retrieve
from src.tools.web_search import web_search

AGENT_NAME = "research"

RESEARCH_SYSTEM_PROMPT = """You are the Research Agent in a multi-agent system about LLM agents, multi-agent systems and retrieval-augmented generation.
You answer using ONLY the numbered sources provided: (corpus) entries are chunks of arXiv papers, (web) entries are live web results.
Rules:
- Put a citation like [2] right after every factual claim; combine as [1][3] when several sources support it.
- Prefer corpus sources for technical claims about papers; use web sources for recent or current events.
- If the sources do not contain the answer, say so plainly in one sentence, with no citation marker, instead of guessing.
- Never invent citation numbers that are not in the source list.
- Answer only your own question. When it is one part of a larger request, never comment on the other parts: other agents answer them.
- Be concise and well structured: short paragraphs or bullet points, markdown allowed."""

# Above this much retrieved context the request counts as long-context synthesis. Five corpus
# chunks are ~5.5k chars, so plain RAG stays "simple"; web results or long chunks push it over.
COMPLEX_CONTEXT_CHARS = 9000

# Cross-encoder relevance (0-1) below which a corpus chunk does not answer the question. Measured:
# on-topic questions have a best chunk of 0.45-1.0; "TCP vs UDP" or "why is the sky blue" score
# 0.000 on every chunk, and "what did OpenAI release last month" peaks at 0.036.
MIN_RELEVANT_SCORE = 0.05


def keep_relevant(chunks: list) -> list:
    # Cosine scores (re-ranker unavailable) are not comparable to this threshold; keep them all.
    if not chunks or not all(chunk.reranked for chunk in chunks):
        return chunks
    return [chunk for chunk in chunks if chunk.score >= MIN_RELEVANT_SCORE]


async def gather_sources(query: str, needs_web: bool) -> tuple[list, list, dict]:
    """Corpus chunks and web results for one question, plus why the web was or was not used."""
    corpus_task = asyncio.to_thread(retrieve, query, settings.retrieval_k)
    if needs_web:
        web_task = web_search(query, caller=AGENT_NAME, max_results=4)
        chunks, web_result = await asyncio.gather(corpus_task, web_task)
        web_detail = {"web_reason": "supervisor set needs_web", "web_error": web_result.error}
        return keep_relevant(chunks), web_result.results, web_detail

    chunks = await corpus_task
    relevant_chunks = keep_relevant(chunks)
    relevant_papers = {chunk.source_id for chunk in relevant_chunks}
    if len(relevant_papers) == 1 and len(relevant_chunks) < settings.retrieval_k:
        # Only one paper answers (live: "What is ColBERT late interaction?" kept 2 ColBERT chunks
        # and 3 irrelevant ones); the per-paper cap left too little of it, so read more of it.
        paper_id = next(iter(relevant_papers))
        paper_chunks = await asyncio.to_thread(retrieve, query, settings.retrieval_k, paper_id)
        relevant_chunks = keep_relevant(paper_chunks) or relevant_chunks
    if relevant_chunks:
        return relevant_chunks, [], {"web_reason": None, "web_error": None}
    # Nothing in the corpus answers this (an off-topic but reasonable question): ask the web
    # rather than reply "the sources do not say".
    web_result = await web_search(query, caller=AGENT_NAME, max_results=4)
    web_detail = {"web_reason": "no relevant corpus chunk", "web_error": web_result.error}
    if web_result.results:
        return [], web_result.results, web_detail
    return chunks, [], web_detail


async def run_research(state: dict, is_final: bool) -> AgentOutput:
    # Retrieval uses this worker's own sub-question, not the whole multi-part request.
    task_text = task_for(state, AGENT_NAME)
    needs_web = bool(state.get("needs_web"))
    chunks, web_results, web_detail = await gather_sources(task_text, needs_web)

    if not chunks and not web_results:
        return AgentOutput(
            agent=AGENT_NAME,
            ok=False,
            error="no sources: corpus retrieval and web search both returned nothing",
            detail=web_detail,
        )

    start_index = next_citation_index(state)
    citations, context_block = build_numbered_sources(chunks, web_results, start_index)

    prior_context = format_prior_outputs(state.get("agent_outputs", {}))
    user_prompt = f"{question_block(state, AGENT_NAME)}\n\nSources:\n{context_block}"
    if prior_context:
        user_prompt = f"{user_prompt}\n\nEarlier agent findings (may cite their own sources):\n{prior_context}"

    # Needing the web is a tool choice, not a capability need: only context size decides.
    complexity = "complex" if len(context_block) > COMPLEX_CONTEXT_CHARS else "simple"
    profile = TaskProfile(complexity=complexity, context_chars=len(user_prompt))
    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    llm_result = await complete("research", messages, profile, stream_tokens=is_final, node=AGENT_NAME)

    # Citations reach the UI only for an answer that exists, so a failed worker leaves no orphans,
    # and only the ones the answer cites, so unrelated retrieved papers are not listed.
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
            "corpus_chunks": len(chunks),
            "web_results": len(web_results),
            **web_detail,
            "llm_latency_ms": llm_result.latency_ms,
            "ttft_ms": llm_result.ttft_ms,
            "input_tokens": llm_result.input_tokens,
            "output_tokens": llm_result.output_tokens,
        },
    )
