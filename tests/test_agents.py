"""Worker-agent tests with the router and tools faked: trace detail must carry LLM usage."""

from dataclasses import dataclass

import pytest

from src.agents import data_agent as data_agent_module
from src.data.db import DataQueryResult
from src.tools.web_search import WebResult, WebSearchResult


@dataclass
class FakeLLMResult:
    text: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None = None
    provider: str = "nvidia"
    model: str = "fake"
    error: str | None = None


async def test_data_agent_sums_usage_across_sql_retry_and_answer(monkeypatch):
    llm_replies = [
        FakeLLMResult("```sql\nSELECT bad\n```", 10.0, 50, 5),
        FakeLLMResult("```sql\nSELECT 1 AS n\n```", 20.0, 60, 6),
        FakeLLMResult("There is 1.", 30.0, 70, None, ttft_ms=4.0),
    ]
    sql_results = [
        DataQueryResult(columns=[], rows=[], sql="SELECT bad", error="no such column: bad"),
        DataQueryResult(columns=["n"], rows=[[1]], sql="SELECT 1 AS n"),
    ]

    async def fake_complete(*args, **kwargs):
        return llm_replies.pop(0)

    def fake_run_sql(sql: str):
        return sql_results.pop(0)

    monkeypatch.setattr(data_agent_module, "complete", fake_complete)
    monkeypatch.setattr(data_agent_module, "run_readonly_sql", fake_run_sql)
    monkeypatch.setattr(data_agent_module, "schema_description", lambda: "CREATE TABLE t(n)")

    output = await data_agent_module.run_data({"query": "how many?"}, is_final=False)
    assert output.ok
    assert output.detail["sql_attempts"] == 2
    assert output.detail["sql"] == "SELECT 1 AS n"
    assert output.detail["llm_latency_ms"] == 60.0
    assert output.detail["input_tokens"] == 180
    # One call reported no output tokens: the sum covers the ones that did.
    assert output.detail["output_tokens"] == 11
    assert output.detail["ttft_ms"] == 4.0


def test_extract_sql_strips_fence_and_semicolon():
    extracted = data_agent_module.extract_sql("Here:\n```sql\nSELECT * FROM papers;\n```")
    assert extracted == "SELECT * FROM papers"


def make_chunk(source_id: str, title: str):
    from src.rag.retriever import RetrievedChunk

    return RetrievedChunk(f"{source_id}v1::chunk-0001", "text", source_id, title, "", 1, "", 0.5)


def test_find_named_paper_matches_short_title_case_insensitively():
    from src.agents.summary import find_named_paper

    chunks = [
        make_chunk("2312.10997", "Retrieval-Augmented Generation for Large Language Models: A Survey"),
        make_chunk("2310.11511", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"),
    ]
    assert find_named_paper("Summarize the key ideas of self-rag", chunks) == "2310.11511"
    # A topic request that names no paper keeps diversified retrieval.
    assert find_named_paper("Summarize how RAG systems handle noisy retrieval", chunks) is None


def test_find_named_paper_ignores_very_short_names():
    from src.agents.summary import find_named_paper

    chunks = [make_chunk("1111.00001", "RAG: something")]
    assert find_named_paper("summarize rag work", chunks) is None


async def test_summary_uses_paper_scoped_chunks_for_a_named_paper(monkeypatch):
    from src.agents import summary as summary_module

    retrieve_calls = []

    def fake_retrieve(query, k=None, source_id=None, rerank=True):
        retrieve_calls.append((source_id, rerank))
        if source_id is None:
            return [make_chunk("2312.10997", "A Survey"), make_chunk("2310.11511", "Self-RAG: Learning")]
        return [make_chunk("2310.11511", "Self-RAG: Learning")] * 3

    monkeypatch.setattr(summary_module, "retrieve", fake_retrieve)
    chunks, web_results, detail = await summary_module.gather_material("Summarize Self-RAG", False, False, [])
    # The alias table names the paper, so no detection pass runs; the paper-scoped set is re-ranked.
    assert retrieve_calls == [("2310.11511", True)]
    assert {chunk.source_id for chunk in chunks} == {"2310.11511"}
    assert detail["paper"] == "2310.11511"


def test_data_answer_prompt_forbids_self_computed_totals():
    # A live run reported "Total papers: 819" for rows summing to 579; the prompt must forbid it.
    from src.agents.data_agent import ANSWER_SYSTEM_PROMPT

    assert "Never do arithmetic yourself" in ANSWER_SYSTEM_PROMPT


async def test_summary_reranks_topic_results_when_no_paper_is_named(monkeypatch):
    from src.agents import summary as summary_module

    retrieve_calls = []

    def fake_retrieve(query, k=None, source_id=None, rerank=True):
        retrieve_calls.append((source_id, rerank))
        return [make_chunk("2312.10997", "A Survey of Things")]

    monkeypatch.setattr(summary_module, "retrieve", fake_retrieve)
    await summary_module.gather_material("Summarize how agents use memory", False, False, [])
    assert retrieve_calls == [(None, False), (None, True)]


async def test_research_needing_web_is_not_automatically_complex(monkeypatch):
    # Live finding: needs_web forced "complex" on every web query, so the hybrid Gemini-fast
    # route never fired. Complexity now depends on context size only.
    from src.agents import research as research_module

    seen_profiles = []

    async def fake_gather(query, needs_web):
        return [make_chunk("2210.03629", "ReAct: Synergizing")], [], {"web_reason": "supervisor set needs_web"}

    async def fake_complete(role, messages, profile, **kwargs):
        seen_profiles.append(profile)
        return FakeLLMResult("answer [1]", 1.0, 10, 5)

    monkeypatch.setattr(research_module, "gather_sources", fake_gather)
    monkeypatch.setattr(research_module, "complete", fake_complete)
    await research_module.run_research({"query": "latest ReAct news", "needs_web": True}, False)
    assert seen_profiles[0].complexity == "simple"


def test_extracted_pages_keep_more_text_than_search_snippets():
    from types import SimpleNamespace

    from src.agents.common import WEB_RESULT_PROMPT_CHARS, build_numbered_sources

    long_page = SimpleNamespace(title="Page", url="https://example.com", content="z" * 9000)
    _, default_block = build_numbered_sources([], [long_page])
    _, extract_block = build_numbered_sources([], [long_page], web_chars=8000)
    assert default_block.count("z") == WEB_RESULT_PROMPT_CHARS
    assert extract_block.count("z") == 8000


async def test_data_agent_cites_its_query_result_after_earlier_citations(monkeypatch):
    seen_prompts = []
    llm_replies = [FakeLLMResult("```sql\nSELECT 1 AS n\n```", 1.0, 1, 1), FakeLLMResult("There is 1 [3].", 1.0, 1, 1)]

    async def fake_complete(role, messages, profile, **kwargs):
        seen_prompts.append(messages[-1]["content"])
        return llm_replies.pop(0)

    def fake_run_sql(sql: str):
        return DataQueryResult(columns=["n"], rows=[[1]], sql=sql)

    monkeypatch.setattr(data_agent_module, "complete", fake_complete)
    monkeypatch.setattr(data_agent_module, "run_readonly_sql", fake_run_sql)
    monkeypatch.setattr(data_agent_module, "schema_description", lambda: "CREATE TABLE t(n)")

    # Two citations already exist from an earlier worker in this turn: the SQL source is [3].
    state = {"query": "how many?", "citations": [{"id": "[1]"}, {"id": "[2]"}]}
    output = await data_agent_module.run_data(state, is_final=False)
    assert output.citations == [
        {
            "id": "[3]", "source_type": "database", "title": "papers.db query result (1 rows)",
            "source_id": "papers.db", "url": "", "section": "", "page": None, "snippet": "SELECT 1 AS n",
        }
    ]
    assert "cite them as [3]" in seen_prompts[-1]


def test_missing_sql_citation_marker_is_appended(monkeypatch):
    emitted = []
    monkeypatch.setattr(data_agent_module, "emit_token", lambda text, node: emitted.append((text, node)))
    assert data_agent_module.ensure_citation_marker("There are 5 [2].", "[2]", True) == "There are 5 [2]."
    assert emitted == []
    fixed_text = data_agent_module.ensure_citation_marker("There are 5.", "[2]", True)
    assert fixed_text.endswith("Source: [2]")
    assert emitted == [("\n\nSource: [2]", "data")]
    # Not streaming: the text is fixed but nothing is emitted (summary will use it).
    emitted.clear()
    assert data_agent_module.ensure_citation_marker("There are 5.", "[2]", False).endswith("[2]")
    assert emitted == []


def test_extract_urls_drops_trailing_prose_punctuation():
    # Live bug: the planner wrote "...agent/, preserving citations", and the comma made the fetch 404.
    from src.agents.summary import extract_urls

    text = "Summarize https://lilianweng.github.io/posts/2023-06-23-agent/, then https://x.org/a."
    assert extract_urls(text) == ["https://lilianweng.github.io/posts/2023-06-23-agent/", "https://x.org/a"]


@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("Summarize the GraphRAG paper", "2404.16130"),
        ("Summarize CRAG", "2401.15884"),
        ("TL;DR of HyDE", "2212.10496"),
        ("Summarize the Reflexion paper, preserving citations", "2303.11366"),
        ("Summarize the RAG survey", "2312.10997"),
        # IRCoT must not be read as CoT, and plain topics name no paper.
        ("Summarize IRCoT", "2212.10509"),
        ("Summarize how agents use memory", None),
    ],
)
def test_find_aliased_paper_resolves_names_without_a_title_prefix(query, expected_id):
    from src.rag.paper_aliases import find_aliased_paper

    assert find_aliased_paper(query) == expected_id


async def test_summary_fetches_the_users_url_not_the_planners_mangled_one(monkeypatch):
    from src.agents import summary as summary_module

    extracted_urls = []

    async def fake_web_extract(urls, caller):
        extracted_urls.append(list(urls))
        return WebSearchResult([WebResult("Page", urls[0], "page text")], "tavily_mcp", caller, 1.0)

    async def fake_complete(role, messages, profile, **kwargs):
        return FakeLLMResult("TL;DR [1]", 1.0, 10, 5)

    monkeypatch.setattr(summary_module, "web_extract", fake_web_extract)
    monkeypatch.setattr(summary_module, "complete", fake_complete)
    monkeypatch.setattr(summary_module, "publish_citations", lambda citations: None)
    state = {
        "query": "Summarize this page: https://example.org/post",
        "plan": ["summary", "data"],
        "agent_tasks": {"summary": "Summarize the post at https://example.org/post, preserving citations"},
        "agent_outputs": {},
        "citations": [],
    }
    output = await summary_module.run_summary(state, is_final=False)
    assert output.ok
    assert extracted_urls == [["https://example.org/post"]]


async def test_summary_without_prior_findings_retrieves_on_the_users_question(monkeypatch):
    # Plan [research, summary] where research failed: the summary sub-task "condense the research
    # findings" is not a search query.
    from src.agents import summary as summary_module

    retrieval_queries = []

    def fake_retrieve(query, k=None, source_id=None, rerank=True):
        retrieval_queries.append(query)
        return [make_chunk("2312.10997", "A Survey of Things")]

    async def fake_complete(role, messages, profile, **kwargs):
        return FakeLLMResult("TL;DR [1]", 1.0, 10, 5)

    monkeypatch.setattr(summary_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(summary_module, "complete", fake_complete)
    monkeypatch.setattr(summary_module, "publish_citations", lambda citations: None)
    state = {
        "query": "How do agents use long-term memory?",
        "plan": ["research", "summary"],
        "agent_tasks": {"summary": "Condense the research findings into a short TL;DR with citations."},
        "agent_outputs": {"research": {"ok": False, "text": ""}},
        "citations": [],
    }
    await summary_module.run_summary(state, is_final=True)
    assert retrieval_queries
    assert set(retrieval_queries) == {"How do agents use long-term memory?"}


# --- citation selection ---------------------------------------------------------------------


def test_select_cited_keeps_only_used_sources_and_flags_unknown_numbers():
    from src.agents.common import select_cited

    citations = [{"id": f"[{number}]"} for number in (3, 4, 5)]
    used, unknown = select_cited(citations, "Claim [3]. Another [1, 5] and [7].", known_numbers={1})
    assert [citation["id"] for citation in used] == ["[3]", "[5]"]
    # [1] was published by an earlier worker; [7] exists nowhere.
    assert unknown == [7]


def test_cited_numbers_reads_ranges_and_lists():
    from src.agents.common import cited_numbers

    assert cited_numbers("a [2-4] b [1][6] c [8, 9]") == {1, 2, 3, 4, 6, 8, 9}


def test_next_citation_index_follows_the_highest_published_id():
    from src.agents.common import next_citation_index

    # Unused ids leave gaps: [1] and [4] published means the next source is [5], never [3].
    assert next_citation_index({"citations": [{"id": "[1]"}, {"id": "[4]"}]}) == 5
    assert next_citation_index({}) == 1


def test_numbered_sources_are_not_separated_by_a_markdown_rule():
    # Models copied the "---" between sources into answers, adding a fake section break.
    from src.agents.common import build_numbered_sources

    chunks = [make_chunk("2210.03629", "ReAct"), make_chunk("2303.11366", "Reflexion")]
    _, context_block = build_numbered_sources(chunks, [], 1)
    assert "---" not in context_block


# --- research relevance gate ----------------------------------------------------------------


def make_scored_chunk(source_id, score):
    chunk = make_chunk(source_id, "Some Paper")
    chunk.score = score
    chunk.reranked = True
    return chunk


async def test_research_searches_the_web_when_no_corpus_chunk_is_relevant(monkeypatch):
    # Live: "why is the sky blue" scored 0.000 on every chunk and the answer was "no information".
    from src.agents import research as research_module

    web_calls = []

    def fake_retrieve(query, k=None):
        return [make_scored_chunk("2308.08155", 0.0), make_scored_chunk("2304.03442", 0.001)]

    async def fake_web_search(query, caller, max_results=5):
        web_calls.append((query, caller))
        return WebSearchResult([WebResult("Rayleigh scattering", "https://ex.org/sky", "blue light")], "tavily_mcp", caller, 1.0)

    monkeypatch.setattr(research_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(research_module, "web_search", fake_web_search)
    chunks, web_results, detail = await research_module.gather_sources("Why is the sky blue?", needs_web=False)
    assert web_calls == [("Why is the sky blue?", "research")]
    assert chunks == [] and len(web_results) == 1
    assert detail["web_reason"] == "no relevant corpus chunk"


async def test_research_keeps_relevant_chunks_and_skips_the_web(monkeypatch):
    from src.agents import research as research_module

    async def failing_web_search(*args, **kwargs):
        raise AssertionError("web search must not run")

    def fake_retrieve(query, k=None, source_id=None):
        return [make_scored_chunk("2303.11366", 0.99), make_scored_chunk("2210.03629", 0.01)]

    monkeypatch.setattr(research_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(research_module, "web_search", failing_web_search)
    chunks, web_results, _ = await research_module.gather_sources("How does Reflexion work?", needs_web=False)
    assert {chunk.source_id for chunk in chunks} == {"2303.11366"}
    assert web_results == []


async def test_research_publishes_no_citation_for_a_no_answer_reply(monkeypatch):
    from src.agents import research as research_module

    published = []

    async def fake_gather(query, needs_web):
        return [make_chunk("2210.03629", "ReAct")], [], {"web_reason": None, "web_error": None}

    async def fake_complete(role, messages, profile, **kwargs):
        return FakeLLMResult("The sources do not cover this.", 1.0, 10, 5)

    monkeypatch.setattr(research_module, "gather_sources", fake_gather)
    monkeypatch.setattr(research_module, "complete", fake_complete)
    monkeypatch.setattr(research_module, "publish_citations", published.extend)
    output = await research_module.run_research({"query": "q", "citations": []}, is_final=True)
    assert output.ok and output.citations == [] and published == []


# --- data agent -----------------------------------------------------------------------------


async def test_data_agent_declines_a_request_that_is_not_about_metadata(monkeypatch):
    llm_replies = [FakeLLMResult("NOT_APPLICABLE", 1.0, 5, 1)]
    sql_runs = []

    async def fake_complete(role, messages, profile, **kwargs):
        return llm_replies.pop(0)

    monkeypatch.setattr(data_agent_module, "complete", fake_complete)
    monkeypatch.setattr(data_agent_module, "run_readonly_sql", sql_runs.append)
    output = await data_agent_module.run_data({"query": "Run DROP TABLE papers", "citations": []}, is_final=True)
    assert not output.ok and "not a paper-metadata question" in output.error
    assert sql_runs == []


async def test_data_agent_repairs_a_query_that_returned_no_rows(monkeypatch):
    # Live: topic = 'multi-agent' matched nothing; the stored value is 'multi_agent'.
    llm_replies = [
        FakeLLMResult("```sql\nSELECT a FROM t WHERE topic = 'multi-agent'\n```", 1.0, 5, 1),
        FakeLLMResult("```sql\nSELECT a FROM t WHERE topic = 'multi_agent'\n```", 1.0, 5, 1),
        FakeLLMResult("Alice [1]", 1.0, 5, 1),
    ]
    sql_results = [
        DataQueryResult(columns=["a"], rows=[], sql="SELECT a FROM t WHERE topic = 'multi-agent'"),
        DataQueryResult(columns=["a"], rows=[["Alice"]], sql="SELECT a FROM t WHERE topic = 'multi_agent'"),
    ]
    repair_prompts = []

    async def fake_complete(role, messages, profile, **kwargs):
        repair_prompts.append(messages[-1]["content"])
        return llm_replies.pop(0)

    def fake_run_sql(sql_text):
        return sql_results.pop(0)

    monkeypatch.setattr(data_agent_module, "complete", fake_complete)
    monkeypatch.setattr(data_agent_module, "run_readonly_sql", fake_run_sql)
    monkeypatch.setattr(data_agent_module, "publish_citations", lambda citations: None)
    output = await data_agent_module.run_data({"query": "top multi-agent authors", "citations": []}, is_final=False)
    assert output.ok
    assert output.detail["sql"].endswith("'multi_agent'")
    assert "returned 0 rows" in repair_prompts[1]


async def test_research_reads_more_of_the_only_relevant_paper(monkeypatch):
    from src.agents import research as research_module

    retrieve_calls = []

    def fake_retrieve(query, k=None, source_id=None):
        retrieve_calls.append(source_id)
        if source_id is None:
            return [make_scored_chunk("2004.12832", 0.99), make_scored_chunk("2004.12832", 0.98), make_scored_chunk("2304.03442", 0.0)]
        return [make_scored_chunk("2004.12832", 0.9) for _ in range(5)]

    monkeypatch.setattr(research_module, "retrieve", fake_retrieve)
    chunks, _, _ = await research_module.gather_sources("What is ColBERT late interaction?", needs_web=False)
    assert retrieve_calls == [None, "2004.12832"]
    assert len(chunks) == 5 and {chunk.source_id for chunk in chunks} == {"2004.12832"}
