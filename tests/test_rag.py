"""RAG layer tests: corpus metadata, PDF cleaning/chunking, and retrieval over the real index.

Retrieval tests need the built index (`python -m src.rag.corpus_fetch && python -m src.rag.ingest`)
and are skipped with a clear reason when it is absent.
"""

import dataclasses
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from src.config import settings
from src.rag import ingest, retriever
from src.rag.retriever import RetrievedChunk, diversify, retrieve
from src.trace_events import fallback_sink

REACT_ID = "2210.03629"
LEWIS_RAG_ID = "2005.11401"
SELF_RAG_ID = "2310.11511"
CHUNK_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}v\d+::chunk-\d{4}$")

index_present = Path(settings.chroma_persist_dir, "chroma.sqlite3").exists()
needs_index = pytest.mark.skipif(not index_present, reason="index not built: run python -m src.rag.ingest")
corpus_metadata_path = Path(settings.corpus_dir) / "metadata.json"
# metadata.json ships in the submission zip but the PDFs do not, so "fetched" means PDFs are on disk.
corpus_pdfs_present = any(Path(settings.corpus_dir).glob("*.pdf"))
corpus_fetched = corpus_metadata_path.exists() and corpus_pdfs_present
needs_corpus = pytest.mark.skipif(not corpus_fetched, reason="corpus not fetched: run python -m src.rag.corpus_fetch")


@pytest.fixture
def captured_events():
    events: list[dict] = []
    sink_token = fallback_sink.set(events.append)
    yield events
    fallback_sink.reset(sink_token)


def trace_events_of_kind(events: list[dict], kind: str) -> list[dict]:
    trace_payloads = [event["data"] for event in events if event["type"] == "trace"]
    return [payload for payload in trace_payloads if payload["kind"] == kind]


def make_chunk(source_id: str, score: float, index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source_id}v1::chunk-{index:04d}",
        text="text",
        source_id=source_id,
        title=f"Paper {source_id}",
        section="",
        page=1,
        url=f"https://arxiv.org/abs/{source_id}",
        score=score,
    )


# ---------------------------------------------------------------- corpus


@needs_corpus
def test_corpus_metadata_complete_and_pdfs_present():
    papers = json.loads(corpus_metadata_path.read_text())
    assert len(papers) >= 40
    required_fields = ["id", "title", "authors", "published", "primary_category", "categories", "abstract", "url"]
    for paper in papers:
        missing_fields = [name for name in required_fields if not paper.get(name)]
        assert not missing_fields, f"{paper.get('id')} missing {missing_fields}"
        assert paper["url"] == f"https://arxiv.org/abs/{paper['id']}"
        pdf_path = Path(settings.corpus_dir) / f"{paper['id']}.pdf"
        assert pdf_path.stat().st_size > 10_000, f"{pdf_path} missing or truncated"
    paper_ids = [paper["id"] for paper in papers]
    assert len(set(paper_ids)) == len(paper_ids)
    assert REACT_ID in paper_ids and LEWIS_RAG_ID in paper_ids


# ---------------------------------------------------------------- cleaning / chunking


def test_dehyphenation_and_paragraph_joining():
    lines = [
        "1 Introduction",
        "Large language models can con-",
        "sider tool outputs when they reason about the world and plan their next step.",
        "Short end.",
        "A new paragraph starts here and keeps going for quite a while longer.",
    ]
    document = ingest.build_clean_document([ingest.PageText(page_number=1, lines=lines)])
    assert "consider tool outputs" in document.text
    assert "con-" not in document.text
    # The heading becomes its own paragraph and is recorded as a section.
    assert document.text.startswith("1 Introduction\n\n")
    assert document.section_names == ["1 Introduction"]


def test_references_dropped_but_appendix_kept():
    body_page = ingest.PageText(page_number=1, lines=["1 Introduction", "Body text about agents."])
    reference_page = ingest.PageText(
        page_number=2,
        lines=["References", "Smith, J. and Doe, A. 2020. A survey of things.", "[2] Another cited work."],
    )
    appendix_page = ingest.PageText(page_number=3, lines=["A Additional Results", "Appendix body text."])
    document = ingest.build_clean_document([body_page, reference_page, appendix_page])
    assert "Body text about agents." in document.text
    assert "survey of things" not in document.text
    assert "Another cited work" not in document.text
    assert "Appendix body text." in document.text
    assert document.page_numbers == [1, 3]
    assert "A Additional Results" in document.section_names


def test_heading_detection_rejects_list_items_and_affiliations():
    tracker = ingest.HeadingTracker()
    assert ingest.detect_heading("1 Department of Computer Science, Tsinghua University", tracker) is None
    assert ingest.detect_heading("1 I NTRODUCTION", tracker) == "1 INTRODUCTION"
    # Out of sequence ("5" right after "1") and a different numbering style are rejected.
    assert ingest.detect_heading("5 Results", tracker) is None
    assert ingest.detect_heading("2. Increasing the money", tracker) is None
    assert ingest.detect_heading("2 Method", tracker) == "2 Method"
    assert ingest.detect_heading("2.1 Retriever Design", tracker) == "2.1 Retriever Design"
    # A table-of-contents line ends in a page number.
    assert ingest.detect_heading("3 Experiments 12", tracker) is None


@needs_corpus
def test_chunk_paper_sizes_pages_and_sections():
    papers = json.loads(corpus_metadata_path.read_text())
    react_paper = next(paper for paper in papers if paper["id"] == REACT_ID)
    pdf_path = Path(settings.corpus_dir) / react_paper["pdf_file"]
    records, page_count = ingest.chunk_paper(react_paper, pdf_path, ingest.make_splitter())
    assert len(records) > 50
    assert all(len(record.text) <= ingest.CHUNK_SIZE for record in records)
    pages = [record.metadata["page"] for record in records]
    assert pages == sorted(pages), "chunk pages must be non-decreasing in document order"
    assert 1 <= pages[0] and pages[-1] <= page_count
    sections = {record.metadata["section"] for record in records}
    assert "1 INTRODUCTION" in sections
    # The bibliography (page 10 onward until the appendix) must not be indexed.
    bibliography_hits = [record for record in records if "Imitating interactive intelligence" in record.text]
    assert not bibliography_hits


# ---------------------------------------------------------------- diversity (unit)


def test_diversify_caps_chunks_per_paper():
    candidates = [make_chunk("1111.00001", 0.9 - index * 0.01, index) for index in range(5)]
    candidates += [make_chunk("2222.00002", 0.5, 10), make_chunk("3333.00003", 0.4, 11)]
    selected = diversify(candidates, k=4)
    source_counts = Counter(chunk.source_id for chunk in selected)
    assert len(selected) == 4
    assert source_counts["1111.00001"] == 2
    assert {"2222.00002", "3333.00003"} <= set(source_counts)


def test_diversify_fills_when_too_few_papers():
    candidates = [make_chunk("1111.00001", 0.9 - index * 0.01, index) for index in range(4)]
    candidates.append(make_chunk("2222.00002", 0.1, 9))
    selected = diversify(candidates, k=4)
    assert len(selected) == 4
    scores = [chunk.score for chunk in selected]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------- retrieval (real index)


@needs_index
def test_react_query_hits_react_paper():
    chunks = retrieve("ReAct: interleaving reasoning traces and actions in language models", k=5)
    assert chunks, "expected results from the index"
    assert chunks[0].source_id == REACT_ID


@needs_index
def test_rag_query_hits_lewis_rag_paper():
    query = "retrieval-augmented generation combining parametric and non-parametric memory with a dense retriever"
    chunks = retrieve(query, k=5)
    top_three_ids = [chunk.source_id for chunk in chunks[:3]]
    assert LEWIS_RAG_ID in top_three_ids


@needs_index
def test_self_rag_query_hits_self_rag_paper():
    chunks = retrieve("reflection tokens let the model decide when to retrieve and critique its output", k=5)
    source_ids = [chunk.source_id for chunk in chunks]
    assert SELF_RAG_ID in source_ids


@needs_index
@pytest.mark.parametrize("requested_k", [1, 3, 7])
def test_top_k_respected(requested_k):
    chunks = retrieve("multi-agent collaboration between LLM agents", k=requested_k)
    assert len(chunks) == requested_k


@needs_index
def test_default_k_comes_from_settings():
    chunks = retrieve("tool use by large language models")
    assert len(chunks) == settings.retrieval_k


@needs_index
@pytest.mark.parametrize(
    "query",
    [
        "How does AutoGen enable multi-agent conversation?",
        "ReAct reasoning and acting",
        "dense passage retrieval dual encoder",
    ],
)
def test_per_paper_cap_holds(query):
    chunks = retrieve(query, k=8)
    source_counts = Counter(chunk.source_id for chunk in chunks)
    assert len(chunks) == 8
    assert max(source_counts.values()) <= retriever.MAX_CHUNKS_PER_PAPER, source_counts


@needs_index
def test_scores_bounded_and_sorted():
    chunks = retrieve("lost in the middle: position of relevant information in long contexts", k=8)
    scores = [chunk.score for chunk in chunks]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    # A relevant hit should be clearly similar, not near zero.
    assert scores[0] > 0.4


@needs_index
def test_chunks_carry_citation_fields():
    chunks = retrieve("tree of thoughts deliberate search over reasoning paths", k=6)
    for chunk in chunks:
        assert CHUNK_ID_PATTERN.match(chunk.chunk_id), chunk.chunk_id
        assert chunk.chunk_id.startswith(chunk.source_id)
        assert chunk.title.strip()
        assert chunk.url == f"https://arxiv.org/abs/{chunk.source_id}"
        assert chunk.page >= 1
        assert len(chunk.text) >= ingest.MIN_CHUNK_CHARS


@needs_index
def test_retrieval_trace_matches_results(captured_events):
    query = "Toolformer self-supervised API call insertion"
    chunks = retrieve(query, k=4)
    retrieval_traces = trace_events_of_kind(captured_events, "rag_retrieval")
    assert len(retrieval_traces) == 1
    detail = retrieval_traces[0]["detail"]
    assert detail["query"] == query
    assert detail["k"] == 4
    assert detail["chunk_ids"] == [chunk.chunk_id for chunk in chunks]
    assert detail["scores"] == [chunk.score for chunk in chunks]
    assert detail["latency_ms"] > 0
    assert not trace_events_of_kind(captured_events, "error")


# ---------------------------------------------------------------- failure paths


@pytest.fixture
def missing_index_settings(tmp_path, monkeypatch):
    missing_dir = tmp_path / "no_such_index"
    patched_settings = dataclasses.replace(settings, chroma_persist_dir=str(missing_dir))
    monkeypatch.setattr(retriever, "settings", patched_settings)
    retriever.reset_cache()
    yield missing_dir
    retriever.reset_cache()


def test_missing_index_returns_empty_and_error_trace(missing_index_settings, captured_events):
    chunks = retrieve("anything at all", k=3)
    assert chunks == []
    error_traces = trace_events_of_kind(captured_events, "error")
    assert len(error_traces) == 1
    assert "index" in error_traces[0]["detail"]["error"]
    retrieval_traces = trace_events_of_kind(captured_events, "rag_retrieval")
    assert retrieval_traces[0]["detail"]["chunk_ids"] == []
    # The lookup must not create an empty index as a side effect.
    assert not missing_index_settings.exists()


def test_missing_collection_returns_empty(tmp_path, monkeypatch, captured_events):
    empty_index_dir = tmp_path / "empty_index"
    empty_index_dir.mkdir()
    patched_settings = dataclasses.replace(settings, chroma_persist_dir=str(empty_index_dir))
    monkeypatch.setattr(retriever, "settings", patched_settings)
    retriever.reset_cache()
    try:
        chunks = retrieve("anything", k=3)
    finally:
        retriever.reset_cache()
    assert chunks == []
    assert trace_events_of_kind(captured_events, "error")


def test_empty_query_returns_empty_and_error_trace(captured_events):
    assert retrieve("   ", k=3) == []
    error_traces = trace_events_of_kind(captured_events, "error")
    assert error_traces and error_traces[0]["detail"]["error"] == "empty query"


def test_warm_up_loads_index_and_returns_true():
    import pytest
    from pathlib import Path

    from src.config import settings
    from src.rag import retriever as retriever_module

    if not Path(settings.chroma_persist_dir).exists():
        pytest.skip("chroma_index not built")
    retriever_module.reset_cache()
    assert retriever_module.warm_up() is True


def test_warm_up_returns_false_when_index_missing(monkeypatch, tmp_path):
    from src.rag import retriever as retriever_module

    def missing_collection():
        raise retriever_module.IndexUnavailableError(f"no index at {tmp_path}")

    monkeypatch.setattr(retriever_module, "get_collection", missing_collection)
    assert retriever_module.warm_up() is False


@needs_index
def test_paper_scoped_retrieval_returns_only_that_paper(captured_events):
    chunks = retrieve("key ideas and contributions", 8, source_id=SELF_RAG_ID)
    assert len(chunks) == 8
    assert {chunk.source_id for chunk in chunks} == {SELF_RAG_ID}
    # The unscoped search for the same text spreads across papers (diversity cap of 2).
    unscoped = retrieve("key ideas and contributions", 8)
    assert len({chunk.source_id for chunk in unscoped}) > 1


AUTOGEN_AGENT_TYPES_QUERY = "What are the built-in agent types in AutoGen and what role does each play in a multi-agent conversation?"


@pytest.fixture
def fresh_reranker():
    retriever.reset_reranker()
    yield
    retriever.reset_reranker()


@needs_index
def test_reranker_surfaces_the_answer_section_embeddings_miss(fresh_reranker):
    if retriever.get_reranker() is None:
        pytest.skip("reranker model unavailable")
    reranked_sections = [chunk.section for chunk in retrieve(AUTOGEN_AGENT_TYPES_QUERY, 5)]
    # Embedding order alone ranks the defining section 49th inside the AutoGen paper.
    embedding_only = retriever.diversify(
        retriever.query_collection(retriever.get_collection(), AUTOGEN_AGENT_TYPES_QUERY, 60), 5
    )
    embedding_sections = [chunk.section for chunk in embedding_only]
    assert "2.1 Conversable Agents" in reranked_sections
    assert "2.1 Conversable Agents" not in embedding_sections


@needs_index
def test_retrieval_falls_back_to_embeddings_when_reranker_fails(monkeypatch, fresh_reranker, captured_events):
    def broken_loader(model_name):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(retriever, "load_cross_encoder", broken_loader)
    chunks = retrieve("ReAct reasoning and acting", 5)
    assert len(chunks) == 5
    detail = trace_events_of_kind(captured_events, "rag_retrieval")[0]["detail"]
    assert detail["reranked"] is False
    assert detail["error"] is None


@needs_index
def test_reranker_disabled_by_empty_setting(monkeypatch, fresh_reranker):
    loader_calls = []

    def counting_loader(model_name):
        loader_calls.append(model_name)
        raise AssertionError("must not load when disabled")

    monkeypatch.setattr(retriever, "load_cross_encoder", counting_loader)
    monkeypatch.setattr(retriever, "settings", dataclasses.replace(settings, rerank_model=""))
    chunks = retrieve("ReAct reasoning and acting", 5)
    assert len(chunks) == 5
    assert loader_calls == []


@needs_index
def test_reranked_paper_scope_stays_in_paper(fresh_reranker):
    chunks = retrieve("key ideas and contributions", 8, source_id=SELF_RAG_ID)
    assert len(chunks) == 8
    assert {chunk.source_id for chunk in chunks} == {SELF_RAG_ID}
    scores = [chunk.score for chunk in chunks]
    assert scores == sorted(scores, reverse=True)


def test_reranker_is_loaded_with_the_configured_thread_count(monkeypatch, fresh_reranker):
    loader_arguments = []

    def recording_loader(model_name, threads=None):
        loader_arguments.append((model_name, threads))
        return object()

    monkeypatch.setattr(retriever, "load_cross_encoder", recording_loader)
    monkeypatch.setattr(retriever, "settings", dataclasses.replace(settings, rerank_model="m", rerank_threads=2))
    retriever.get_reranker()
    assert loader_arguments == [("m", 2)]


def test_rerank_scores_in_small_batches_to_bound_memory():
    batch_sizes_seen = []

    class RecordingReranker:
        def rerank(self, query, passages, batch_size=64):
            batch_sizes_seen.append(batch_size)
            return [0.0 for _ in passages]

    class FakeCollection:
        def query(self, **query_arguments):
            chunk_ids = [f"1111.0000{index}v1::chunk-0001" for index in range(3)]
            metadata = [{"source_id": f"1111.0000{index}", "title": "t", "section": "", "page": 1, "url": ""} for index in range(3)]
            return {"ids": [chunk_ids], "documents": [["a", "b", "c"]], "metadatas": [metadata], "distances": [[0.1, 0.2, 0.3]]}

    chunks = retriever.rerank_search(RecordingReranker(), FakeCollection(), "q", 2, None)
    assert len(chunks) == 2
    # Default batch of 64 got the 2 GiB Cloud Run instance OOM-killed.
    assert batch_sizes_seen == [retriever.RERANK_BATCH_SIZE]
    assert retriever.RERANK_BATCH_SIZE <= 16


@needs_index
def test_a_named_paper_reaches_the_rerank_pool_even_when_embeddings_miss_it(fresh_reranker):
    # Measured: "What is ColBERT late interaction?" returned no ColBERT chunk from embeddings
    # (all-MiniLM does not know the name); the alias table puts the paper into the pool.
    chunks = retriever.retrieve("What is ColBERT late interaction?", 5)
    assert chunks[0].source_id == "2004.12832"
    assert chunks[0].reranked is True


@needs_index
def test_retrieval_trace_reports_reranking_per_call(fresh_reranker, captured_events):
    retriever.retrieve("How does Reflexion work?", 5, rerank=False)
    retriever.retrieve("How does Reflexion work?", 5)
    traces = [event["data"] for event in captured_events if event["type"] == "trace" and event["data"]["kind"] == "rag_retrieval"]
    assert [trace["detail"]["reranked"] for trace in traces] == [False, True]


def test_find_aliased_papers_returns_every_named_paper_once():
    from src.rag.paper_aliases import find_aliased_papers

    assert find_aliased_papers("Compare ReAct and Reflexion with ReAct again") == ["2303.11366", "2210.03629"]
    assert find_aliased_papers("how do agents plan?") == []
