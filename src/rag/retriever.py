"""Semantic retrieval over the arXiv chunk index built by `src.rag.ingest`.

`retrieve()` never raises for an expected failure: a missing index or a broken query returns []
and emits an `error` trace. Every call also emits exactly one `rag_retrieval` trace.
"""

import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import settings
from src.rag.paper_aliases import find_aliased_papers
from src.trace_events import TraceEvent, emit_trace

logger = logging.getLogger(__name__)

TRACE_NODE = "retriever"
# At most this many chunks per paper in the top-k, unless too few papers match to fill k.
MAX_CHUNKS_PER_PAPER = 2
# Over-fetch so the per-paper cap still leaves k candidates to choose from.
CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATES = 60

# Re-ranking pool. MiniLM embeddings find the right paper but often the wrong section of it
# (AutoGen's agent-type definitions ranked 49th inside its own paper), so the pool adds the
# best chunks *within* the top papers to the global top hits, and a cross-encoder, which reads
# query and passage together, orders them. Sizes were picked on the eval set: a bigger pool
# scored slightly higher but cost ~3x the CPU time.
RERANK_GLOBAL_CANDIDATES = 50
RERANK_TOP_PAPERS = 3
RERANK_CHUNKS_PER_PAPER = 10
# Passages scored per ONNX run. The default of 64 raised peak memory by ~1.45 GB and got the
# 2 GiB Cloud Run instance OOM-killed; 8 adds ~0.3 GB and is also slightly faster.
RERANK_BATCH_SIZE = 8


@dataclass
class RetrievedChunk:
    chunk_id: str  # e.g. "2308.08155v2::chunk-0042"
    text: str
    source_id: str  # arXiv id, e.g. "2308.08155"
    title: str
    section: str  # best-effort heading, "" if unknown
    page: int  # 1-based page of the chunk start
    url: str  # https://arxiv.org/abs/<id>
    score: float  # higher = more relevant (cosine similarity, clamped to [0, 1])
    # True when `score` is the cross-encoder relevance, which callers can threshold on.
    reranked: bool = False


_cached_reranker = None
_reranker_failed = False
_reranker_lock = threading.Lock()


def load_cross_encoder(model_name: str, threads: int | None = None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name, threads=threads)


def get_reranker():
    """The cross-encoder, loaded once. None when disabled or when it failed to load."""
    global _cached_reranker, _reranker_failed
    if not settings.rerank_model:
        return None
    with _reranker_lock:
        if _cached_reranker is None and not _reranker_failed:
            try:
                _cached_reranker = load_cross_encoder(settings.rerank_model, settings.rerank_threads)
            except Exception:
                # Retrieval still works on embedding order; the trace reports reranked=False.
                logger.exception("reranker %s failed to load; using embedding order", settings.rerank_model)
                _reranker_failed = True
        return _cached_reranker


def reset_reranker() -> None:
    global _cached_reranker, _reranker_failed
    with _reranker_lock:
        _cached_reranker = None
        _reranker_failed = False


def logit_to_unit(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


class IndexUnavailableError(Exception):
    """The Chroma directory or collection does not exist (run `python -m src.rag.ingest`)."""


# Lazily opened collection, keyed by (persist_dir, collection_name) so a settings change in
# tests or scripts transparently opens the right index.
_cached_collection: chromadb.Collection | None = None
_cached_key: tuple[str, str] | None = None
_cache_lock = threading.Lock()


def reset_cache() -> None:
    global _cached_collection, _cached_key
    with _cache_lock:
        _cached_collection = None
        _cached_key = None


def open_collection(persist_dir: str, collection_name: str) -> chromadb.Collection:
    # PersistentClient would silently create an empty index at a missing path, so check first.
    if not Path(persist_dir).is_dir():
        raise IndexUnavailableError(f"index directory not found: {persist_dir}")
    client_settings = ChromaSettings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=persist_dir, settings=client_settings)
    try:
        return client.get_collection(collection_name, embedding_function=DefaultEmbeddingFunction())
    except Exception as lookup_error:
        raise IndexUnavailableError(f"collection {collection_name!r} not found in {persist_dir}") from lookup_error


def get_collection() -> chromadb.Collection:
    global _cached_collection, _cached_key
    requested_key = (settings.chroma_persist_dir, settings.chroma_collection)
    with _cache_lock:
        if _cached_collection is None or _cached_key != requested_key:
            # Failures are not cached, so a later ingest is picked up without a restart.
            _cached_collection = open_collection(*requested_key)
            _cached_key = requested_key
        return _cached_collection


def distance_to_similarity(cosine_distance: float) -> float:
    # Chroma's cosine distance is 1 - cos_sim, in [0, 2]; clamp similarity into [0, 1].
    similarity = 1.0 - cosine_distance
    return max(0.0, min(1.0, similarity))


def to_chunks(query_result: dict) -> list[RetrievedChunk]:
    # Chroma returns one list per query text; we always send exactly one query.
    ids = query_result["ids"][0]
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
    chunks: list[RetrievedChunk] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document or "",
                source_id=str(metadata.get("source_id", "")),
                title=str(metadata.get("title", "")),
                section=str(metadata.get("section", "")),
                page=int(metadata.get("page", 0)),
                url=str(metadata.get("url", "")),
                score=round(distance_to_similarity(distance), 4),
            )
        )
    return chunks


def sort_key_by_score(chunk: RetrievedChunk) -> float:
    return chunk.score


def diversify(candidates: list[RetrievedChunk], k: int, max_per_paper: int = MAX_CHUNKS_PER_PAPER) -> list[RetrievedChunk]:
    """Pick the top-k by score with at most `max_per_paper` chunks from any one paper.

    If the cap leaves fewer than k chunks (too few distinct papers matched), the best
    remaining chunks fill the gap. The result is sorted by score, highest first.
    """
    ranked = sorted(candidates, key=sort_key_by_score, reverse=True)
    selected: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    per_paper_counts: Counter[str] = Counter()
    for chunk in ranked:
        if len(selected) >= k:
            break
        if per_paper_counts[chunk.source_id] < max_per_paper:
            selected.append(chunk)
            per_paper_counts[chunk.source_id] += 1
        else:
            overflow.append(chunk)
    missing_count = k - len(selected)
    if missing_count > 0:
        selected.extend(overflow[:missing_count])
    return sorted(selected, key=sort_key_by_score, reverse=True)


def emit_error(query: str, message: str) -> None:
    emit_trace(
        TraceEvent(
            kind="error",
            node=TRACE_NODE,
            summary=f"retrieval failed: {message}",
            detail={"query": query, "error": message},
        )
    )


def emit_retrieval(query: str, k: int, chunks: list[RetrievedChunk], latency_ms: float, error: str | None) -> None:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    scores = [chunk.score for chunk in chunks]
    source_ids = [chunk.source_id for chunk in chunks]
    distinct_papers = len(set(source_ids))
    emit_trace(
        TraceEvent(
            kind="rag_retrieval",
            node=TRACE_NODE,
            summary=f"retrieved {len(chunks)} chunks from {distinct_papers} papers",
            detail={
                "query": query,
                "k": k,
                "chunk_ids": chunk_ids,
                "source_ids": source_ids,
                "scores": scores,
                "latency_ms": round(latency_ms, 1),
                # Per call: summary's detection pass skips the re-ranker even when it is loaded.
                "reranked": any(chunk.reranked for chunk in chunks),
                "error": error,
            },
        )
    )


def query_collection(collection, query: str, count: int, source_id: str | None = None) -> list[RetrievedChunk]:
    query_arguments = {
        "query_texts": [query],
        "n_results": count,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_id is not None:
        query_arguments["where"] = {"source_id": source_id}
    return to_chunks(collection.query(**query_arguments))


def build_rerank_pool(collection, query: str, source_id: str | None) -> list[RetrievedChunk]:
    if source_id is not None:
        return query_collection(collection, query, RERANK_CHUNKS_PER_PAPER * 2, source_id)
    pool = query_collection(collection, query, RERANK_GLOBAL_CANDIDATES)
    top_papers: list[str] = []
    for chunk in pool:
        if chunk.source_id not in top_papers:
            top_papers.append(chunk.source_id)
    seen_chunk_ids = {chunk.chunk_id for chunk in pool}
    # Papers the query names come first: the embedder often misses them entirely.
    named_papers = find_aliased_papers(query)
    papers_to_expand = named_papers + [paper_id for paper_id in top_papers if paper_id not in named_papers]
    for paper_id in papers_to_expand[: RERANK_TOP_PAPERS + len(named_papers)]:
        for chunk in query_collection(collection, query, RERANK_CHUNKS_PER_PAPER, paper_id):
            if chunk.chunk_id not in seen_chunk_ids:
                pool.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
    return pool


def rerank_search(reranker, collection, query: str, k: int, source_id: str | None) -> list[RetrievedChunk]:
    pool = build_rerank_pool(collection, query, source_id)
    passages = [chunk.text for chunk in pool]
    relevance_logits = list(reranker.rerank(query, passages, batch_size=RERANK_BATCH_SIZE))
    # After re-ranking, `score` is the cross-encoder relevance squashed into [0, 1].
    for chunk, logit in zip(pool, relevance_logits):
        chunk.score = round(logit_to_unit(float(logit)), 4)
        chunk.reranked = True
    if source_id is not None:
        return sorted(pool, key=sort_key_by_score, reverse=True)[:k]
    return diversify(pool, k)


def search_index(query: str, k: int, source_id: str | None = None, rerank: bool = True) -> list[RetrievedChunk]:
    collection = get_collection()
    reranker = get_reranker() if rerank else None
    if reranker is not None:
        return rerank_search(reranker, collection, query, k, source_id)
    if source_id is not None:
        # Paper-scoped search: every chunk comes from one paper, so no diversity cap applies.
        query_result = collection.query(
            query_texts=[query],
            n_results=k,
            where={"source_id": source_id},
            include=["documents", "metadatas", "distances"],
        )
        return to_chunks(query_result)
    candidate_count = max(k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    query_result = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    candidates = to_chunks(query_result)
    return diversify(candidates, k)


def retrieve(
    query: str, k: int | None = None, source_id: str | None = None, rerank: bool = True
) -> list[RetrievedChunk]:
    started_at = time.perf_counter()
    top_k = k if k is not None else settings.retrieval_k
    chunks: list[RetrievedChunk] = []
    error_message: str | None = None

    if not query or not query.strip():
        error_message = "empty query"
    elif top_k <= 0:
        error_message = f"k must be positive, got {top_k}"
    else:
        try:
            chunks = search_index(query.strip(), top_k, source_id, rerank)
        except IndexUnavailableError as missing_index:
            error_message = f"{missing_index}; run `python -m src.rag.ingest`"
        except Exception as query_error:
            # Any Chroma/ONNX failure is reported, never raised to the agent.
            logger.exception("retrieval failed")
            error_message = f"{type(query_error).__name__}: {query_error}"

    if error_message is not None:
        emit_error(query, error_message)
    latency_ms = (time.perf_counter() - started_at) * 1000
    emit_retrieval(query, top_k, chunks, latency_ms, error_message)
    return chunks


def warm_up() -> bool:
    """Open the collection and run one query so the ONNX model is loaded before the first user.

    Called by main.py's startup hook. Returns True when the index answered, False otherwise.
    """
    try:
        # Also loads the cross-encoder, so the first user query does not pay for it.
        warm_chunks = search_index("retrieval-augmented generation", 1)
    except Exception:
        logger.exception("retriever warm-up failed")
        return False
    return len(warm_chunks) > 0
