# syntax=docker/dockerfile:1
#
# Cloud Run image for the multi-agent RAG service.
#
# The Chroma index (chroma_index/) and the paper-metadata DB (data/papers.db) are
# generated locally by `python -m src.rag.ingest` and `python -m src.data.build_db`
# and BAKED into the image, so the container never needs the corpus PDFs. The ONNX
# embedding model Chroma uses for queries is downloaded at build time as well, so a
# cold start makes no network call before serving the first retrieval.

# ---------------------------------------------------------------- build stage
FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.13-slim AS runtime

# HOME is fixed so Chroma's model cache path (~/.cache/chroma/onnx_models, computed from
# Path.home() at import time) is the same during the build-time download and at runtime.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/app \
    ANONYMIZED_TELEMETRY=False \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed \
    PORT=8080

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv

WORKDIR /app

# Explicit COPYs (not `COPY . .`): the build fails loudly if a generated artifact is
# missing, and nothing outside this list can slip into the image.
COPY --chown=app:app requirements.txt main.py ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app chroma_index/ ./chroma_index/
COPY --chown=app:app data/papers.db ./data/papers.db

# Chroma opens its SQLite store read-write (locks, WAL), so /app stays owned by `app`.
USER app

# Pre-download all-MiniLM-L6-v2 into /home/app/.cache/chroma as the runtime user, embed
# one string to prove the model loads, then verify the files are where runtime will look.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; \
vectors = DefaultEmbeddingFunction()(['warm up the embedding model']); \
assert len(vectors) == 1 and len(vectors[0]) == 384, 'unexpected embedding shape'; \
print('ONNX embedding model cached, dim', len(vectors[0]))" \
    && test -f /home/app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/model.onnx

# Pre-download the cross-encoder re-ranker the same way (fastembed reads FASTEMBED_CACHE_PATH),
# and score one pair so a broken model fails the build instead of silently disabling re-ranking.
RUN python -c "from src.config import settings; from src.rag.retriever import load_cross_encoder; \
reranker = load_cross_encoder(settings.rerank_model, settings.rerank_threads); \
scores = list(reranker.rerank('what is retrieval', ['retrieval finds passages', 'bananas are yellow'])); \
assert len(scores) == 2 and scores[0] > scores[1], f'reranker scores look wrong: {scores}'; \
print('reranker cached:', settings.rerank_model)"

# Fail the build (not the first user request) if the baked index or DB is broken: the
# collection must open, hold chunks, and answer a query; papers.db must hold papers.
RUN python -c "import chromadb, sqlite3; \
collection = chromadb.PersistentClient(path='/app/chroma_index').get_collection('arxiv_corpus'); \
chunk_count = collection.count(); \
assert chunk_count > 1000, f'index too small: {chunk_count} chunks'; \
hits = collection.query(query_texts=['retrieval-augmented generation'], n_results=1); \
assert hits['ids'][0], 'index returned no hits'; \
paper_count = sqlite3.connect('/app/data/papers.db').execute('SELECT COUNT(*) FROM papers').fetchone()[0]; \
assert paper_count > 100, f'papers.db too small: {paper_count}'; \
print('baked index ok:', chunk_count, 'chunks;', paper_count, 'papers')"

EXPOSE 8080

# Shell form so $PORT (injected by Cloud Run) expands; exec keeps uvicorn as PID 1 so it
# receives SIGTERM directly and shuts down gracefully.
CMD exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --proxy-headers --forwarded-allow-ips="*" --timeout-keep-alive 75
