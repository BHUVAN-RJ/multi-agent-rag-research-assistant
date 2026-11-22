"""Central configuration. Every module reads settings from here, never from os.environ directly."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def read_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def read_env_list(name: str, default: str = "") -> list[str]:
    raw_value = read_env(name, default)
    stripped_items = [item.strip() for item in raw_value.split(",")]
    return [item for item in stripped_items if item]


@dataclass(frozen=True)
class Settings:
    # NVIDIA NIM (OpenAI-compatible)
    nvidia_api_key: str = field(default_factory=lambda: read_env("NVIDIA_API_KEY"))
    nvidia_base_url: str = field(
        default_factory=lambda: read_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    nvidia_model_fast: str = field(default_factory=lambda: read_env("NVIDIA_MODEL_FAST"))
    nvidia_model_strong: str = field(default_factory=lambda: read_env("NVIDIA_MODEL_STRONG"))

    # Google Gemini (direct)
    google_api_key: str = field(default_factory=lambda: read_env("GOOGLE_API_KEY"))
    gemini_model_fast: str = field(default_factory=lambda: read_env("GEMINI_MODEL_FAST"))
    gemini_model_strong: str = field(default_factory=lambda: read_env("GEMINI_MODEL_STRONG"))

    # OpenRouter: free models only, tried in order
    openrouter_api_key: str = field(default_factory=lambda: read_env("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(
        default_factory=lambda: read_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_free_models: list[str] = field(
        default_factory=lambda: read_env_list("OPENROUTER_FREE_MODELS")
    )

    # Real-time knowledge
    tavily_api_key: str = field(default_factory=lambda: read_env("TAVILY_API_KEY"))

    # RAG
    chroma_persist_dir: str = field(
        default_factory=lambda: read_env("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_index"))
    )
    chroma_collection: str = field(default_factory=lambda: read_env("CHROMA_COLLECTION", "arxiv_corpus"))
    corpus_dir: str = field(default_factory=lambda: read_env("CORPUS_DIR", str(PROJECT_ROOT / "data" / "corpus")))
    retrieval_k: int = field(default_factory=lambda: int(read_env("RETRIEVAL_K", "5")))
    # Cross-encoder that re-ranks embedding candidates (fastembed, ONNX). Empty disables it.
    rerank_model: str = field(default_factory=lambda: read_env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    # ONNX otherwise sizes its thread pool from the host's cores, not the container's vCPUs,
    # and oversubscribes Cloud Run's CPU (re-ranking took 15-27 s on 1 vCPU before this).
    rerank_threads: int = field(default_factory=lambda: int(read_env("RERANK_THREADS", "2")))

    # Data agent
    papers_db_path: str = field(
        default_factory=lambda: read_env("PAPERS_DB_PATH", str(PROJECT_ROOT / "data" / "papers.db"))
    )


settings = Settings()
