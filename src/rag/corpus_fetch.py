"""Download a curated arXiv corpus (PDFs + metadata.json) into settings.corpus_dir.

Run with `python -m src.rag.corpus_fetch`. Idempotent: PDFs already on disk are skipped, and
metadata.json is rewritten from the arXiv API each run.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import requests

from src.config import settings

logger = logging.getLogger(__name__)

# Seconds to wait between PDF downloads, per arXiv's polite-use guidance.
DOWNLOAD_DELAY_SECONDS = 3.0
DOWNLOAD_TIMEOUT_SECONDS = 60
USER_AGENT = "csci599-rag-corpus-fetch/1.0 (academic assignment)"

# (arXiv id, a word or phrase that must appear in the title, used to verify the id is right)
CURATED_PAPERS: list[tuple[str, str]] = [
    # Reasoning and acting
    ("2201.11903", "Chain-of-Thought"),
    ("2203.11171", "Self-Consistency"),
    ("2305.10601", "Tree of Thoughts"),
    ("2210.03629", "ReAct"),
    ("2303.11366", "Reflexion"),
    ("2210.03350", "Compositionality Gap"),
    # Tool use
    ("2302.04761", "Toolformer"),
    ("2303.17580", "HuggingGPT"),
    ("2305.15334", "Gorilla"),
    ("2307.16789", "ToolLLM"),
    ("2304.08354", "Tool Learning"),
    # Single-agent systems, memory, embodied
    ("2304.03442", "Generative Agents"),
    ("2305.16291", "Voyager"),
    ("2310.08560", "MemGPT"),
    ("2405.15793", "SWE-agent"),
    # Multi-agent LLM systems
    ("2308.08155", "AutoGen"),
    ("2308.00352", "MetaGPT"),
    ("2303.17760", "CAMEL"),
    ("2307.07924", "Software Development"),
    ("2308.10848", "AgentVerse"),
    ("2305.14325", "Multiagent Debate"),
    # Agent surveys
    ("2308.11432", "Survey"),
    ("2309.07864", "Survey"),
    ("2402.01680", "Multi-Agents"),
    # Agent benchmarks and evaluation
    ("2308.03688", "AgentBench"),
    ("2307.13854", "WebArena"),
    ("2310.06770", "SWE-bench"),
    ("2306.05685", "LLM-as-a-Judge"),
    # Retrieval foundations
    ("2004.04906", "Dense Passage Retrieval"),
    ("2004.12832", "ColBERT"),
    ("2002.08909", "REALM"),
    ("2112.04426", "Retrieving from Trillions"),
    ("2208.03299", "Atlas"),
    # Retrieval-augmented generation
    ("2005.11401", "Retrieval-Augmented Generation"),
    ("2312.10997", "Retrieval-Augmented Generation"),
    ("2310.11511", "Self-RAG"),
    ("2401.18059", "RAPTOR"),
    ("2212.10496", "Relevance Labels"),
    ("2401.15884", "Corrective Retrieval"),
    ("2404.16130", "Graph RAG"),
    ("2305.06983", "Active Retrieval"),
    ("2212.10509", "Interleaving Retrieval"),
    ("2309.15217", "RAGAS"),
    ("2307.03172", "Lost in the Middle"),
]


@dataclass
class PaperMetadata:
    id: str  # base arXiv id without version, e.g. "2308.08155"
    versioned_id: str  # e.g. "2308.08155v2"
    title: str
    authors: list[str]
    published: str  # ISO date, e.g. "2023-08-16"
    primary_category: str
    categories: list[str]
    abstract: str
    url: str  # https://arxiv.org/abs/<id>
    pdf_file: str  # file name inside corpus_dir


def strip_version(versioned_id: str) -> str:
    # "2308.08155v2" -> "2308.08155"
    base_id, _separator, _version = versioned_id.partition("v")
    return base_id


def normalise_whitespace(text: str) -> str:
    words = text.split()
    return " ".join(words)


def to_metadata(result: arxiv.Result) -> PaperMetadata:
    versioned_id = result.get_short_id()
    base_id = strip_version(versioned_id)
    author_names = [author.name for author in result.authors]
    published_date = result.published.date()
    return PaperMetadata(
        id=base_id,
        versioned_id=versioned_id,
        title=normalise_whitespace(result.title),
        authors=author_names,
        published=published_date.isoformat(),
        primary_category=result.primary_category,
        categories=list(result.categories),
        abstract=normalise_whitespace(result.summary),
        url=f"https://arxiv.org/abs/{base_id}",
        pdf_file=f"{base_id}.pdf",
    )


def lookup_papers(client: arxiv.Client, arxiv_ids: list[str]) -> dict[str, arxiv.Result]:
    """One batched API query for every id; returns base id -> Result."""
    search = arxiv.Search(id_list=arxiv_ids, max_results=len(arxiv_ids))
    results_by_id: dict[str, arxiv.Result] = {}
    for result in client.results(search):
        base_id = strip_version(result.get_short_id())
        results_by_id[base_id] = result
    return results_by_id


def title_matches(result: arxiv.Result, expected_phrase: str) -> bool:
    lowered_title = result.title.lower()
    return expected_phrase.lower() in lowered_title


def download_pdf(session: requests.Session, pdf_url: str, destination: Path) -> None:
    response = session.get(pdf_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    # Write to a temp name first so an interrupted download never leaves a truncated "valid" file.
    partial_path = destination.with_suffix(".pdf.part")
    partial_path.write_bytes(response.content)
    partial_path.replace(destination)


def fetch_corpus(corpus_dir: str | None = None) -> list[PaperMetadata]:
    target_dir = Path(corpus_dir or settings.corpus_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=100, delay_seconds=DOWNLOAD_DELAY_SECONDS, num_retries=3)
    curated_ids = [arxiv_id for arxiv_id, _phrase in CURATED_PAPERS]
    results_by_id = lookup_papers(client, curated_ids)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    collected: list[PaperMetadata] = []
    downloaded_count = 0
    skipped_count = 0
    for arxiv_id, expected_phrase in CURATED_PAPERS:
        result = results_by_id.get(arxiv_id)
        if result is None:
            logger.warning("arXiv id %s did not resolve; skipping", arxiv_id)
            continue
        if not title_matches(result, expected_phrase):
            logger.warning("arXiv id %s title %r lacks %r; skipping", arxiv_id, result.title, expected_phrase)
            continue

        metadata = to_metadata(result)
        destination = target_dir / metadata.pdf_file
        if destination.exists() and destination.stat().st_size > 0:
            skipped_count += 1
        elif result.pdf_url is None:
            logger.warning("arXiv id %s has no PDF link; skipping", arxiv_id)
            continue
        else:
            # Only sleep before real network downloads, never for skipped files.
            if downloaded_count > 0:
                time.sleep(DOWNLOAD_DELAY_SECONDS)
            try:
                download_pdf(session, result.pdf_url, destination)
            except requests.RequestException as download_error:
                logger.warning("download failed for %s: %s", arxiv_id, download_error)
                continue
            downloaded_count += 1
            logger.info("downloaded %s (%s)", arxiv_id, metadata.title)
        collected.append(metadata)

    metadata_path = target_dir / "metadata.json"
    serialisable = [asdict(paper) for paper in collected]
    metadata_path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False))
    logger.info(
        "corpus ready: %d papers (%d downloaded, %d already present) -> %s",
        len(collected),
        downloaded_count,
        skipped_count,
        target_dir,
    )
    return collected


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_corpus()


if __name__ == "__main__":
    main()
