"""Build data/papers.db: arXiv paper METADATA for the LLM-agents / multi-agent / RAG domain.

The Data agent answers structured questions over this store ("RAG papers per year since 2020",
"most frequent authors", "cs.MA papers in 2024"). Rebuilds from scratch every run (idempotent):
the new database is written to a temp file and swapped in only when complete.

CLI: python -m src.data.build_db [--per-query N]
"""

import argparse
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings

DATE_FILTER = "submittedDate:[202001010000 TO 202612312359]"
CATEGORY_FILTER = "(cat:cs.CL OR cat:cs.AI OR cat:cs.MA OR cat:cs.LG OR cat:cs.IR)"

# topic tag -> arXiv query terms. A paper's `topic` is the first query (in this order) that found it;
# every matching topic is also kept in paper_topics.
TOPIC_QUERIES: dict[str, str] = {
    "rag": 'abs:"retrieval-augmented generation" OR abs:"retrieval augmented generation"',
    "llm_agent": 'abs:"LLM agent" OR abs:"LLM-based agent" OR abs:"language model agent"',
    "multi_agent": 'abs:"multi-agent" AND abs:"large language model"',
    "tool_use": 'abs:"tool use" OR abs:"tool learning" OR abs:"function calling"',
    "agent_planning": 'abs:"agent" AND abs:"planning" AND abs:"large language model"',
}

DEFAULT_PER_QUERY = 120
PAGE_SIZE = 60
DELAY_SECONDS = 3.5
NUM_RETRIES = 5

SCHEMA_SQL = """
CREATE TABLE papers (
    arxiv_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    published        DATE NOT NULL,
    updated          DATE,
    published_year   INTEGER NOT NULL,
    primary_category TEXT NOT NULL,
    abstract         TEXT,
    url              TEXT NOT NULL,
    topic            TEXT NOT NULL
);
CREATE TABLE authors (
    author_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);
CREATE TABLE paper_authors (
    arxiv_id  TEXT NOT NULL REFERENCES papers(arxiv_id),
    author_id INTEGER NOT NULL REFERENCES authors(author_id),
    position  INTEGER NOT NULL,
    PRIMARY KEY (arxiv_id, author_id)
);
CREATE TABLE categories (
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    category TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, category)
);
CREATE TABLE paper_topics (
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id),
    topic    TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, topic)
);
CREATE INDEX idx_papers_published ON papers(published);
CREATE INDEX idx_papers_year ON papers(published_year);
CREATE INDEX idx_papers_primary_category ON papers(primary_category);
CREATE INDEX idx_papers_topic ON papers(topic);
CREATE INDEX idx_paper_authors_author ON paper_authors(author_id);
CREATE INDEX idx_categories_category ON categories(category);
CREATE INDEX idx_paper_topics_topic ON paper_topics(topic);
"""

# Trailing "v2" etc. in an arXiv short id; the DB keys papers by the version-less id.
VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    published: str
    updated: str | None
    published_year: int
    primary_category: str
    abstract: str
    url: str
    topic: str
    authors: list[str]
    categories: list[str]
    topics: list[str] = field(default_factory=list)


def build_topic_query(topic_terms: str) -> str:
    return f"({topic_terms}) AND {CATEGORY_FILTER} AND {DATE_FILTER}"


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def record_from_result(arxiv_result, topic: str) -> PaperRecord:
    short_id = arxiv_result.get_short_id()
    arxiv_id = VERSION_SUFFIX_PATTERN.sub("", short_id)
    published_date = arxiv_result.published.date()
    updated_date = arxiv_result.updated.date() if arxiv_result.updated else None
    author_names = [author.name.strip() for author in arxiv_result.authors if author.name.strip()]
    return PaperRecord(
        arxiv_id=arxiv_id,
        title=collapse_whitespace(arxiv_result.title),
        published=published_date.isoformat(),
        updated=updated_date.isoformat() if updated_date else None,
        published_year=published_date.year,
        primary_category=arxiv_result.primary_category,
        abstract=collapse_whitespace(arxiv_result.summary or ""),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        topic=topic,
        authors=author_names,
        categories=list(dict.fromkeys(arxiv_result.categories)),
        topics=[topic],
    )


def fetch_papers(per_query: int) -> dict[str, PaperRecord]:
    """Run every topic query against the arXiv API; dedupe by arXiv id, keeping all topic tags."""
    import arxiv

    # A modest page size plus the client's delay between pages keeps us within arXiv's API etiquette.
    arxiv_client = arxiv.Client(page_size=PAGE_SIZE, delay_seconds=DELAY_SECONDS, num_retries=NUM_RETRIES)
    papers_by_id: dict[str, PaperRecord] = {}
    for topic, topic_terms in TOPIC_QUERIES.items():
        search = arxiv.Search(
            query=build_topic_query(topic_terms),
            max_results=per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        found_count = 0
        try:
            for arxiv_result in arxiv_client.results(search):
                record = record_from_result(arxiv_result, topic)
                found_count += 1
                existing_record = papers_by_id.get(record.arxiv_id)
                if existing_record is None:
                    papers_by_id[record.arxiv_id] = record
                elif topic not in existing_record.topics:
                    existing_record.topics.append(topic)
        except Exception as fetch_error:
            # A partial topic (e.g. arXiv hiccup mid-pagination) is still useful; keep what we got.
            print(f"[build_db] topic {topic}: stopped early: {fetch_error}", file=sys.stderr)
        print(f"[build_db] topic {topic}: {found_count} results, {len(papers_by_id)} unique so far")
    return papers_by_id


def write_database(papers_by_id: dict[str, PaperRecord], db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SCHEMA_SQL)
        author_ids: dict[str, int] = {}
        for record in papers_by_id.values():
            connection.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.arxiv_id,
                    record.title,
                    record.published,
                    record.updated,
                    record.published_year,
                    record.primary_category,
                    record.abstract,
                    record.url,
                    record.topic,
                ),
            )
            for position, author_name in enumerate(record.authors, start=1):
                if author_name not in author_ids:
                    cursor = connection.execute("INSERT INTO authors (name) VALUES (?)", (author_name,))
                    author_ids[author_name] = cursor.lastrowid
                # Same name listed twice on one paper would violate the PK; keep the first position.
                connection.execute(
                    "INSERT OR IGNORE INTO paper_authors VALUES (?, ?, ?)",
                    (record.arxiv_id, author_ids[author_name], position),
                )
            for category in record.categories:
                connection.execute("INSERT OR IGNORE INTO categories VALUES (?, ?)", (record.arxiv_id, category))
            for topic in record.topics:
                connection.execute("INSERT OR IGNORE INTO paper_topics VALUES (?, ?)", (record.arxiv_id, topic))
        connection.commit()
    finally:
        connection.close()


def count_rows(db_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(db_path)
    try:
        table_names = ["papers", "authors", "paper_authors", "categories", "paper_topics"]
        row_counts = {}
        for table_name in table_names:
            count_row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            row_counts[table_name] = count_row[0]
        return row_counts
    finally:
        connection.close()


def build_database(db_path: Path, per_query: int = DEFAULT_PER_QUERY) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    papers_by_id = fetch_papers(per_query)
    if not papers_by_id:
        raise RuntimeError("arXiv returned no papers; keeping the existing database untouched")
    temp_path = db_path.with_suffix(".db.tmp")
    if temp_path.exists():
        temp_path.unlink()
    write_database(papers_by_id, temp_path)
    # Atomic swap: readers never see a half-built database, and reruns fully replace the old one.
    os.replace(temp_path, db_path)
    return count_rows(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the arXiv paper-metadata SQLite DB.")
    parser.add_argument("--per-query", type=int, default=DEFAULT_PER_QUERY, help="max papers per topic query")
    parser.add_argument("--db-path", default=settings.papers_db_path)
    arguments = parser.parse_args()
    row_counts = build_database(Path(arguments.db_path), arguments.per_query)
    print(f"[build_db] wrote {arguments.db_path}: {row_counts}")


if __name__ == "__main__":
    main()
