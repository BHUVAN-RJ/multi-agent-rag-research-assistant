"""Tests for src/data/db.py against the built data/papers.db."""

import dataclasses
import sqlite3
from pathlib import Path

import pytest

from src.config import settings
from src.data import db
from src.trace_events import fallback_sink

DB_TABLES = ["papers", "authors", "paper_authors", "categories", "paper_topics"]

requires_db = pytest.mark.skipif(
    not Path(settings.papers_db_path).exists(), reason="data/papers.db not built (python -m src.data.build_db)"
)


@pytest.fixture
def captured_events():
    event_list = []
    sink_token = fallback_sink.set(event_list.append)
    yield event_list
    fallback_sink.reset(sink_token)


def data_traces(event_list):
    trace_payloads = [event["data"] for event in event_list if event["type"] == "trace"]
    return [payload for payload in trace_payloads if payload["kind"] == "data_query"]


@pytest.mark.parametrize(
    "unsafe_sql",
    [
        "DROP TABLE papers",
        "INSERT INTO authors (name) VALUES ('x')",
        "SELECT 1; DROP TABLE papers",
        "SELECT * FROM papers; SELECT * FROM authors",
        "PRAGMA table_info(papers)",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "WITH x AS (SELECT 1) DELETE FROM papers",
        "UPDATE papers SET title = 'x'",
        "  ",
    ],
)
def test_guard_rejects_unsafe_sql(unsafe_sql, captured_events):
    result = db.run_readonly_sql(unsafe_sql)
    assert result.error is not None
    assert result.error.startswith("rejected")
    assert result.rows == []
    [trace] = data_traces(captured_events)
    assert trace["detail"]["error"] == result.error


def test_guard_allows_keywords_inside_strings_and_trailing_semicolon():
    assert db.validate_readonly_sql("SELECT title FROM papers WHERE title LIKE '%drop; table%';") is None
    assert db.validate_readonly_sql("SELECT replace(title, 'a', 'b'), updated FROM papers") is None


@requires_db
def test_group_by_query_returns_columns_and_rows(captured_events):
    sql = (
        "SELECT p.published_year AS year, COUNT(*) AS paper_count "
        "FROM papers p JOIN paper_topics t USING (arxiv_id) "
        "WHERE t.topic = 'rag' GROUP BY p.published_year ORDER BY year"
    )
    result = db.run_readonly_sql(sql)
    assert result.error is None
    assert result.columns == ["year", "paper_count"]
    assert len(result.rows) > 0
    assert all(row[1] > 0 for row in result.rows)
    [trace] = data_traces(captured_events)
    assert trace["detail"]["row_count"] == len(result.rows)
    assert trace["detail"]["sql"] == sql


@requires_db
def test_with_query_and_author_join():
    sql = (
        "WITH author_counts AS (SELECT author_id, COUNT(*) AS n FROM paper_authors GROUP BY author_id) "
        "SELECT a.name, c.n FROM author_counts c JOIN authors a USING (author_id) ORDER BY c.n DESC LIMIT 3"
    )
    result = db.run_readonly_sql(sql)
    assert result.error is None
    assert len(result.rows) == 3
    assert result.rows[0][1] >= result.rows[-1][1]


@requires_db
def test_max_rows_is_enforced():
    result = db.run_readonly_sql("SELECT arxiv_id FROM papers", max_rows=7)
    assert result.error is None
    assert len(result.rows) == 7
    assert result.truncated is True


@requires_db
def test_sql_error_is_returned_not_raised():
    result = db.run_readonly_sql("SELECT no_such_column FROM papers")
    assert result.error is not None
    assert "no_such_column" in result.error


@requires_db
def test_readonly_connection_cannot_write_even_without_guard():
    connection = db.open_readonly_connection()
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO authors (name) VALUES ('guard bypass')")
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DROP TABLE papers")
    finally:
        connection.close()
    surviving_count = db.run_readonly_sql("SELECT COUNT(*) FROM papers").rows[0][0]
    assert surviving_count > 0


@requires_db
def test_schema_description_mentions_every_table():
    description = db.schema_description()
    for table_name in DB_TABLES:
        assert f"CREATE TABLE {table_name}" in description
    assert "published_year" in description


@requires_db
def test_built_db_has_expected_domain_content():
    paper_count = db.run_readonly_sql("SELECT COUNT(*) FROM papers").rows[0][0]
    topic_rows = db.run_readonly_sql("SELECT DISTINCT topic FROM paper_topics ORDER BY topic").rows
    assert 300 <= paper_count <= 700
    assert [row[0] for row in topic_rows] == sorted(["rag", "llm_agent", "multi_agent", "tool_use", "agent_planning"])


def test_missing_db_returns_error(monkeypatch, tmp_path):
    missing_settings = dataclasses.replace(settings, papers_db_path=str(tmp_path / "absent.db"))
    monkeypatch.setattr(db, "settings", missing_settings)
    result = db.run_readonly_sql("SELECT 1")
    assert result.error is not None
    assert "not found" in result.error
    assert "unavailable" in db.schema_description()


def test_runaway_query_is_interrupted(monkeypatch):
    from src.data import db as db_module

    monkeypatch.setattr(db_module, "QUERY_TIMEOUT_SECONDS", 0.2)
    # A triple self cross join over authors is billions of rows: it must be cut off, not hang.
    runaway_sql = "SELECT count(*) FROM authors a, authors b, authors c"
    result = db_module.run_readonly_sql(runaway_sql)
    assert result.error is not None
    assert "interrupt" in result.error.lower()
