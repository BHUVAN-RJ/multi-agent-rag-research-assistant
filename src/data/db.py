"""Read-only SQL access to data/papers.db for the Data agent.

The SQL comes from an LLM, so it is treated as untrusted: a lexical guard allows exactly one
SELECT/WITH statement, and the connection itself is opened read-only (`mode=ro`) so a statement
that slips past the guard still cannot write. Nothing here raises; errors go into `error`.
"""

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.trace_events import TraceEvent, emit_trace

QUERY_TIMEOUT_SECONDS = 10.0
PROGRESS_HANDLER_OPCODES = 10_000
TRACE_ROW_PREVIEW = 5

COLUMN_NOTES = """\
-- papers: one row per arXiv paper (metadata only, submitted 2020 or later; see coverage line).
--   arxiv_id (version-less, e.g. '2308.08155'), published/updated are ISO 'YYYY-MM-DD' strings,
--   published_year is an INTEGER copy of the year for easy GROUP BY, primary_category like 'cs.CL',
--   topic is ONE of 'rag', 'llm_agent', 'multi_agent', 'tool_use', 'agent_planning'
--   (the first search query that found the paper), url = https://arxiv.org/abs/<arxiv_id>.
-- authors: one row per distinct author name.
-- paper_authors: links papers to authors; position is 1-based author order (1 = first author).
-- categories: every arXiv category of a paper (including the primary one), e.g. 'cs.MA'.
-- paper_topics: every topic query that matched a paper (a paper may have several topics);
--   prefer this over papers.topic when counting papers per topic.
-- Example: SELECT published_year, COUNT(*) FROM papers p JOIN paper_topics t USING (arxiv_id)
--          WHERE t.topic = 'rag' GROUP BY published_year ORDER BY published_year;"""

# Statement-level keywords that can change state or the connection; rejected anywhere in the query.
FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "create", "alter", "attach", "detach",
    "pragma", "vacuum", "reindex", "analyze", "begin", "commit", "rollback", "savepoint",
    "release", "truncate", "grant", "load_extension",
)
# REPLACE is only blocked as "REPLACE INTO" so the replace() string function stays usable.
FORBIDDEN_PATTERN = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r"|replace\s+into)\b", re.IGNORECASE)
LEADING_KEYWORD_PATTERN = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# String literals and comments are blanked before keyword checks so 'drop' inside a title is fine.
STRING_LITERAL_PATTERN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
LINE_COMMENT_PATTERN = re.compile(r"--[^\n]*")
BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)


@dataclass
class DataQueryResult:
    columns: list[str]
    rows: list[list]
    sql: str
    error: str | None = None
    truncated: bool = False
    latency_ms: float = 0.0


def resolve_db_path() -> Path:
    return Path(settings.papers_db_path)


def open_readonly_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """SQLite connection that the engine itself refuses to write through."""
    target_path = db_path or resolve_db_path()
    if not target_path.exists():
        raise FileNotFoundError(f"papers DB not found at {target_path}; run python -m src.data.build_db")
    readonly_uri = f"file:{target_path.resolve()}?mode=ro"
    connection = sqlite3.connect(readonly_uri, uri=True, timeout=QUERY_TIMEOUT_SECONDS)
    # Belt and braces: query_only also blocks writes if mode=ro were ever dropped.
    connection.execute("PRAGMA query_only = ON")
    # sqlite's timeout only bounds lock waits; this handler aborts a runaway LLM-written query
    # (e.g. an accidental cross join) once it runs past QUERY_TIMEOUT_SECONDS.
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def abort_past_deadline() -> int:
        return 1 if time.monotonic() > deadline else 0

    connection.set_progress_handler(abort_past_deadline, PROGRESS_HANDLER_OPCODES)
    return connection


def strip_literals_and_comments(sql: str) -> str:
    without_block_comments = BLOCK_COMMENT_PATTERN.sub(" ", sql)
    without_line_comments = LINE_COMMENT_PATTERN.sub(" ", without_block_comments)
    return STRING_LITERAL_PATTERN.sub("''", without_line_comments)


def normalize_sql(sql: str) -> str:
    """Trim whitespace and one optional trailing semicolon (LLMs habitually add it)."""
    trimmed_sql = sql.strip()
    if trimmed_sql.endswith(";"):
        trimmed_sql = trimmed_sql[:-1].rstrip()
    return trimmed_sql


def validate_readonly_sql(sql: str) -> str | None:
    """Return a rejection reason, or None if `sql` is a single SELECT/WITH statement."""
    normalized_sql = normalize_sql(sql)
    if not normalized_sql:
        return "empty SQL"
    code_only = strip_literals_and_comments(normalized_sql)
    if ";" in code_only:
        return "only a single statement is allowed (found ';')"
    if LEADING_KEYWORD_PATTERN.match(code_only) is None:
        return "only SELECT or WITH queries are allowed"
    forbidden_match = FORBIDDEN_PATTERN.search(code_only)
    if forbidden_match is not None:
        return f"forbidden keyword: {forbidden_match.group(1).upper()}"
    return None


def emit_query_trace(result: DataQueryResult) -> None:
    status_text = f"error: {result.error}" if result.error else f"{len(result.rows)} rows"
    detail = {
        "sql": result.sql,
        "row_count": len(result.rows),
        "columns": result.columns,
        "rows_preview": result.rows[:TRACE_ROW_PREVIEW],
        "truncated": result.truncated,
        "latency_ms": round(result.latency_ms, 1),
        "error": result.error,
    }
    emit_trace(TraceEvent(kind="data_query", node="data", summary=f"SQL -> {status_text}", detail=detail))


def execute_query(sql: str, max_rows: int) -> tuple[list[str], list[list], bool]:
    connection = open_readonly_connection()
    try:
        cursor = connection.execute(sql)
        column_names = [column[0] for column in cursor.description or []]
        # Fetch one extra row so we know whether the result was cut off at max_rows.
        fetched_rows = cursor.fetchmany(max_rows + 1)
    finally:
        connection.close()
    was_truncated = len(fetched_rows) > max_rows
    kept_rows = [list(row) for row in fetched_rows[:max_rows]]
    return column_names, kept_rows, was_truncated


def run_readonly_sql(sql: str, max_rows: int = 50) -> DataQueryResult:
    start_time = time.perf_counter()
    normalized_sql = normalize_sql(sql or "")
    result = DataQueryResult(columns=[], rows=[], sql=normalized_sql)
    rejection_reason = validate_readonly_sql(normalized_sql)
    if rejection_reason is not None:
        result.error = f"rejected: {rejection_reason}"
    else:
        try:
            safe_max_rows = max(1, int(max_rows))
            result.columns, result.rows, result.truncated = execute_query(normalized_sql, safe_max_rows)
        except Exception as query_error:
            result.error = f"{type(query_error).__name__}: {query_error}"
    result.latency_ms = (time.perf_counter() - start_time) * 1000
    emit_query_trace(result)
    return result


def schema_description() -> str:
    """DDL of every table/index plus column notes, for the SQL-writing LLM."""
    try:
        connection = open_readonly_connection()
    except Exception as open_error:
        return f"-- papers DB unavailable: {open_error}\n{COLUMN_NOTES}"
    try:
        ddl_rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL "
            "ORDER BY type DESC, name"
        ).fetchall()
        summary_row = connection.execute(
            "SELECT COUNT(*), MIN(published), MAX(published) FROM papers"
        ).fetchone()
    except Exception as schema_error:
        return f"-- could not read schema: {schema_error}\n{COLUMN_NOTES}"
    finally:
        connection.close()
    ddl_statements = [ddl_row[0].strip() + ";" for ddl_row in ddl_rows]
    paper_count, earliest_date, latest_date = summary_row
    coverage_line = f"-- {paper_count} papers, published {earliest_date} .. {latest_date}"
    return "\n".join(["-- SQLite database (read-only)", *ddl_statements, coverage_line, COLUMN_NOTES])
