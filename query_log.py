"""Query logging: every question the app or CLI answers, into SQLite.

One row per query. The schema mirrors what the pipeline already knows at the
moment it finishes a turn:
    id, ts (UTC ISO), session_id,
    original_query, rewritten_query, retrieval_mode, top_k,
    decompose_used, agent_mode, sub_queries (JSON list),
    results (JSON list of {file, page, dense_score, bm25_rank, rerank_score, ocr}),
    top_dense_score, gated_out (bool), unsupported_topics (JSON list),
    retries, clarified (bool), llm_provider, latency_ms, error.

Privacy: the answer text is NOT stored unless QUERY_LOG_ANSWERS=on (opt-in,
stored in a separate `answers` table keyed by query id). API keys are never
stored. Logging is best-effort: every entry point wraps in try/except and
swallows errors with a warning — a logging problem must never break a query.

Env vars:
    QUERY_LOG          on|off (default on)
    QUERY_LOG_PATH     SQLite file (default logs/queries.db)
    QUERY_LOG_ANSWERS  on to also store answer text (default off)

Usage (usually via log_query, not directly):
    from query_log import log_query
    log_query(session_id="...", original_query="...", ...)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs") / "queries.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    original_query TEXT NOT NULL,
    rewritten_query TEXT,
    retrieval_mode TEXT,
    top_k INTEGER,
    decompose_used INTEGER NOT NULL DEFAULT 0,
    agent_mode INTEGER NOT NULL DEFAULT 0,
    sub_queries TEXT,
    results TEXT,
    top_dense_score REAL,
    gated_out INTEGER NOT NULL DEFAULT 0,
    unsupported_topics TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    clarified INTEGER NOT NULL DEFAULT 0,
    llm_provider TEXT,
    latency_ms REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS answers (
    query_id INTEGER PRIMARY KEY REFERENCES queries(id),
    answer TEXT
);
"""


def enabled() -> bool:
    """QUERY_LOG=off disables all logging (nothing is written)."""
    return os.environ.get("QUERY_LOG", "on").lower() != "off"


def answers_enabled() -> bool:
    """QUERY_LOG_ANSWERS=on opts in to storing answer text."""
    return os.environ.get("QUERY_LOG_ANSWERS", "off").lower() == "on"


def log_path() -> Path:
    return Path(os.environ.get("QUERY_LOG_PATH") or DEFAULT_LOG_PATH)


def _connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=5)
    conn.executescript(_SCHEMA)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add missing columns to an existing table (safe to run on every write).
    SQLite has no ALTER ... ADD COLUMN IF NOT EXISTS, so check the schema."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
    for name, ddl in (
        ("page_filter", "TEXT"),  # Step 2: JSON of per-file page ranges
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE queries ADD COLUMN {name} {ddl}")


def log_query(
    session_id: str = "",
    original_query: str = "",
    rewritten_query: str | None = None,
    retrieval_mode: str = "",
    top_k: int | None = None,
    decompose_used: bool = False,
    agent_mode: bool = False,
    sub_queries: list[str] | None = None,
    results: list[dict] | None = None,
    top_dense_score: float | None = None,
    gated_out: bool = False,
    unsupported_topics: list[str] | None = None,
    retries: int = 0,
    clarified: bool = False,
    llm_provider: str = "",
    latency_ms: float | None = None,
    error: str | None = None,
    answer: str | None = None,
    page_filter: list | None = None,
    path: Path | str | None = None,
) -> int | None:
    """Insert one row; returns the row id (None if disabled or on failure).

    Never raises: any failure is logged as a warning and swallowed, so the
    app keeps working when the DB file is locked, unwritable, or corrupt.
    """
    if not enabled():
        return None
    try:
        conn = _connect(Path(path) if path else None)
        try:
            _migrate(conn)
            cur = conn.execute(
                """INSERT INTO queries
                   (ts, session_id, original_query, rewritten_query, retrieval_mode,
                    top_k, decompose_used, agent_mode, sub_queries, results,
                    top_dense_score, gated_out, unsupported_topics, retries,
                    clarified, llm_provider, latency_ms, error, page_filter)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds"),
                    session_id or "",
                    original_query or "",
                    rewritten_query,
                    retrieval_mode or "",
                    top_k,
                    int(bool(decompose_used)),
                    int(bool(agent_mode)),
                    json.dumps(sub_queries) if sub_queries else None,
                    json.dumps(results) if results else None,
                    top_dense_score,
                    int(bool(gated_out)),
                    json.dumps(unsupported_topics) if unsupported_topics else None,
                    int(retries or 0),
                    int(bool(clarified)),
                    llm_provider or "",
                    latency_ms,
                    error,
                    json.dumps(page_filter) if page_filter else None,
                ),
            )
            if answer is not None and answers_enabled():
                conn.execute("INSERT INTO answers (query_id, answer) VALUES (?,?)",
                             (cur.lastrowid, answer))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Query logging failed (ignored): %s", exc)
        return None


def rows_since(since: str | None = None, path: Path | str | None = None) -> list[sqlite3.Row]:
    """Read rows, newest first. `since` is an ISO timestamp lower bound."""
    if not Path(path or log_path()).exists():
        return []
    sql = "SELECT * FROM queries"
    params: tuple = ()
    if since:
        sql += " WHERE ts >= ?"
        params = (since,)
    sql += " ORDER BY id DESC"
    with closing(sqlite3.connect(Path(path or log_path()))) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def answer_for(query_id: int, path: Path | str | None = None) -> str | None:
    """Stored answer text for one query (only when QUERY_LOG_ANSWERS=on)."""
    if not Path(path or log_path()).exists():
        return None
    with closing(sqlite3.connect(Path(path or log_path()))) as conn:
        row = conn.execute("SELECT answer FROM answers WHERE query_id = ?", (query_id,)).fetchone()
    return row[0] if row else None


def results_payload(hits: list[dict] | None) -> list[dict]:
    """Trim retrieval hits to the logged shape: no chunk text (privacy/size),
    just {file, page, dense_score, bm25_rank, rerank_score, ocr}."""
    out = []
    for h in hits or []:
        out.append({
            "file": h.get("source", ""),
            "page": h.get("page"),
            "dense_score": h.get("dense_score"),
            "bm25_rank": h.get("bm25_rank"),
            "rerank_score": h.get("rerank_score"),
            "ocr": bool(h.get("ocr", False)),
        })
    return out


def top_dense(hits: list[dict] | None) -> float | None:
    scores = [h.get("dense_score") for h in hits or [] if h.get("dense_score") is not None]
    return max(scores) if scores else None


def gated(hits: list[dict] | None, min_score: float) -> bool:
    """True when NO hit reaches the relevance gate — the single definition of
    `gated_out` shared by every entry point (generate.py CLI, app.py, agent.py)
    and by generate_answer's own gate.

    The gate passes iff the BEST dense cosine among the hits reaches
    min_score, i.e. exactly `top_dense(hits) < min_score`, so a logged row is
    always self-consistent: gated_out == (top_dense_score < MIN_SCORE).

    Why best and not first: hits are ordered by the mode's ranking score
    (RRF / cross-encoder), so hits[0] can be a low-dense hit while a later hit
    clears the gate — gating on hits[0] used to suppress answers (and log
    gated_out=1 with top_dense_score >= MIN_SCORE) even though retrieval had
    found relevant notes. `retrieve_multi` already gates per sub-query with
    this same "any hit >= MIN_SCORE" rule.
    """
    return not [h for h in (hits or [])
                if h.get("dense_score", h.get("score", 0.0)) >= min_score]


def _load_json(value: str | None) -> list:
    try:
        return json.loads(value) if value else []
    except (json.JSONDecodeError, ValueError):
        return []
