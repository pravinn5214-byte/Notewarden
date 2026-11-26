"""Suggested study questions per document section, cached in SQLite.

Generates one exam-style question per section (heading sections when present,
otherwise parent chunks), scores each with best_dense_score(), and caches in
logs/suggested.db keyed by content_hash so unchanged sections are never
regenerated on page reload.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from store import db_path as default_db_dir, get_collections

HIGH_CONF = 0.5
MED_CONF = 0.3
MAX_SECTIONS_PER_FILE = 15
MIN_SECTION_WORDS = 20
DEFAULT_DB_PATH = Path("logs/suggested.db")

PROMPT_TEMPLATE = (
    "Write one exam-style question that this passage answers. "
    "Return only the question.\n\n{text}"
)


def confidence_band(score: float) -> str:
    """Map a dense cosine score to 'High', 'Medium', or 'Low'."""
    if score >= HIGH_CONF:
        return "High"
    if score >= MED_CONF:
        return "Medium"
    return "Low"


def _db_path(db_path: Path | str | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return Path(os.environ.get("SUGGESTED_DB_PATH", str(DEFAULT_DB_PATH)))


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = _db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS suggested (
            file TEXT NOT NULL,
            page INTEGER NOT NULL,
            section TEXT NOT NULL,
            question TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (file, page, section)
        )
        """
    )
    conn.commit()
    return conn


def clear_cache(db_path: Path | str | None = None) -> None:
    """Clear all cached suggested questions (called after Repair index)."""
    path = _db_path(db_path)
    if not path.exists():
        return
    with _connect(path) as conn:
        conn.execute("DELETE FROM suggested")
        conn.commit()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_file_sections(
    source: str,
    db_dir: str | None = None,
    page_range: tuple[int, int] | None = None,
    limit: int = MAX_SECTIONS_PER_FILE,
) -> list[dict]:
    """Return up to `limit` sections for `source` from Chroma.

    Uses heading sections from `notes` when present; otherwise falls back to
    `parents` chunks. Skips sections shorter than MIN_SECTION_WORDS.
    """
    notes, parents = get_collections(db_dir or default_db_dir())
    got = notes.get(where={"source": source}, include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []

    has_headings = any((m or {}).get("section") for m in metas)
    sections: list[dict] = []

    if has_headings:
        grouped: dict[tuple[int, str], list[str]] = {}
        for doc, meta in zip(docs, metas):
            m = meta or {}
            page = int(m.get("page", 1))
            if page_range and not (page_range[0] <= page <= page_range[1]):
                continue
            sec = (m.get("section") or f"Page {page}").strip()
            grouped.setdefault((page, sec), []).append(doc or "")
        for (page, sec), parts in sorted(grouped.items()):
            text = "\n".join(parts).strip()
            if len(text.split()) >= MIN_SECTION_WORDS:
                sections.append({"file": source, "page": page, "section": sec, "text": text})
    else:
        pgot = parents.get(where={"source": source}, include=["documents", "metadatas"])
        pdocs = pgot.get("documents") or []
        pmetas = pgot.get("metadatas") or []
        items = []
        for idx, (doc, meta) in enumerate(zip(pdocs, pmetas)):
            m = meta or {}
            page = int(m.get("page", 1))
            if page_range and not (page_range[0] <= page <= page_range[1]):
                continue
            items.append((page, idx, doc or ""))
        items.sort(key=lambda t: (t[0], t[1]))
        for page, idx, text in items:
            if len(text.split()) >= MIN_SECTION_WORDS:
                sections.append({
                    "file": source,
                    "page": page,
                    "section": f"Page {page} · part {idx + 1}",
                    "text": text.strip(),
                })

    return sections[:limit]


def get_or_generate_suggestions(
    sources: list[str],
    db_dir: str | None = None,
    page_ranges: dict[str, tuple[int, int]] | None = None,
    force: bool = False,
    llm: Callable[[str], str] | None = None,
    score_fn: Callable[..., float] | None = None,
    db_path: Path | str | None = None,
) -> tuple[list[dict], str | None]:
    """Return (suggestions_sorted_desc, error_msg_or_None) for `sources`."""
    if not sources:
        return [], None

    if llm is None:
        from generate import complete
        llm = complete
    if score_fn is None:
        from retrieve import best_dense_score
        score_fn = best_dense_score

    active_db = db_dir or default_db_dir()
    conn = _connect(db_path)
    results: list[dict] = []
    error: str | None = None

    try:
        for src in sources:
            pr = (page_ranges or {}).get(src)
            try:
                secs = get_file_sections(src, db_dir=active_db, page_range=pr)
            except Exception:
                secs = []

            for sec in secs:
                chash = _content_hash(sec["text"])
                row = conn.execute(
                    "SELECT file, page, section, question, content_hash, confidence, created_at "
                    "FROM suggested WHERE file=? AND page=? AND section=?",
                    (sec["file"], sec["page"], sec["section"]),
                ).fetchone()

                if not force and row is not None and row["content_hash"] == chash:
                    results.append(dict(row))
                    continue

                if error is not None:
                    # Provider already failed in this pass; keep any stale row if present.
                    if row is not None:
                        results.append(dict(row))
                    continue

                try:
                    q_text = llm(PROMPT_TEMPLATE.format(text=sec["text"])).strip()
                    if not q_text:
                        continue
                    conf = float(score_fn(
                        q_text,
                        db_dir=active_db,
                        allowed_sources=[src],
                        page_ranges={src: pr} if pr else None,
                    ))
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        INSERT INTO suggested (file, page, section, question, content_hash, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(file, page, section) DO UPDATE SET
                            question=excluded.question,
                            content_hash=excluded.content_hash,
                            confidence=excluded.confidence,
                            created_at=excluded.created_at
                        """,
                        (sec["file"], sec["page"], sec["section"], q_text, chash, conf, now),
                    )
                    conn.commit()
                    results.append({
                        "file": sec["file"],
                        "page": sec["page"],
                        "section": sec["section"],
                        "question": q_text,
                        "content_hash": chash,
                        "confidence": conf,
                        "created_at": now,
                    })
                except Exception as exc:
                    error = f"Could not generate suggestions: {exc}"
                    if row is not None:
                        results.append(dict(row))
    finally:
        conn.close()

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results, error
