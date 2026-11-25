"""Tests for page-range metadata filters (Step 2).

Run: python -m pytest tests/test_page_filter.py -v
No network; throwaway corpus via stubbed load_pages, tmp CHROMA_DIR (conftest).
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import ingest_pdfs  # noqa: E402
from retrieve import (  # noqa: E402
    _normalize_page_ranges, _page_ok, _page_where_parts, parse_page_specs,
    retrieve, retrieve_multi,
)

# Vocabulary:
#  one.pdf p1: alpha words; p2: beta words; p3: gamma words
#  two.pdf p1: delta words
PAGES = {
    "one.pdf": {
        1: "Alpha beta gamma delta epsilon. " * 40,
        2: "Zeta eta theta iota kappa lambda. " * 40,
        3: "Gamma radiation shielding materials. " * 40,
    },
    "two.pdf": {
        1: "Delta force operations manual. " * 40,
    },
}
Q_BETA = "zeta eta theta"          # only on one.pdf p2
Q_GAMMA = "gamma"                  # on one.pdf p1 AND p3
Q_DELTA = "delta"                  # on one.pdf p1 and two.pdf p1


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import ingest as ingest_mod
    docs = tmp_path_factory.mktemp("pf_docs")
    for name in PAGES:
        (docs / name).write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("pf_db"))
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        for page_num, text in PAGES[pdf_path.name].items():
            yield page_num, text

    ingest_mod.load_pages = fake_load_pages
    try:
        ingest_pdfs(docs_dir=docs, db_dir=db)
    finally:
        ingest_mod.load_pages = orig
    import store as store_mod
    store_mod.reset_client()
    return docs, db


# --- Parsing and normalisation -------------------------------------------------

def test_parse_page_specs():
    assert parse_page_specs(["Unit3.pdf:10-35", "Unit1.pdf:1-5"]) == \
        {"Unit3.pdf": (10, 35), "Unit1.pdf": (1, 5)}
    assert parse_page_specs(None) == {}
    assert parse_page_specs([]) == {}
    with pytest.raises(ValueError):
        parse_page_specs(["no-range-here.pdf"])
    with pytest.raises(ValueError):
        parse_page_specs(["a.pdf:35-10"])   # inverted
    with pytest.raises(ValueError):
        parse_page_specs(["a.pdf:0-5"])     # pages start at 1


def test_normalize_shapes():
    assert _normalize_page_ranges({"a.pdf": (2, 4)}) == {"a.pdf": (2, 4)}
    assert _normalize_page_ranges([("a.pdf", 2, 4), ("b.pdf", 1, 1)]) == \
        {"a.pdf": (2, 4), "b.pdf": (1, 1)}
    assert _normalize_page_ranges({"a.pdf": None}) == {}
    assert _normalize_page_ranges(None) == {}
    assert _normalize_page_ranges({}) == {}


def test_page_ok_helper():
    ranges = {"a.pdf": (2, 4)}
    assert _page_ok({"source": "a.pdf", "page": 2}, ranges)
    assert _page_ok({"source": "a.pdf", "page": 4}, ranges)   # inclusive end
    assert not _page_ok({"source": "a.pdf", "page": 5}, ranges)
    assert not _page_ok({"source": "a.pdf", "page": 1}, ranges)
    assert _page_ok({"source": "b.pdf", "page": 99}, ranges)  # unconstrained file
    assert _page_ok({"source": "a.pdf", "page": 1}, {})       # no filter


def test_page_where_parts_grouping():
    # constrained file -> $and; unconstrained selected files -> grouped $in
    parts = _page_where_parts({"a.pdf", "b.pdf"}, {"a.pdf": (2, 3)})
    assert parts == [
        ("a.pdf", {"$and": [{"source": "a.pdf"}, {"page": {"$gte": 2}}, {"page": {"$lte": 3}}]}),
        (None, {"source": {"$in": ["b.pdf"]}}),
    ]
    # no ranges: plain source filter
    assert _page_where_parts({"a.pdf"}, {}) == [(None, {"source": {"$in": ["a.pdf"]}})]
    assert _page_where_parts(None, {}) == [(None, None)]


# --- Dense mode ----------------------------------------------------------------

def test_dense_respects_page_range(store):
    docs, db = store
    hits = retrieve(Q_GAMMA, k=4, db_dir=db, mode="dense",
                    page_ranges={"one.pdf": (3, 3)})
    assert hits, "page-filtered dense retrieval returned nothing"
    # one.pdf hits must all be page 3 (other files are unconstrained)
    assert all(h["page"] == 3 for h in hits if h["source"] == "one.pdf")
    assert any(h["source"] == "one.pdf" and h["page"] == 3 for h in hits)


def test_dense_range_excludes_page(store):
    docs, db = store
    # Dense always returns top-k (no score floor at this layer); the filter's
    # contract is that excluded pages never appear.
    hits = retrieve(Q_BETA, k=4, db_dir=db, mode="dense",
                    allowed_sources=["one.pdf"],
                    page_ranges={"one.pdf": (1, 1)})
    assert hits
    assert all(h["page"] == 1 for h in hits)  # p2 (the beta page) excluded


# --- BM25 filtered BEFORE scoring ----------------------------------------------

def test_bm25_prefilter_top_hits_outside_range(store, monkeypatch):
    """Even when the top BM25 hits for a query are outside the page range,
    in-range matches still come back."""
    docs, db = store
    import retrieve as r
    # Monkeypatch BM25Okapi to record the candidate set size at scoring time:
    # with the range applied, the p1/p2 chunks must already be gone.
    seen_sizes = []
    real_bm25 = r.BM25Okapi

    class SpyBM25(real_bm25):
        def __init__(self, corpus):
            seen_sizes.append(len(corpus))
            super().__init__(corpus)

    monkeypatch.setattr(r, "BM25Okapi", SpyBM25)

    # Scoring-time candidate set: p3's 2 chunks + unconstrained two.pdf's 2
    # (it has no range) = 4 — the other 5 (one.pdf p1/p2) are already gone.
    hits = retrieve(Q_GAMMA, k=4, db_dir=db, mode="hybrid",
                    page_ranges={"one.pdf": (3, 3)})
    assert seen_sizes[-1] == 4, f"BM25 candidate sizes: {seen_sizes}"
    assert seen_sizes[0] == 9  # the unfiltered corpus, for contrast
    # the out-of-range one.pdf pages (1, 2) can never appear
    assert hits and all(not (h["source"] == "one.pdf" and h["page"] != 3) for h in hits)


def test_bm25_prefilter_rerank_mode(store, monkeypatch):
    """Same pre-score filter in hybrid_rerank: in-range matches still come
    back non-empty even when the global top BM25 hits are out of range."""
    docs, db = store
    import retrieve as r
    seen_sizes = []
    real_bm25 = r.BM25Okapi

    class SpyBM25(real_bm25):
        def __init__(self, corpus):
            seen_sizes.append(len(corpus))
            super().__init__(corpus)

    monkeypatch.setattr(r, "BM25Okapi", SpyBM25)
    hits = retrieve(Q_GAMMA, k=4, db_dir=db, mode="hybrid_rerank",
                    page_ranges={"one.pdf": (3, 3)})
    assert seen_sizes[-1] == 4, f"BM25 candidate sizes: {seen_sizes}"
    assert hits, "hybrid_rerank returned nothing with an in-range candidate set"
    assert "rerank_score" in hits[0]
    assert all(not (h["source"] == "one.pdf" and h["page"] != 3) for h in hits)


def test_hybrid_respects_range(store):
    docs, db = store
    # one.pdf filtered to p2; two.pdf unconstrained.
    hits = retrieve(Q_BETA, k=4, db_dir=db, mode="hybrid",
                    allowed_sources=["one.pdf"],
                    page_ranges={"one.pdf": (2, 2)})
    assert hits
    assert all(h["source"] == "one.pdf" and h["page"] == 2 for h in hits)


def test_hybrid_rerank_respects_range(store):
    docs, db = store
    hits = retrieve(Q_BETA, k=4, db_dir=db, mode="hybrid_rerank",
                    allowed_sources=["one.pdf"],
                    page_ranges={"one.pdf": (2, 2)})
    assert hits
    assert all(h["source"] == "one.pdf" and h["page"] == 2 for h in hits)
    assert "rerank_score" in hits[0]
    # parent text came from the in-range parent only
    assert hits[0]["parent_text"]


def test_range_excluding_everything_returns_empty(store):
    docs, db = store
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        assert retrieve(Q_DELTA, k=4, db_dir=db, mode=mode,
                        allowed_sources=["two.pdf"],
                        page_ranges={"two.pdf": (99, 100)}) == [], mode


# --- Top BM25 hits outside the range -------------------------------------------

# notes.pdf p1/p3 are the two strongest BM25 matches for Q_WEAK but sit OUTSIDE
# the (2,2) range; the only in-range match is one 'zeta' deep inside p2, so the
# filter has real work to do and the in-range BM25 match is the weaker one.
FILLER = "alpha beta gamma delta epsilon zulu yotta. "
BM25_PAGES = {
    "notes.pdf": {
        1: "eta theta iota kappa lambda nu xi omicron pi rho sigma. " * 7,   # 77 w
        2: FILLER * 18 + "zeta " + FILLER * 10,                              # 197 w
        3: "eta theta iota kappa lambda nu xi omicron pi rho sigma. " * 5,   # 55 w
    },
    "other.pdf": {
        1: FILLER * 21,                                                      # 147 w
    },
}
Q_WEAK = "zeta eta theta"


@pytest.fixture(scope="module")
def store_bm25_outside(tmp_path_factory):
    """Second throwaway corpus (the `store` fixture's pages can't put the
    global BM25 top hits outside the range being tested)."""
    import ingest as ingest_mod
    docs = tmp_path_factory.mktemp("bm25_docs")
    for name in BM25_PAGES:
        (docs / name).write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("bm25_db"))
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        for page_num, text in BM25_PAGES[pdf_path.name].items():
            yield page_num, text

    ingest_mod.load_pages = fake_load_pages
    try:
        ingest_pdfs(docs_dir=docs, db_dir=db)
    finally:
        ingest_mod.load_pages = orig
    import store as store_mod
    store_mod.reset_client()
    return docs, db


def test_top_bm25_hits_outside_range_inrange_still_returned(store_bm25_outside):
    """The global top BM25 hits are outside the page range, yet in-range
    results still come back non-empty (and BM25 still contributes) in both
    hybrid and hybrid_rerank."""
    docs, db = store_bm25_outside
    import retrieve as r

    # 1) Without the filter the two strongest BM25 hits are the OUT-of-range
    #    pages — so the assertion below is testing something real.
    ids, texts, metas, parent_texts, bm25 = r._get_bm25(db, None)
    scores = bm25.get_scores(r.tokenize(Q_WEAK))
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    top_pages = [(metas[i]["source"], metas[i]["page"]) for i in order[:2]]
    assert top_pages == [("notes.pdf", 1), ("notes.pdf", 3)], top_pages

    ranges = {"notes.pdf": (2, 2)}
    for mode in ("hybrid", "hybrid_rerank"):
        hits = retrieve(Q_WEAK, k=4, db_dir=db, mode=mode, page_ranges=ranges)
        assert hits, f"{mode} returned nothing while the top BM25 hits were out of range"
        assert all(not (h["source"] == "notes.pdf" and h["page"] != 2) for h in hits), mode
        # the weak in-range BM25 match survived the pre-score filter
        assert any(h.get("bm25_rank") for h in hits), f"{mode}: no BM25 contribution"


# --- Multi-topic path ----------------------------------------------------------

def test_multi_topic_applies_filter_per_subquery(store):
    docs, db = store
    subs = [Q_BETA, "delta force operations"]
    # one.pdf restricted to p1 excludes beta (p2); two.pdf unconstrained
    hits, supported = retrieve_multi(subs, mode="hybrid", top_k=4, db_dir=db,
                                     files=["one.pdf", "two.pdf"],
                                     page_ranges={"one.pdf": (1, 1)})
    assert 0 not in supported  # beta sub-query has no in-range evidence
    assert 1 in supported
    for h in hits:
        if h["source"] == "one.pdf":
            assert h["page"] == 1  # nothing from one.pdf p2 can appear
    # same call without the filter supports both
    hits2, supported2 = retrieve_multi(subs, mode="hybrid", top_k=4, db_dir=db,
                                       files=["one.pdf", "two.pdf"])
    assert supported2 == {0, 1}


# --- Parity: no filter = identical results --------------------------------------

def test_no_filter_parity_all_modes(store):
    docs, db = store
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        base = retrieve(Q_GAMMA, k=4, db_dir=db, mode=mode)
        none_arg = retrieve(Q_GAMMA, k=4, db_dir=db, mode=mode, page_ranges=None)
        empty_arg = retrieve(Q_GAMMA, k=4, db_dir=db, mode=mode, page_ranges={})
        assert [(h["id"], h["score"]) for h in base] == [(h["id"], h["score"]) for h in none_arg]
        assert [(h["id"], h["score"]) for h in base] == [(h["id"], h["score"]) for h in empty_arg]


# --- query_log page_filter column + migration -----------------------------------

def test_page_filter_column_logged(tmp_path, monkeypatch):
    path = tmp_path / "q.db"
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setenv("QUERY_LOG_PATH", str(path))
    from query_log import log_query
    log_query(original_query="q", page_filter=[{"file": "a.pdf", "start": 2, "end": 4}])
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT page_filter FROM queries").fetchone()
    conn.close()
    assert json.loads(row[0]) == [{"file": "a.pdf", "start": 2, "end": 4}]


def test_migrate_old_db_without_page_filter(tmp_path, monkeypatch):
    """An old DB (pre-page_filter) migrates cleanly and keeps its rows."""
    path = tmp_path / "old.db"
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setenv("QUERY_LOG_PATH", str(path))
    # Build an old-schema DB by hand (schema as of Step 1: no page_filter).
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE queries (
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
    """)
    conn.execute("INSERT INTO queries (ts, original_query) VALUES ('2026-01-01T00:00:00', 'old row')")
    conn.commit()
    conn.close()

    from query_log import log_query
    log_query(original_query="new row", page_filter=[{"file": "a.pdf", "start": 1, "end": 2}])

    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT id, original_query, page_filter FROM queries ORDER BY id").fetchall()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(queries)")}
    conn.close()
    assert cols >= {"page_filter"}
    assert len(rows) == 2                      # old row kept
    assert rows[0][1] == "old row" and rows[0][2] is None
    assert json.loads(rows[1][2]) == [{"file": "a.pdf", "start": 1, "end": 2}]


# --- Agent path carries the filter ----------------------------------------------

def test_agent_passes_page_filter_and_logs_it(tmp_path, monkeypatch):
    import agent as agent_mod
    from agent import answer_with_agent

    captured = {}

    def fake_rwf(query, mode, top_k, files, db_dir, llm=None, trace=None, page_ranges=None):
        captured["page_ranges"] = page_ranges
        trace["page_filter"] = {f: f"{s}-{e}" for f, (s, e) in sorted((page_ranges or {}).items())}
        trace["queries_tried"] = [query]
        from decompose import Decomposition
        from generate import MIN_SCORE
        hits = [{"text": "ctx", "source": "a.pdf", "page": 3, "score": 0.8,
                 "dense_score": 0.8, "parent_id": "P"}]
        return hits, {0}, Decomposition([query]), trace

    monkeypatch.setattr(agent_mod, "retrieve_with_fallback", fake_rwf)
    monkeypatch.setattr("generate._call_claude", lambda p, api_key: "answer [a.pdf p.3]")
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "q.db"))

    result = answer_with_agent("explain the alpha beta gamma process in detail",
                               history=[], db_dir="unused",
                               page_ranges={"a.pdf": (2, 4)})
    assert captured["page_ranges"] == {"a.pdf": (2, 4)}
    assert result.trace["page_filter"] == {"a.pdf": "2-4"}
    conn = sqlite3.connect(tmp_path / "q.db")
    pf = conn.execute("SELECT page_filter FROM queries").fetchone()[0]
    conn.close()
    assert json.loads(pf) == [{"file": "a.pdf", "start": 2, "end": 4}]
