"""Tests for query logging (query_log.py), the log_report maths, and the
pipeline hooks (agent.py, app.py, generate.py __main__).

Run: python -m pytest tests/test_query_log.py -v
No network; every test uses a tmp QUERY_LOG_PATH and tmp CHROMA_DIR (conftest).
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import query_log
from query_log import log_query, results_payload, top_dense


@pytest.fixture(autouse=True)
def _fake_answer_llm(monkeypatch):
    """generate_answer's final LLM call never hits the network in tests
    (same pattern as tests/test_agent.py: provider forced to claude with a
    dummy key, _call_claude faked)."""
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-tests")
    monkeypatch.setattr(
        "generate._call_claude",
        lambda prompt, api_key: "Ribosomes translate mRNA into proteins. [a.pdf p.1]")


@pytest.fixture(autouse=True)
def _log_env(tmp_path, monkeypatch):
    """Every test logs into its own tmp DB and starts with logging on."""
    path = tmp_path / "queries.db"
    monkeypatch.setenv("QUERY_LOG_PATH", str(path))
    monkeypatch.setenv("QUERY_LOG", "on")
    monkeypatch.delenv("QUERY_LOG_ANSWERS", raising=False)
    yield path


def _rows(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM queries ORDER BY id").fetchall()
    finally:
        conn.close()


HITS = [
    {"id": "a", "text": "T", "source": "one.pdf", "page": 1, "parent_id": "P1",
     "dense_score": 0.55, "bm25_rank": 2, "ocr": False},
    {"id": "b", "text": "T2", "source": "two.pdf", "page": 3, "parent_id": "P2",
     "dense_score": 0.42, "bm25_rank": 1, "rerank_score": 1.7, "ocr": True},
]


# --- Row written per query ----------------------------------------------------

def test_row_written_per_query(_log_env):
    rid = log_query(original_query="what is normalization?",
                    retrieval_mode="hybrid_rerank", top_k=4,
                    results=results_payload(HITS), top_dense_score=0.55,
                    llm_provider="ollama", latency_ms=123.4)
    assert rid == 1
    rows = _rows(_log_env)
    assert len(rows) == 1
    r = rows[0]
    assert r["original_query"] == "what is normalization?"
    assert r["retrieval_mode"] == "hybrid_rerank"
    assert r["top_k"] == 4
    assert r["llm_provider"] == "ollama"
    assert abs(r["latency_ms"] - 123.4) < 1e-9
    assert r["results"] and json.loads(r["results"])[0]["file"] == "one.pdf"
    # Second query appends a new row.
    log_query(original_query="second")
    assert len(_rows(_log_env)) == 2


def test_results_payload_shape():
    payload = results_payload(HITS)
    assert payload[0] == {"file": "one.pdf", "page": 1, "dense_score": 0.55,
                          "bm25_rank": 2, "rerank_score": None, "ocr": False}
    assert payload[1]["rerank_score"] == 1.7 and payload[1]["ocr"] is True
    # No chunk text ever leaks into the log payload.
    assert "text" not in json.dumps(payload)


def test_top_dense():
    assert top_dense(HITS) == 0.55
    assert top_dense([]) is None
    assert top_dense([{"source": "x", "page": 1}]) is None


# --- Multi-topic fields -------------------------------------------------------

def test_multitopic_fields_populated(_log_env):
    log_query(
        original_query="compare X and Y",
        retrieval_mode="hybrid", top_k=4,
        decompose_used=True, sub_queries=["what is X?", "what is Y?"],
        results=results_payload(HITS), top_dense_score=0.55,
        unsupported_topics=["what is Y?"],
    )
    r = _rows(_log_env)[0]
    assert r["decompose_used"] == 1
    assert json.loads(r["sub_queries"]) == ["what is X?", "what is Y?"]
    assert json.loads(r["unsupported_topics"]) == ["what is Y?"]


# --- QUERY_LOG=off ------------------------------------------------------------

def test_query_log_off_writes_nothing(_log_env, monkeypatch):
    monkeypatch.setenv("QUERY_LOG", "off")
    assert log_query(original_query="q") is None
    assert not _log_env.exists()


# --- Answers opt-in -----------------------------------------------------------

def test_answers_not_stored_by_default(_log_env):
    rid = log_query(original_query="q", answer="the secret answer text")
    conn = sqlite3.connect(_log_env)
    tables = {t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "answers" not in tables or conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0
    conn.close()


def test_answers_optin_stored(_log_env, monkeypatch):
    monkeypatch.setenv("QUERY_LOG_ANSWERS", "on")
    rid = log_query(original_query="q", answer="the secret answer text")
    assert query_log.answer_for(rid) == "the secret answer text"


# --- Failure must not raise ---------------------------------------------------

def test_logging_failure_does_not_raise(tmp_path, monkeypatch):
    # Unwritable path: log_query swallows the error.
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "no" / "dir" / "q.db"))
    monkeypatch.setenv("QUERY_LOG", "on")
    # Make mkdir fail: point the parent at an existing *file*.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("QUERY_LOG_PATH", str(blocker / "q.db"))
    assert log_query(original_query="q") is None  # no exception


def test_corrupt_db_does_not_raise(tmp_path, monkeypatch):
    db = tmp_path / "queries.db"
    db.write_bytes(b"this is not a sqlite database" * 10)
    monkeypatch.setenv("QUERY_LOG_PATH", str(db))
    assert log_query(original_query="q") is None


# --- Gated-out consistency ----------------------------------------------------

def test_gated_out_matches_min_score_semantics(_log_env):
    """Regression: gated_out must be exactly 'no hit reaches MIN_SCORE'.
    Mirrors the hooks in agent.py/_log_turn and app.py's plain path."""
    import time as _t
    from generate import MIN_SCORE
    strong = [{"source": "a.pdf", "page": 1, "dense_score": MIN_SCORE + 0.05}]
    weak = [{"source": "a.pdf", "page": 1, "dense_score": MIN_SCORE - 0.05}]
    for label, hits, want in (("strong", strong, 0), ("weak", weak, 1)):
        log_query(original_query=label, results=results_payload(hits),
                  top_dense_score=top_dense(hits),
                  gated_out=not [h for h in hits if h.get("dense_score", 0.0) >= MIN_SCORE])
    rows = _rows(_log_env)
    by_q = {r["original_query"]: r for r in rows[-2:]}
    assert by_q["strong"]["gated_out"] == 0 and by_q["strong"]["top_dense_score"] > MIN_SCORE
    assert by_q["weak"]["gated_out"] == 1 and by_q["weak"]["top_dense_score"] < MIN_SCORE
    # invariant: gated_out == (top_dense_score < MIN_SCORE)
    assert (by_q["weak"]["gated_out"] == 1) == (by_q["weak"]["top_dense_score"] < MIN_SCORE)
    assert (by_q["strong"]["gated_out"] == 1) == (by_q["strong"]["top_dense_score"] < MIN_SCORE)


def test_gated_invariant_holds_for_ranking_reordered_hits(_log_env):
    """Regression (verification 1c): hits are ordered by the mode's ranking
    score, NOT by dense cosine, so the ranking top can be below the gate while
    another hit clears it. Every entry point must log the shared
    query_log.gated() verdict, which is by construction identical to
    top_dense_score < MIN_SCORE — in both directions."""
    from generate import MIN_SCORE
    from query_log import gated
    reordered = [  # ranking order: weak dense first, strong dense second
        {"source": "a.pdf", "page": 1, "dense_score": MIN_SCORE - 0.2, "rerank_score": 5.0},
        {"source": "a.pdf", "page": 2, "dense_score": MIN_SCORE + 0.2, "rerank_score": 1.0},
    ]
    all_weak = [{**h, "dense_score": MIN_SCORE - 0.2} for h in reordered]
    for label, hits in (("reordered", reordered), ("all_weak", all_weak)):
        log_query(original_query=label, results=results_payload(hits),
                  top_dense_score=top_dense(hits), gated_out=gated(hits, MIN_SCORE))
    rows = {r["original_query"]: r for r in _rows(_log_env)}
    # ranking top is weak, but a later hit clears the gate -> not gated
    assert rows["reordered"]["gated_out"] == 0
    assert rows["reordered"]["top_dense_score"] > MIN_SCORE
    # nothing clears the gate -> gated
    assert rows["all_weak"]["gated_out"] == 1
    assert rows["all_weak"]["top_dense_score"] < MIN_SCORE
    # the invariant the verification asks for, on both rows
    for r in rows.values():
        assert bool(r["gated_out"]) == (r["top_dense_score"] < MIN_SCORE)


# --- Report maths on a seeded DB ----------------------------------------------

def _seed(path, n_weak=0):
    """Seed deterministic rows; weak = gated out with a low dense score."""
    for i in range(10):
        log_query(
            original_query=f"query {i}",
            retrieval_mode="hybrid",
            top_k=4,
            sub_queries=["sub a", "sub b"] if i % 5 == 0 else None,
            results=results_payload([{"source": f"file{i % 2}.pdf", "page": 1,
                                      "dense_score": 0.1 if i < n_weak else 0.6}]),
            top_dense_score=0.1 if i < n_weak else 0.6,
            gated_out=i < n_weak,
            unsupported_topics=["sub a"] if i % 5 == 0 and i < n_weak else None,
            retries=1 if i % 4 == 0 else 0,
            clarified=i == 0,
            latency_ms=100.0 + i * 10,
        )


def test_report_math(_log_env):
    from log_report import build_report
    _seed(_log_env, n_weak=3)
    rows = _rows(_log_env)
    report = build_report(rows)
    assert "Total queries: 10" in report
    # 3 gated out, 0 errored -> 30% no-answer rate.
    assert "No-answer / gated-out rate: 30%" in report
    # 1 clarified / 10, 3 retried (i in {0,4,8}).
    assert "Clarify rate: 10%" in report
    assert "Retry rate: 30%" in report
    # latencies 100..190, avg = 145
    assert "Average latency: 145 ms" in report
    assert "p95 latency: 190 ms" in report
    # Weakest-10 table contains the low scores first.
    assert "| query 0 |" in report or "query 0" in report
    assert "0.100" in report
    # Most-used files: each file appeared in 5 rows' results.
    assert "| file0.pdf | 5 |" in report and "| file1.pdf | 5 |" in report


def test_report_empty_db(_log_env):
    from log_report import build_report
    assert "No queries logged yet." in build_report([])


def test_report_weakest_sorted(_log_env):
    from log_report import build_report
    _seed(_log_env, n_weak=2)
    rows = _rows(_log_env)
    report = build_report(rows)
    weakest_section = report.split("## 10 lowest")[1].split("## Most-used")[0]
    scores = [float(line.split("|")[3]) for line in weakest_section.splitlines()
              if line.startswith("| 20") or line.startswith("| 2")]
    assert scores == sorted(scores)


# --- --since and --csv --------------------------------------------------------

def test_since_ts_parsing():
    from log_report import since_ts
    assert since_ts("7d") is not None
    assert since_ts("24h") is not None
    assert since_ts("4w") is not None
    assert since_ts("2mo") is not None
    assert since_ts("junk") is None
    assert since_ts("") is None


def test_rows_since_filter(_log_env):
    from log_report import since_ts
    from query_log import rows_since
    log_query(original_query="old")
    # Cutoff 'now': only rows written after this moment match.
    cutoff = since_ts("0h")
    import time as _t
    _t.sleep(0.05)
    log_query(original_query="new")
    rows = rows_since(cutoff)
    assert [r["original_query"] for r in rows] == ["new"]


def test_csv_export(_log_env, tmp_path):
    from log_report import write_csv
    _seed(_log_env, n_weak=1)
    out = tmp_path / "out.csv"
    write_csv(_rows(_log_env), str(out))
    import csv as _csv
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(_csv.reader(f))
    assert rows[0][0] == "id" and rows[0][3] == "original_query"
    assert len(rows) == 11  # header + 10 seeded rows


# --- Pipeline hooks -----------------------------------------------------------

def test_agent_turn_logs_query(store_agent_db, tmp_path, monkeypatch):
    """answer_with_agent writes one row with agent fields populated."""
    db = store_agent_db
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "queries.db"))
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    from agent import answer_with_agent
    result = answer_with_agent("how do ribosomes translate mrna into proteins",
                               history=[], mode="hybrid", top_k=3, db_dir=str(db))
    rows = _rows(os.path.join(str(tmp_path), "queries.db"))
    assert len(rows) == 1
    r = rows[0]
    assert r["original_query"].startswith("how do ribosomes")
    assert r["agent_mode"] == 1
    assert r["retrieval_mode"] == "hybrid"
    assert json.loads(r["results"])[0]["file"] == "a.pdf"
    assert r["top_dense_score"] is not None and r["top_dense_score"] > 0.3
    assert not json.loads(r["results"])[0].get("text", "")  # no chunk text
    assert result.action == "answer"


def test_agent_turn_failure_does_not_break_answer(store_agent_db, monkeypatch):
    """If the log DB is unwritable, the agent still answers."""
    db = store_agent_db
    blocker = Path(db) / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("QUERY_LOG_PATH", str(blocker / "queries.db"))
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    from agent import answer_with_agent
    result = answer_with_agent("how do ribosomes translate mrna into proteins",
                               history=[], mode="hybrid", top_k=3, db_dir=str(db))
    assert result.action == "answer"


def test_agent_clarify_logged(store_agent_db, tmp_path, monkeypatch):
    db = store_agent_db
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "queries.db"))
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    # Clarify phrasing goes through the faked _call_claude (no network).
    from agent import answer_with_agent
    result = answer_with_agent("quantum chromodynamics confinement of quarks",
                               history=[], mode="hybrid", top_k=3, db_dir=str(db))
    rows = _rows(os.path.join(str(tmp_path), "queries.db"))
    assert len(rows) == 1
    if result.action == "clarify":
        assert rows[0]["clarified"] == 1


# --- Store fixture (reuses the agent-test pattern) ----------------------------

@pytest.fixture(scope="module")
def store_agent_db(tmp_path_factory):
    """Minimal two-PDF corpus ingested into a throwaway store (no network:
    load_pages stubbed like tests/test_retrieval.py)."""
    import ingest as ingest_mod
    TOPIC_A = "ribosomes translate mRNA into proteins using transfer RNA. " * 30
    TOPIC_A_P2 = "ribosome subunits assemble in the nucleolus before export. " * 30
    TOPIC_B = "the Tagverse CRM lead pipeline moves leads through stages. " * 25
    pages = {"a.pdf": {1: TOPIC_A, 2: TOPIC_A_P2}, "b.pdf": {1: TOPIC_B}}
    docs = tmp_path_factory.mktemp("qlog_docs")
    for name in pages:
        (docs / name).write_bytes(b"%PDF-1.4 stub")
    db = tmp_path_factory.mktemp("qlog_db")
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        for page_num, text in pages[pdf_path.name].items():
            yield page_num, text

    ingest_mod.load_pages = fake_load_pages
    try:
        ingest_mod.ingest_pdfs(docs_dir=docs, db_dir=str(db))
    finally:
        ingest_mod.load_pages = orig
    import store as store_mod
    store_mod.reset_client()  # drop handles from any earlier test's client
    return str(db)


# --- A2: decomposition format=json + object schema ------------------------------

def test_valid_subqueries_accepts_query_objects():
    """llama3.2 (format=json) returns [{"query": "..."}] objects; the
    validator must accept both shapes."""
    from decompose import _valid_subqueries
    assert _valid_subqueries(["plain string", "another one"]) == \
        ["plain string", "another one"]
    assert _valid_subqueries([{"query": "topic one"}, {"query": "topic two"}]) == \
        ["topic one", "topic two"]
    assert _valid_subqueries([{"query": "x"}, "plain"]) is None  # mixed shapes rejected
    assert _valid_subqueries([{"nope": "x"}]) is None
    assert _valid_subqueries([{"query": 5}]) is None
    assert _valid_subqueries([{"query": "a"}, {"query": "a"}]) is None  # duplicates
    assert _valid_subqueries([]) is None


def test_multitopic_log_row_has_sub_queries(store_agent_db, tmp_path, monkeypatch):
    """A decomposed turn logs the sub_queries JSON (fake provider, no network)."""
    db = store_agent_db
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "queries.db"))
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    monkeypatch.setenv("DECOMPOSE", "on")
    # Fake the LLM: any decompose/utility call gets a valid two-topic JSON
    # reply in llama3.2's object schema; generate_answer uses the faked claude.
    import generate
    monkeypatch.setattr(generate, "complete",
                        lambda prompt, max_tokens=200, json_mode=False:
                        '{"subqueries": [{"query": "how do ribosomes translate mrna"},'
                        ' {"query": "crm lead pipeline stages"}], "answer_as_parts": true}')
    from agent import answer_with_agent
    result = answer_with_agent(
        "explain how ribosomes translate mrna and also describe the crm lead pipeline stages",
        history=[], mode="hybrid", top_k=3, db_dir=str(db))
    import decompose as dec_mod
    dec_mod.clear_cache()
    conn = sqlite3.connect(tmp_path / "queries.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM queries ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    subs = json.loads(row["sub_queries"]) if row["sub_queries"] else []
    assert len(subs) == 2, f"sub_queries not logged: {row['sub_queries']}"
    assert row["decompose_used"] == 1
    assert result.action == "answer"


def test_complete_json_mode_flag_forwarded(monkeypatch):
    """complete(json_mode=True) adds format:json for ollama only."""
    import generate
    captured = {}
    class FakeResp:
        status_code = 200
        def json(self):
            return {"response": "{}"}
    def fake_post(url, json=None, **kw):
        captured["url"] = url
        captured["body"] = json
        return FakeResp()
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(generate.requests, "post", fake_post)
    generate.complete("p", json_mode=True)
    assert captured["body"]["format"] == "json"
    generate.complete("p", json_mode=False)
    assert "format" not in captured["body"]
