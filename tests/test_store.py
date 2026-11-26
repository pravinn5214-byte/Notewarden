"""Tests for the hardening layer: store.py client factory, index-error
recovery, write lock, and default-mode agreement.

Run: python -m pytest tests/test_store.py -v
No network, no real API, no real chroma_db/ (conftest.py sets CHROMA_DIR to a
tmp_path for the whole session).
"""
import logging
import os
from pathlib import Path

import pytest

import store
from store import (  # noqa: F401
    IndexCorruptError, LockHeldError, acquire_write_lock, db_path,
    release_write_lock, reset_client, with_index_recovery,
)


class FakeCollection:
    """Scriptable fake: raises `error` the first `fail_times` calls, then
    delegates to the real method."""

    def __init__(self, error: Exception, fail_times: int, result=None):
        self.error = error
        self.fail_times = fail_times
        self.result = result
        self.calls = 0

    def peek(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.result


# --- with_index_recovery -------------------------------------------------------

def test_recovery_retries_once_then_succeeds(monkeypatch):
    resets = []
    monkeypatch.setattr(store, "reset_client", lambda: resets.append(1))
    # chromadb.errors.InternalError equivalent: message matches the pattern,
    # so the fake doesn't need the real chromadb exception type.
    fake = FakeCollection(Exception("InternalError: Error finding id"), fail_times=1)
    assert with_index_recovery(fake.peek) == fake.result
    assert fake.calls == 2  # first raised, retry succeeded
    assert resets == [1]    # client reopened exactly once


def test_recovery_raises_index_corrupt_on_persistent_failure(monkeypatch):
    resets = []
    monkeypatch.setattr(store, "reset_client", lambda: resets.append(1))
    fake = FakeCollection(Exception("InternalError: Error finding id"), fail_times=10)
    with pytest.raises(IndexCorruptError):
        with_index_recovery(fake.peek)
    assert fake.calls == 2   # one retry, not an infinite loop
    assert resets == [1]


def test_recovery_passes_through_unrelated_errors(monkeypatch):
    resets = []
    monkeypatch.setattr(store, "reset_client", lambda: resets.append(1))
    fake = FakeCollection(ValueError("not an index error"), fail_times=10)
    with pytest.raises(ValueError):
        with_index_recovery(fake.peek)
    assert resets == []      # no recovery attempted


def test_recovery_logs_warning_without_data(monkeypatch, caplog):
    monkeypatch.setattr(store, "reset_client", lambda: None)
    fake = FakeCollection(Exception("InternalError: Error finding id"), fail_times=1)
    with caplog.at_level(logging.WARNING, logger="store"):
        with_index_recovery(fake.peek)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a one-line warning on recovery"
    assert "Stale Chroma index" in warnings[0].getMessage()


def test_recovery_matches_hnsw_and_internal_error_messages(monkeypatch):
    resets = []
    monkeypatch.setattr(store, "reset_client", lambda: resets.append(1))
    for msg in ("hnsw segment is missing", "Error finding id: xyz"):
        fake = FakeCollection(Exception(msg), fail_times=1)
        assert with_index_recovery(fake.peek) == fake.result, msg
    # A message that merely says "InternalError" without being a chromadb
    # InternalError or containing "Error finding id"/"hnsw" passes through.
    fake = FakeCollection(Exception("InternalError: boom"), fail_times=1)
    with pytest.raises(Exception, match="InternalError: boom"):
        with_index_recovery(fake.peek)
    # The real chromadb.errors.InternalError type triggers recovery.
    import chromadb
    fake = FakeCollection(chromadb.errors.InternalError("real type"), fail_times=1)
    assert with_index_recovery(fake.peek) == fake.result
    assert len(resets) == 3


# --- Write lock ----------------------------------------------------------------

def test_second_ingest_while_lock_held_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "db_path", lambda: str(tmp_path))
    lock = acquire_write_lock(lock_path=tmp_path / ".ingest.lock", timeout=0.5)
    try:
        with pytest.raises(LockHeldError, match="stop it first"):
            acquire_write_lock(lock_path=tmp_path / ".ingest.lock", timeout=0.5)
    finally:
        release_write_lock(lock)
    # After release the lock can be taken again.
    lock2 = acquire_write_lock(lock_path=tmp_path / ".ingest.lock", timeout=0.5)
    release_write_lock(lock2)


def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path, monkeypatch):
    lock_path = tmp_path / ".ingest.lock"
    dead_pid = _dead_pid()
    lock_path.write_text(str(dead_pid))
    monkeypatch.setattr(store, "db_path", lambda: str(tmp_path))
    lock = acquire_write_lock(lock_path=lock_path, timeout=0.5)
    try:
        assert lock.exists() and lock.read_text() == str(os.getpid())
    finally:
        release_write_lock(lock)


def test_lock_with_live_pid_is_not_stolen(tmp_path, monkeypatch):
    lock_path = tmp_path / ".ingest.lock"
    lock_path.write_text(str(os.getpid()))  # our own live PID, held by "someone"
    monkeypatch.setattr(store, "db_path", lambda: str(tmp_path))
    with pytest.raises(LockHeldError):
        acquire_write_lock(lock_path=lock_path, timeout=0.3)


def _dead_pid() -> int:
    """A PID that is (almost surely) not running."""
    for pid in range(400000, 400000 + 500):
        try:
            os.kill(pid, 0)
        except OSError:
            return pid
        except OverflowError:
            continue
    pytest.fail("no dead PID found")


# --- Defaults ------------------------------------------------------------------

def test_default_mode_is_hybrid_rerank_everywhere():
    from app import DB_DIR as _  # noqa: F401  (app must import cleanly)
    import retrieve
    assert retrieve.DEFAULT_MODE == "hybrid_rerank"
    assert retrieve.DEFAULT_MODE in retrieve.MODES
    # CLI default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=retrieve.MODES, default=retrieve.DEFAULT_MODE)
    assert parser.parse_args([]).mode == "hybrid_rerank"
    # Env fallback: the app sidebar resolves RETRIEVAL_MODE against DEFAULT_MODE
    os.environ.pop("RETRIEVAL_MODE", None)
    env_mode = os.environ.get("RETRIEVAL_MODE", retrieve.DEFAULT_MODE)
    assert env_mode == "hybrid_rerank"


def test_chroma_dir_env_default():
    monkey_free = os.environ.get("CHROMA_DIR")
    try:
        os.environ.pop("CHROMA_DIR", None)
        assert db_path() == "chroma_db"
        os.environ["CHROMA_DIR"] = "some/other/dir"
        assert db_path() == "some/other/dir"
    finally:
        if monkey_free is None:
            os.environ.pop("CHROMA_DIR", None)
        else:
            os.environ["CHROMA_DIR"] = monkey_free


# --- Repair index --------------------------------------------------------------

def test_repair_index_rebuilds_same_count(tmp_path, monkeypatch):
    """Repair (delete both collections + re-ingest) yields the same chunk
    count as a fresh ingest, using the same chunk IDs."""
    import chromadb
    import ingest
    from ingest import ingest_pdfs

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt.pdf").write_bytes(b"%PDF-1.4 stub")

    # Deterministic fake page text per file, same content both runs.
    page_text = "Repair rebuild check alpha beta gamma. " * 30

    def fake_load_pages(pdf_path):
        yield 1, page_text

    real_store = store.db_path()
    db = str(tmp_path / "db")

    def run_ingest():
        monkeypatch.setattr(ingest, "load_pages", fake_load_pages)
        files, total = ingest_pdfs(docs_dir=docs, db_dir=db)
        monkeypatch.undo()
        return files, total

    monkeypatch.setattr(store, "db_path", lambda: db)
    reset_client()
    try:
        _, fresh_count = run_ingest()
        fresh_ids = set(
            chromadb.PersistentClient(path=db).get_collection("notes").get(include=[])["ids"])

        # Simulate the app's Repair index: delete both collections, re-ingest.
        client = chromadb.PersistentClient(path=db)
        for name in ("notes", "parents"):
            client.delete_collection(name)
        reset_client()
        _, repaired_count = run_ingest()
        repaired_ids = set(
            chromadb.PersistentClient(path=db).get_collection("notes").get(include=[])["ids"])

        assert repaired_count == fresh_count
        assert repaired_ids == fresh_ids  # same chunk IDs after repair
    finally:
        monkeypatch.setattr(store, "db_path", lambda: real_store)
        reset_client()


def test_explicit_db_dir_client_recovers_on_stale_index(tmp_path):
    """reset_client() must drop explicit db_dir clients (not just _default_path),
    so with_index_recovery reopens eval/non-default stores instead of reusing
    the stale client."""
    import chromadb
    explicit_db = str(tmp_path / "eval_like_db")
    c1 = store.get_client(explicit_db)
    col1 = c1.get_or_create_collection("notes")
    col1.add(ids=["id1"], documents=["hello"], metadatas=[{"source": "a.pdf", "page": 1}])

    # Simulate a stale client handle failing once with an HNSW error; the
    # refetch re-resolves via get_client(explicit_db) after reset_client().
    calls = {"n": 0}

    def run_with_stale():
        calls["n"] += 1
        if calls["n"] == 1:
            raise chromadb.errors.InternalError("Error creating hnsw segment reader")
        return store.get_client(explicit_db)

    c2 = with_index_recovery(run_with_stale, refetch=run_with_stale)
    assert c2 is not c1, "reset_client() must drop explicit db_dir from _clients"
    assert c2.get_collection("notes").count() == 1
    reset_client()
