"""Single Chroma client factory + index-error recovery + write lock.

One place creates the PersistentClient (path from CHROMA_DIR env var, default
chroma_db/); all modules go through it. If a read/query/get dies with a stale
HNSW segment ("InternalError: Error finding id", etc.), with_index_recovery
reopens the client once and retries; a second failure raises IndexCorruptError.

A file lock (lockfile with PID + stale detection, no third-party dependency)
serializes writes (ingest / deletes) between the app and the CLI.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, TypeVar

import chromadb

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = "chroma_db"
# Single source of truth; ingest.py re-exports these (COLLECTION names must
# not change — see ingest.py).
COLLECTION = "notes"           # child chunks (embedded)
PARENT_COLLECTION = "parents"  # parent chunks (not embedded, context only)
_clients: dict[str, chromadb.api.ClientAPI] = {}
_default_path: str | None = None
_reset_hooks: list[Callable[[], None]] = []


def on_reset(hook: Callable[[], None]) -> None:
    """Register a callback run by reset_client() so consumers drop handles
    obtained from the old client (e.g. retrieve.py's cached collections)."""
    _reset_hooks.append(hook)

T = TypeVar("T")


def db_path() -> str:
    return os.environ.get("CHROMA_DIR", DEFAULT_DB_DIR)


def get_client(db_dir: str | None = None) -> chromadb.api.ClientAPI:
    """Cached PersistentClient. No argument (or the CHROMA_DIR path) returns
    the shared client that reset_client() can drop; an explicit path (tests,
    evaluate.py) gets its own cached client."""
    global _default_path
    if db_dir is None or db_dir == db_path():
        path = db_path()
        if path not in _clients:
            _clients[path] = chromadb.PersistentClient(path=path)
        _default_path = path
        return _clients[path]
    if db_dir not in _clients:
        _clients[db_dir] = chromadb.PersistentClient(path=db_dir)
    return _clients[db_dir]


def get_collections(db_dir: str | None = None) -> tuple[chromadb.Collection, chromadb.Collection]:
    """(notes, parents) collections."""
    client = get_client(db_dir)
    return client.get_collection(COLLECTION), client.get_collection(PARENT_COLLECTION)


def reset_client() -> None:
    """Drop the cached default client so the next get_client() reopens it."""
    global _default_path
    if _default_path is not None:
        _clients.pop(_default_path, None)
        _default_path = None
    for hook in _reset_hooks:
        hook()


class IndexCorruptError(RuntimeError):
    """Chroma still fails after a client reopen — the index needs rebuilding."""


def _is_index_error(exc: BaseException) -> bool:
    if isinstance(exc, chromadb.errors.InternalError):
        return True
    msg = str(exc).lower()
    return "error finding id" in msg or "hnsw" in msg


def with_index_recovery(fn: Callable[[], T], refetch: Callable[[], T] | None = None) -> T:
    """Run fn(); on a stale-index error reopen the client and retry once.

    fn must not capture collection handles from the old client in its closure.
    If it does, pass `refetch`: a factory that re-resolves everything through
    get_client() — the retry runs refetch() instead of fn(), so it uses the
    reopened client's fresh handles.
    """
    try:
        return fn()
    except Exception as exc:
        if not _is_index_error(exc):
            raise
        logger.warning("Stale Chroma index detected; reopening the client and retrying once")
        reset_client()
    retry_fn = refetch if refetch is not None else fn
    try:
        return retry_fn()
    except Exception as exc:
        if _is_index_error(exc):
            raise IndexCorruptError(
                "The Chroma index is out of sync (read failed twice). Rebuild it "
                "with 'Repair index' in the app sidebar, or delete the index "
                "directory and re-ingest."
            ) from exc
        raise


def health_check(db_dir: str | None = None) -> None:
    """Tiny read on both collections inside with_index_recovery.

    Raises (IndexCorruptError or the underlying error) if the store can't be
    read even after recovery; returns the child chunk count on success.
    """
    if db_dir is not None:
        os.environ["CHROMA_DIR"] = db_dir
        reset_client()

    def peek() -> int:
        client = get_client()
        notes = client.get_collection(COLLECTION)
        notes.peek()
        parents = client.get_collection(PARENT_COLLECTION)
        parents.peek()
        return notes.count()

    return with_index_recovery(peek)


# --- Write lock ---------------------------------------------------------------
# Simple lockfile with PID + stale detection (avoids adding a dependency).
# Held only for writes (ingestion / deletes); reads never lock.

LOCK_TIMEOUT = 30.0  # seconds to wait before giving up


class LockHeldError(RuntimeError):
    """Another process holds the write lock."""


def acquire_write_lock(lock_path: str | Path | None = None, timeout: float = LOCK_TIMEOUT) -> Path:
    """Block up to `timeout` seconds for the write lock; raise LockHeldError
    if it stays held. A stale lock (dead PID, older than 6h) is reclaimed.
    Returns the lock path (caller may pass it to release_write_lock)."""
    path = Path(lock_path) if lock_path else Path(db_path()) / ".ingest.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return path
        except FileExistsError:
            if _lock_is_stale(path):
                logger.warning("Reclaiming stale ingest lock (dead PID)")
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise LockHeldError(
                    "The app (or another ingest) is running; stop it first"
                )
            time.sleep(0.25)


def release_write_lock(path: str | Path | None = None) -> None:
    if path is None:
        return
    Path(path).unlink(missing_ok=True)


def _lock_is_stale(path: Path) -> bool:
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return True  # unreadable/corrupt lock: treat as stale
    if pid == os.getpid():
        return False
    # Windows has no safe kill-0 probe for other users' PIDs; os.kill with
    # signal 0 raises OSError for a dead PID and succeeds for a live one.
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    except Exception:
        return False
    # Live PID but ancient lock (crashed machine / reused PID): reclaim after 6h.
    try:
        return time.time() - path.stat().st_mtime > 6 * 3600
    except OSError:
        return True
