"""Pytest fixtures shared across test modules."""
import os
import sys
from pathlib import Path

# Make repo-root modules (ingest, retrieve, ...) importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# Test isolation: every test runs against a throwaway Chroma dir. This runs
# at conftest import time — before test modules import ingest/store — so
# ingest.DB_DIR and store.db_path() resolve to the tmp path, never the real
# chroma_db/, even if a test forgets to pass db_dir=.
_REAL_DB = Path(__file__).resolve().parents[1] / "chroma_db"
os.environ.setdefault("CHROMA_DIR", str(Path(__file__).parent / "_test_chroma_db"))


@pytest.fixture(scope="session", autouse=True)
def _isolated_chroma_dir(tmp_path_factory):
    os.environ["CHROMA_DIR"] = str(tmp_path_factory.mktemp("chroma_db_test"))
    try:
        import store
        store.reset_client()
    except ImportError:
        pass
    yield os.environ["CHROMA_DIR"]
    os.environ.pop("CHROMA_DIR", None)
    try:
        import store
        store.reset_client()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _fail_if_touching_real_db(request):
    """Loud guard: any test whose store resolves to the real chroma_db/ fails."""
    yield  # run the test first, then check what path the store used
    try:
        import store
    except ImportError:
        return
    resolved = Path(store.db_path()).resolve()
    if resolved == _REAL_DB.resolve():
        pytest.fail(
            f"Test isolation broken: {request.node.nodeid} resolved the store "
            f"to the real chroma_db/ ({resolved}). CHROMA_DIR must point at a "
            "tmp_path directory in tests."
        )
