"""Pytest fixtures shared across test modules."""
import sys
from pathlib import Path

# Make repo-root modules (ingest, retrieve, ...) importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
