"""Tests for suggested.py: one question per section, SQLite caching, hash
regeneration, confidence band thresholds, and LLM provider failure handling.

Run: python -m pytest tests/test_suggested.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import suggested


def _fake_sections(sections):
    return lambda source, db_dir=None, page_range=None, limit=suggested.MAX_SECTIONS_PER_FILE: sections


def test_one_question_per_section_and_caching(monkeypatch, tmp_path):
    db_file = tmp_path / "suggested.db"
    secs = [
        {"file": "notes.pdf", "page": 1, "section": "Intro", "text": "word " * 25},
        {"file": "notes.pdf", "page": 2, "section": "Methods", "text": "alpha " * 25},
    ]
    monkeypatch.setattr(suggested, "get_file_sections", _fake_sections(secs))

    llm_calls = []

    def fake_llm(prompt):
        llm_calls.append(prompt)
        return f"Question {len(llm_calls)}?"

    scores = iter([0.62, 0.35])
    rows, err = suggested.get_or_generate_suggestions(
        ["notes.pdf"],
        llm=fake_llm,
        score_fn=lambda q, **kw: next(scores),
        db_path=db_file,
    )
    assert err is None
    assert len(llm_calls) == 2
    assert len(rows) == 2
    # Sorted high to low by confidence
    assert rows[0]["confidence"] == 0.62
    assert rows[1]["confidence"] == 0.35

    # Second call with unchanged content must NOT call the LLM again.
    rows2, err2 = suggested.get_or_generate_suggestions(
        ["notes.pdf"],
        llm=fake_llm,
        score_fn=lambda q, **kw: 0.99,
        db_path=db_file,
    )
    assert err2 is None
    assert len(llm_calls) == 2
    assert [r["question"] for r in rows2] == [r["question"] for r in rows]


def test_regeneration_on_changed_hash(monkeypatch, tmp_path):
    db_file = tmp_path / "suggested.db"
    secs = [{"file": "notes.pdf", "page": 1, "section": "Intro", "text": "original " * 25}]
    monkeypatch.setattr(suggested, "get_file_sections", _fake_sections(secs))

    calls = {"n": 0}

    def fake_llm(_prompt):
        calls["n"] += 1
        return f"Q{calls['n']}?"

    suggested.get_or_generate_suggestions(
        ["notes.pdf"], llm=fake_llm, score_fn=lambda q, **kw: 0.55, db_path=db_file,
    )
    assert calls["n"] == 1

    # Change section content -> hash changes -> must regenerate.
    secs[0]["text"] = "updated " * 25
    rows, err = suggested.get_or_generate_suggestions(
        ["notes.pdf"], llm=fake_llm, score_fn=lambda q, **kw: 0.55, db_path=db_file,
    )
    assert err is None
    assert calls["n"] == 2
    assert rows[0]["question"] == "Q2?"


def test_confidence_band_thresholds():
    assert suggested.confidence_band(suggested.HIGH_CONF) == "High"
    assert suggested.confidence_band(0.75) == "High"
    assert suggested.confidence_band(suggested.MED_CONF) == "Medium"
    assert suggested.confidence_band(0.49) == "Medium"
    assert suggested.confidence_band(0.29) == "Low"
    assert suggested.confidence_band(0.0) == "Low"


def test_provider_failure_handled_gracefully(monkeypatch, tmp_path):
    db_file = tmp_path / "suggested.db"
    secs = [{"file": "notes.pdf", "page": 1, "section": "Intro", "text": "word " * 25}]
    monkeypatch.setattr(suggested, "get_file_sections", _fake_sections(secs))

    def failing_llm(_prompt):
        raise RuntimeError("Ollama connection refused")

    rows, err = suggested.get_or_generate_suggestions(
        ["notes.pdf"], llm=failing_llm, score_fn=lambda q, **kw: 0.5, db_path=db_file,
    )
    assert rows == []
    assert err is not None and "Ollama connection refused" in err
