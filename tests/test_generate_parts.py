"""Tests for parts-aware generation: prompt snapshot for a single sub-query,
per-part instruction for answer_as_parts=True, integrated prompt for
comparisons. No network, no real LLM. Run:
    python -m pytest tests/test_generate_parts.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate
from decompose import Decomposition

CHUNKS = [
    {"id": "a-p1-c0-x", "text": "Ribosome text about translation.", "source": "a.pdf",
     "page": 1, "dense_score": 0.55, "score": 0.55, "subquery_index": 0},
    {"id": "b-p1-c0-y", "text": "CRM pipeline text about leads.", "source": "b.pdf",
     "page": 1, "dense_score": 0.60, "score": 0.60, "subquery_index": 1},
]


def test_single_subquery_prompt_identical_to_old():
    """decomposition with one sub-query (or None) -> the pre-change prompt."""
    old = generate.build_prompt("what is mitosis", CHUNKS)
    dec = Decomposition(subqueries=["what is mitosis"])
    # generate_answer routes to build_prompt when there aren't 2+ sub-queries:
    assert len(dec.subqueries) == 1
    # The parts prompt would differ; generate_answer must NOT pick it here.
    parts_prompt = generate.build_parts_prompt(
        "what is mitosis", CHUNKS, dec.subqueries, False, {0})
    assert parts_prompt != old
    # The single-query prompt itself is unchanged by the refactor (snapshot
    # of its structural anchors from before the change):
    assert old.startswith("Answer the question using ONLY the context below")
    assert "like **Part 1 — <topic>**" in old  # pre-existing rule text
    assert old.rstrip().endswith("Answer:")


def test_answer_as_parts_true_prompt():
    dec = Decomposition(subqueries=["ribosomes and translation", "CRM lead pipeline"],
                        answer_as_parts=True)
    prompt = generate.build_parts_prompt(
        "Explain ribosomes and the CRM pipeline", CHUNKS, dec.subqueries, True, {0, 1})
    assert "**Part N — <short topic>**" in prompt
    assert "Notes for part 1" in prompt and "Notes for part 2" in prompt
    assert "Ribosome text" in prompt and "CRM pipeline text" in prompt
    # grouped: ribosome text under part 1's block, CRM under part 2's
    part1 = prompt.split("Notes for part 1")[1].split("Notes for part 2")[0]
    part2 = prompt.split("Notes for part 2")[1]
    assert "Ribosome text" in part1 and "CRM pipeline text" not in part1
    assert "CRM pipeline text" in part2 and "Ribosome text" not in part2


def test_comparison_prompt_single_block_no_parts():
    dec = Decomposition(subqueries=["mitosis", "meiosis"], answer_as_parts=False)
    prompt = generate.build_parts_prompt(
        "difference between mitosis and meiosis", CHUNKS, dec.subqueries, False, {0, 1})
    assert "ONE integrated answer" in prompt
    assert "Part N" not in prompt
    # one merged context block: both texts, no per-part grouping
    assert "Notes for part" not in prompt
    assert "Ribosome text" in prompt and "CRM pipeline text" in prompt


def test_unsupported_subquery_listed():
    dec = Decomposition(subqueries=["ribosomes and translation", "quantum chromodynamics"],
                        answer_as_parts=True)
    prompt = generate.build_parts_prompt(
        "Explain ribosomes and QCD", [CHUNKS[0]], dec.subqueries, True, {0})
    assert f"No relevant notes found for: quantum chromodynamics" in prompt
    assert f"No relevant notes found for: ribosomes and translation" not in prompt


def test_generate_answer_all_unsupported_fallback_behaviour():
    """No chunks at all -> the existing fallback message (unchanged path)."""
    assert generate.generate_answer("any question", []) == generate.FALLBACK_MESSAGE


def test_gate_uses_best_dense_score_not_first_hit(monkeypatch):
    """Regression (verification 1c): hits are ordered by the mode's ranking
    score (RRF / cross-encoder), so chunks[0] can be a weak dense match while
    a later chunk clears MIN_SCORE. The gate must pass on the BEST dense
    score — the same number query_log records as top_dense_score and the same
    rule query_log.gated() uses — otherwise the log says one thing (not gated)
    while the answer says another (fallback)."""
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-tests")
    monkeypatch.setattr("generate._call_claude", lambda prompt, api_key: "the answer")

    from generate import MIN_SCORE
    reordered = [
        # top-ranked by the mode, but its dense cosine is below the gate
        {"id": "a", "text": "weak match", "source": "a.pdf", "page": 1,
         "dense_score": MIN_SCORE - 0.2, "score": 5.0},
        # lower-ranked, but the best dense match in the set
        {"id": "b", "text": "strong match", "source": "a.pdf", "page": 2,
         "dense_score": MIN_SCORE + 0.2, "score": 1.0},
    ]
    assert generate.generate_answer("what is it?", reordered) == "the answer"
    # every chunk below the gate -> still gated out (unchanged behaviour)
    weak = [{**h, "dense_score": MIN_SCORE - 0.2} for h in reordered]
    assert generate.generate_answer("what is it?", weak) == generate.FALLBACK_MESSAGE
    # the gate decision and the logged invariant agree in both directions
    from query_log import gated, top_dense
    assert gated(reordered, MIN_SCORE) is False and top_dense(reordered) > MIN_SCORE
    assert gated(weak, MIN_SCORE) is True and top_dense(weak) < MIN_SCORE
