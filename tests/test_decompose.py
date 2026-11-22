"""Tests for decompose.py: skip rule, LLM split, validation fallbacks, cache.

No network, no real LLM: the LLM is a fake callable. Run:
    python -m pytest tests/test_decompose.py -v
"""
import logging
import pytest

import decompose as decompose_mod
from decompose import decompose

SPLIT_INDEPENDENT = '{"subqueries": ["how do ribosomes make proteins", "Tagverse CRM lead pipeline stages"], "answer_as_parts": true}'
SPLIT_COMPARE = '{"subqueries": ["mitosis", "meiosis"], "answer_as_parts": false}'
SPLIT_SINGLE = '{"subqueries": ["explain how ribosomes make proteins"], "answer_as_parts": false}'


@pytest.fixture(autouse=True)
def _fresh_cache_and_env(monkeypatch):
    decompose_mod.clear_cache()
    monkeypatch.delenv("DECOMPOSE", raising=False)
    monkeypatch.delenv("DECOMPOSE_MAX_SUBQUERIES", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    yield
    decompose_mod.clear_cache()


def test_short_simple_question_skipped_without_llm():
    calls = []
    result = decompose("what is mitosis", llm=lambda *a, **k: calls.append(1))
    assert result.subqueries == ["what is mitosis"]
    assert result.reason == "skipped_simple"
    assert result.used_llm is False
    assert result.answer_as_parts is False
    assert calls == []  # the LLM was never called


def test_long_question_without_signals_skipped():
    calls = []
    q = "explain the full process of " + "photosynthesis " * 10
    result = decompose(q, llm=lambda *a, **k: calls.append(1))
    assert result.reason == "skipped_simple"
    assert calls == []


def test_two_independent_topics_split_as_parts():
    calls = []
    def fake_llm(prompt, max_tokens=200):
        calls.append(prompt)
        return SPLIT_INDEPENDENT

    result = decompose(
        "Explain how ribosomes make proteins, and what the Tagverse CRM lead "
        "pipeline does step by step for new sales reps",
        llm=fake_llm)
    assert len(result.subqueries) == 2
    assert result.answer_as_parts is True
    assert result.used_llm is True
    assert result.reason == "split"
    assert len(calls) == 1


def test_difference_question_compares_not_parts():
    result = decompose(
        "What is the difference between mitosis and meiosis in terms of "
        "chromosome numbers during the division phases",
        llm=lambda prompt, max_tokens=200: SPLIT_COMPARE)
    assert result.subqueries == ["mitosis", "meiosis"]
    assert result.answer_as_parts is False
    assert result.reason == "split"


def test_single_topic_llm_answer_returns_one_subquery():
    result = decompose(
        "explain how ribosomes make proteins and also what happens when they "
        "misfold, in detail please",
        llm=lambda prompt, max_tokens=200: SPLIT_SINGLE)
    assert result.subqueries == ["explain how ribosomes make proteins"]
    assert result.answer_as_parts is False


@pytest.mark.parametrize("raw", [
    "not json at all",
    '{"subqueries": [], "answer_as_parts": true}',                       # empty
    '{"subqueries": [1, 2], "answer_as_parts": true}',                   # non-strings
    '{"subqueries": ["ab", "cd"], "answer_as_parts": true}',             # too short
    '{"subqueries": ["same query here", "same  query HERE"], "answer_as_parts": true}',  # near-dupe
    '{"subqueries": ["a b", "a b"], "answer_as_parts": true}',           # exact dupe (also short)
    '{"answer_as_parts": true}',                                          # missing key
    '[1, 2, 3]',                                                          # not a dict
])
def test_invalid_llm_output_falls_back(raw):
    long_q = ("compare the Tagverse CRM lead pipeline with ribosomal protein "
              "synthesis and also note where they differ, with references")
    result = decompose(long_q, llm=lambda prompt, max_tokens=200: raw)
    assert result.subqueries == [long_q]
    assert result.answer_as_parts is False
    assert result.reason == "fallback_invalid"


def test_code_fenced_json_is_parsed():
    raw = '```json\n{"subqueries": ["a b c", "d e f"], "answer_as_parts": true}\n```'
    long_q = ("compare the Tagverse CRM lead pipeline with ribosomal protein "
              "synthesis and also note where they differ, with references")
    result = decompose(long_q, llm=lambda prompt, max_tokens=200: raw)
    assert result.subqueries == ["a b c", "d e f"]
    assert result.reason == "split"


def test_llm_exception_falls_back():
    def boom(prompt, max_tokens=200):
        raise RuntimeError("api down")
    long_q = ("compare the Tagverse CRM lead pipeline with ribosomal protein "
              "synthesis and also note where they differ, with references")
    result = decompose(long_q, llm=boom)
    assert result.subqueries == [long_q]
    assert result.reason == "fallback_error"
    assert result.used_llm is False


def test_more_than_max_subqueries_truncated(monkeypatch):
    raw = '{"subqueries": ["one topic here", "two topic here", "three topic here", "four topic here"], "answer_as_parts": true}'
    result = decompose(
        "explain one, two, three and also four separate topics in detail now",
        llm=lambda prompt, max_tokens=200: raw)
    assert len(result.subqueries) == decompose_mod.MAX_SUBQUERIES if hasattr(decompose, "MAX_SUBQUERIES") else len(result.subqueries) == 3
    assert len(result.subqueries) == 3


def test_decompose_off_never_calls_llm(monkeypatch):
    monkeypatch.setenv("DECOMPOSE", "off")
    calls = []
    long_q = ("compare the Tagverse CRM lead pipeline with ribosomal protein "
              "synthesis and also note where they differ, with references")
    result = decompose(long_q, llm=lambda *a, **k: calls.append(1))
    assert result.subqueries == [long_q]
    assert result.reason == "disabled"
    assert calls == []


def test_cache_same_question_one_llm_call():
    calls = []
    def fake_llm(prompt, max_tokens=200):
        calls.append(1)
        return SPLIT_INDEPENDENT
    q = ("Explain how ribosomes make proteins, and what the Tagverse CRM lead "
         "pipeline does step by step for new sales reps")
    r1 = decompose(q, llm=fake_llm)
    r2 = decompose(q, llm=fake_llm)
    r3 = decompose("  " + q.upper() + "  ", llm=fake_llm)  # normalized -> cache hit
    assert len(calls) == 1
    assert r1.subqueries == r2.subqueries == r3.subqueries


def test_decompose_never_raises_and_never_logs_question(monkeypatch, caplog):
    def evil(prompt, max_tokens=200):
        raise ValueError("SECRET-QUESTION-TEXT leaked")
    long_q = ("compare the Tagverse CRM lead pipeline with ribosomal protein "
              "synthesis and also note where they differ, with references")
    with caplog.at_level(logging.DEBUG, logger="decompose"):
        result = decompose(long_q, llm=evil)
    assert result.reason == "fallback_error"
    all_logs = " ".join(r.getMessage() for r in caplog.records)
    assert "SECRET-QUESTION-TEXT" not in all_logs and long_q not in all_logs
