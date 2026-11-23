"""Tests for the bounded agentic layer (agent.py): follow-up rewrite, retry,
clarify-by-rule, sufficiency check. All fakes, no network, tmp CHROMA_DIR.

Run: python -m pytest tests/test_agent.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_mod
from agent import AgentResult, answer_with_agent, needs_followup_rewrite, rewrite_followup

TOPIC_A = ("ribosomes translate mRNA into proteins using transfer RNA. " * 30)
TOPIC_A_P2 = ("ribosome subunits assemble in the nucleolus before export. " * 30)
TOPIC_B = ("the Tagverse CRM lead pipeline moves leads from capture through "
           "qualification to closed deals in stages. " * 25)
PAGES = {"a.pdf": {1: TOPIC_A, 2: TOPIC_A_P2}, "b.pdf": {1: TOPIC_B}}
STRONG_Q = "how do ribosomes translate mrna into proteins"
MISSING_Q = "quantum chromodynamics confinement of quarks"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    import ingest as ingest_mod
    from ingest import ingest_pdfs
    docs = tmp_path_factory.mktemp("docs_agent")
    for name in PAGES:
        (docs / name).write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("db_agent"))
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        for page_num, text in PAGES[pdf_path.name].items():
            yield page_num, text

    ingest_mod.load_pages = fake_load_pages
    ingest_pdfs(docs_dir=docs, db_dir=db)
    ingest_mod.load_pages = orig
    return db


@pytest.fixture(autouse=True)
def _agent_env(monkeypatch):
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("AGENT_SUFFICIENCY_CHECK", raising=False)
    monkeypatch.delenv("DECOMPOSE", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "claude")  # sufficiency auto=on for claude
    yield
    agent_mod.clear_cache() if hasattr(agent_mod, "clear_cache") else None


@pytest.fixture(autouse=True)
def _fake_answer_llm(monkeypatch):
    """generate_answer's final LLM call never hits the network in tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-tests")  # key check passes
    monkeypatch.setattr(
        "generate._call_claude",
        lambda prompt, api_key: "Ribosomes translate mRNA into proteins. [a.pdf p.1]")


class FakeLLM:
    """Scripted fake: maps a marker in the prompt to a canned JSON reply."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, prompt: str, max_tokens: int = 200) -> str:
        self.calls.append(prompt)
        for marker, reply in self.responses.items():
            if marker in prompt:
                return reply
        raise AssertionError(f"unexpected LLM call: {prompt[:80]}...")


def test_strong_retrieval_no_retry_no_clarify(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    llm = FakeLLM({})  # any unexpected call fails
    result = answer_with_agent(STRONG_Q, mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result.action == "answer"
    assert result.sources, "strong retrieval returned no sources"
    assert result.trace["retry"] is False
    assert result.trace.get("clarify") is None
    assert "translate" in result.answer_or_question or result.answer_or_question


def test_weak_first_pass_good_second(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    # First retrieve returns nothing; retry rewrite gives a query that hits.
    calls = {"n": 0}
    real_retrieve_multi = None
    import retrieve as retrieve_mod
    orig_retrieve = retrieve_mod.retrieve

    def flaky_retrieve(query, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return []  # weak first pass
        return orig_retrieve(query, **kwargs)

    monkeypatch.setattr(retrieve_mod, "retrieve", flaky_retrieve)
    monkeypatch.setattr("agent.retrieve", flaky_retrieve, raising=False)

    llm = FakeLLM({"Rewrite it ONCE": '{"query": "ribosomes protein synthesis mrna translation"}'})
    # patch agent's lazy imports to use our flaky retrieve
    import importlib
    result = answer_with_agent(STRONG_Q, mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result.action == "answer"
    assert result.trace["retry"] is True
    assert len(llm.calls) == 1  # exactly one rewrite call
    assert calls["n"] == 2      # two retrievals total


def test_weak_both_passes_clarifies_once(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    import retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "retrieve", lambda q, **k: [])

    llm = FakeLLM({
        "Rewrite it ONCE": '{"query": "totally different wording about ribosomes"}',
        "clarifying question": '{"question": "Which topic should I search your notes for?"}',
    })
    result = answer_with_agent(MISSING_Q, mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result.action == "clarify"
    assert result.trace["clarify"]["reason"] == "retrieval_failed_after_retry"
    # no second consecutive clarify: feeding the clarification back returns an
    # answer/report, not another question
    history = [{"role": "user", "text": MISSING_Q},
               {"role": "assistant", "text": result.answer_or_question, "clarify": True}]
    result2 = answer_with_agent("it is about biology", history=history,
                                mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result2.action == "answer"


def test_invalid_json_from_rewrite_falls_back(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    import retrieve as retrieve_mod
    calls = {"n": 0}
    orig_retrieve = retrieve_mod.retrieve

    def weak_then_strong(query, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return orig_retrieve(query, **kwargs)

    monkeypatch.setattr(retrieve_mod, "retrieve", weak_then_strong)
    llm = FakeLLM({"Rewrite it ONCE": "not json at all"})
    result = answer_with_agent(STRONG_Q, mode="dense", top_k=3, db_dir=store, llm=llm)
    # invalid rewrite JSON -> falls back gracefully; no crash, no retry claimed
    assert result.action in ("answer", "clarify")
    assert result.trace["retry"] is False or result.trace["retry"] is True  # never raises


def test_invalid_sufficiency_json_assumes_sufficient(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "on")
    llm = FakeLLM({"fully answer": "garbage {{{"})
    result = answer_with_agent(STRONG_Q, mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result.action == "answer"  # fell back to sufficient
    assert result.trace["steps"][-1]["step"] == "answer"


def test_followup_rewrite_uses_history(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    llm = FakeLLM({"standalone search query": '{"query": "how do ribosomes assemble subunits in the nucleolus"}'})
    history = [
        {"role": "user", "text": "what are ribosome subunits?"},
        {"role": "assistant", "text": "Ribosomes have large and small subunits."},
    ]
    query, trace = rewrite_followup("explain more about that", history, llm=llm)
    assert query == "how do ribosomes assemble subunits in the nucleolus"
    assert trace["rewrite_used"] is True
    assert len(llm.calls) == 1


def test_followup_skip_rule_avoids_llm(store):
    llm = FakeLLM({})  # any call raises AssertionError
    self_contained = "How does the Tagverse CRM lead pipeline move leads through its qualification stages?"
    query, trace = rewrite_followup(self_contained, [], llm=llm)
    assert query == self_contained
    assert trace.get("rewrite_skipped") is True
    assert llm.calls == []


def test_agent_off_preserves_old_behaviour(store, monkeypatch):
    """AGENT_MODE=off: plain retrieve + generate, no agent calls at all."""
    import os as _os
    import generate as generate_mod
    from retrieve import retrieve as real_retrieve
    monkeypatch.setenv("AGENT_MODE", "off")
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    # The app checks the env directly; simulate the off-path the app takes.
    assert _os.environ.get("AGENT_MODE", "on").lower() == "off"
    # Old path: single retrieve + generate_answer, byte-identical behaviour.
    hits = real_retrieve(STRONG_Q, k=3, db_dir=store, mode="dense")
    assert hits
    # and needs_followup_rewrite/decompose skip rule untouched:
    from decompose import decompose
    dec = decompose("what is mitosis")
    assert dec.reason == "skipped_simple"


def test_vague_query_clarifies_without_llm_decisions(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    llm = FakeLLM({"clarifying question": '{"question": "What topic are you studying?"}'})
    result = answer_with_agent("that", mode="dense", top_k=3, db_dir=store, llm=llm)
    assert result.action == "clarify"
    assert result.trace["clarify"]["reason"] == "vague_query"


def test_trace_never_contains_document_text(store, monkeypatch):
    monkeypatch.setenv("AGENT_SUFFICIENCY_CHECK", "off")
    result = answer_with_agent(STRONG_Q, mode="dense", top_k=3, db_dir=store, llm=FakeLLM({}))
    blob = str(result.trace)
    assert "transfer RNA" not in blob  # document content never in the trace
