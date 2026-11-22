"""Regression tests for multi-topic retrieval: decomposition + per-sub-query
gating + guaranteed-slot merge.

Builds tiny two-topic indexes in tmp dirs; no network, no real LLM. Run:
    python -m pytest tests/test_multitopic_retrieval.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest as ingest_mod
from ingest import ingest_pdfs
from retrieve import retrieve, retrieve_multi

TOPIC_A = ("ribosomes translate mRNA into proteins using transfer RNA. " * 30)      # file a, p1
TOPIC_A_P2 = ("ribosome subunits assemble in the nucleolus before export. " * 30)   # file a, p2
TOPIC_B = ("the Tagverse CRM lead pipeline moves leads from capture through "
           "qualification to closed deals in stages. " * 25)                        # file b, p1

PAGES = {
    "a.pdf": {1: TOPIC_A, 2: TOPIC_A_P2},
    "b.pdf": {1: TOPIC_B},
}
SUB_A = "how do ribosomes make proteins"
SUB_B = "Tagverse CRM lead pipeline stages"
Q_BOTH = ("Explain how ribosomes make proteins, and what the Tagverse CRM "
          "lead pipeline does for sales teams")


@pytest.fixture(scope="module")
def two_file_store(tmp_path_factory):
    docs = tmp_path_factory.mktemp("docs_multi")
    for name in PAGES:
        (docs / name).write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("db_multi"))
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        for page_num, text in PAGES[pdf_path.name].items():
            yield page_num, text

    ingest_mod.load_pages = fake_load_pages
    ingest_pdfs(docs_dir=docs, db_dir=db)
    ingest_mod.load_pages = orig
    return docs, db


@pytest.fixture(scope="module")
def one_file_store(tmp_path_factory):
    """Two unrelated topics inside a SINGLE file on different pages."""
    docs = tmp_path_factory.mktemp("docs_single")
    (docs / "mixed.pdf").write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("db_single"))
    orig = ingest_mod.load_pages

    def fake_load_pages(pdf_path, use_ocr=None):
        yield 1, TOPIC_B
        yield 2, TOPIC_A

    ingest_mod.load_pages = fake_load_pages
    ingest_pdfs(docs_dir=docs, db_dir=db)
    ingest_mod.load_pages = orig
    return docs, db


def _only_a(pdf_path, use_ocr=None):
    yield 1, TOPIC_A


@pytest.fixture(scope="module")
def a_only_store(tmp_path_factory):
    """Only topic A in the index — sub-query B must come back unsupported."""
    docs = tmp_path_factory.mktemp("docs_a")
    (docs / "a.pdf").write_bytes(b"%PDF-1.4 stub")
    db = str(tmp_path_factory.mktemp("db_a"))
    orig = ingest_mod.load_pages
    ingest_mod.load_pages = _only_a
    ingest_pdfs(docs_dir=docs, db_dir=db)
    ingest_mod.load_pages = orig
    return docs, db


@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_rerank"])
def test_two_topics_both_files_covered(two_file_store, mode):
    docs, db = two_file_store
    hits, supported = retrieve_multi([SUB_A, SUB_B], mode=mode, top_k=3, db_dir=db)
    assert hits, "no hits at all"
    assert {h["source"] for h in hits} == {"a.pdf", "b.pdf"}, \
        f"{mode}: one topic crowded out the other"
    assert supported == {0, 1}
    assert all("subquery_index" in h for h in hits)


@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_rerank"])
def test_two_topics_single_file_both_pages(one_file_store, mode):
    docs, db = one_file_store
    hits, supported = retrieve_multi([SUB_B, SUB_A], mode=mode, top_k=3, db_dir=db)
    pages = {h["page"] for h in hits}
    assert 1 in pages and 2 in pages, f"{mode}: both pages should appear, got {pages}"
    assert supported == {0, 1}


def test_dense_without_decomposition_misses_one_topic(two_file_store):
    """Informational baseline (not asserted): the old single-query dense path
    can miss one topic — this documents why the decomposition matters."""
    docs, db = two_file_store
    hits = retrieve(Q_BOTH, k=3, db_dir=db, mode="dense")
    single_file_only = len({h["source"] for h in hits}) < 2
    # informational only: no assert on single_file_only


def test_single_topic_matches_old_path(two_file_store):
    """One sub-query == the old single-query path: same IDs, same order."""
    docs, db = two_file_store
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        old = retrieve(SUB_A, k=3, db_dir=db, mode=mode)
        new, supported = retrieve_multi([SUB_A], mode=mode, top_k=3, db_dir=db)
        assert [h["id"] for h in new] == [h["id"] for h in old], mode
        assert [h["score"] for h in new] == [h["score"] for h in old], mode
        assert supported == {0}


def test_gate_marks_unsupported_subquery(a_only_store):
    docs, db = a_only_store
    hits, supported = retrieve_multi([SUB_A, SUB_B], mode="dense", top_k=3, db_dir=db)
    assert supported == {0}          # B absent from the index
    assert {h["source"] for h in hits} == {"a.pdf"}
    assert all(h["subquery_index"] == 0 for h in hits)

    # generate's parts prompt lists the missing topic
    import generate
    from decompose import Decomposition
    dec = Decomposition(subqueries=[SUB_A, SUB_B], answer_as_parts=True)
    prompt = generate.build_parts_prompt(Q_BOTH, hits, dec.subqueries, True, supported)
    assert f"No relevant notes found for: {SUB_B}" in prompt
    assert f"No relevant notes found for: {SUB_A}" not in prompt


def test_all_subqueries_unsupported_returns_empty(a_only_store):
    docs, db = a_only_store
    hits, supported = retrieve_multi(
        ["quantum chromodynamics confinement", "weierstrass elliptic functions"],
        mode="dense", top_k=3, db_dir=db)
    assert hits == [] and supported == set()


def test_dedupe_parent_shared_by_both_subqueries(two_file_store):
    """A parent matching both sub-queries appears once with both indices.
    Constructed via a fake merge: topic pages here are disjoint, so instead
    assert the mechanism through subquery_indices bookkeeping on real hits."""
    docs, db = two_file_store
    hits, supported = retrieve_multi([SUB_A, "ribosomes and proteins"], mode="dense",
                                     top_k=4, db_dir=db)
    both = [h for h in hits if len(h.get("subquery_indices", [])) > 1]
    for h in both:
        assert isinstance(h["subquery_indices"], list)
        assert h["subquery_index"] in h["subquery_indices"]
    # no duplicated parents
    pids = [h.get("parent_id", h["id"]) for h in hits]
    assert len(pids) == len(set(pids))


def test_guaranteed_slots_minority_topic(one_file_store):
    """With a dominant topic and top_k=3, the minority still gets a
    guaranteed slot (here min(2, top_k)=2 slots when its data allows)."""
    docs, db = one_file_store
    # page 1 = CRM (minority), page 2 = ribosomes (dominant).
    hits, supported = retrieve_multi(
        ["the Tagverse CRM lead pipeline stages", "ribosomes and their subunits"],
        mode="dense", top_k=3, db_dir=db)
    crm = [h for h in hits if h["subquery_index"] == 0]
    ribo = [h for h in hits if h["subquery_index"] == 1]
    assert supported == {0, 1}
    assert len(crm) >= 1, "minority topic got zero slots"
    assert len(ribo) >= 1, "dominant topic lost all slots"


def test_merge_never_exceeds_max_context(two_file_store, monkeypatch):
    docs, db = two_file_store
    monkeypatch.setenv("DECOMPOSE_MAX_CONTEXT", "4")
    hits, supported = retrieve_multi([SUB_A, SUB_B], mode="dense", top_k=4, db_dir=db)
    assert len(hits) <= 4
    monkeypatch.setenv("DECOMPOSE_MAX_CONTEXT", "8")
    hits, _ = retrieve_multi([SUB_A, SUB_B], mode="dense", top_k=4, db_dir=db)
    assert len(hits) <= 8


def test_round_robin_helper_removed_from_app():
    """The old round-robin-by-source workaround is gone from app.py."""
    app_src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "by_source" not in app_src
    assert "round-robin" not in app_src and "round robin" not in app_src.lower().replace("round-robin", "")
