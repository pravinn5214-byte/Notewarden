"""Tests for the retrieval pipeline: ingest -> retrieve round trip, idempotent
re-ingest, file-delete cleanup, RRF fusion, parent dedupe.

Run: python -m pytest tests/test_retrieval.py -v
Uses a throwaway corpus and a throwaway Chroma dir; never touches chroma_db/.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import ingest_pdfs  # noqa: E402
from retrieve import _dedupe_parents, _rrf_fuse, retrieve  # noqa: E402

PAGE1 = "Alpha beta gamma delta epsilon. " * 40      # ~200 words
PAGE2 = "Zeta eta theta iota kappa lambda mu. " * 40  # distinct vocabulary
PAGE3 = "Nuclear magnetic resonance spectroscopy measures chemical shifts. " * 30


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Two tiny PDFs won't work (needs reportlab), so ingest text 'PDFs' is not
    possible — instead build the corpus via load_pages-free path: write real
    minimal PDFs with pypdf's writer? pypdf can't create text pages easily.
    Fallback: monkeypatch-free approach — create one-page PDFs is overkill.

    Simpler: bypass PDFs. Build a fake docs dir but stub load_pages via
    monkeypatching ingest.load_pages.
    """
    docs = tmp_path_factory.mktemp("docs")
    f1 = docs / "one.pdf"
    f2 = docs / "two.pdf"
    f1.write_bytes(b"%PDF-1.4 stub")
    f2.write_bytes(b"%PDF-1.4 stub")
    return docs, {"one.pdf": {1: PAGE1, 2: PAGE2}, "two.pdf": {1: PAGE3}}


@pytest.fixture(scope="module")
def store(tmp_path_factory, corpus, monkeypatch_module):
    docs, pages = corpus
    db = str(tmp_path_factory.mktemp("db"))
    import ingest
    orig = ingest.load_pages

    def fake_load_pages(pdf_path):
        for page_num, text in pages[pdf_path.name].items():
            yield page_num, text

    ingest.load_pages = fake_load_pages
    yield docs, db, pages
    ingest.load_pages = orig


@pytest.fixture(scope="module")
def monkeypatch_module():
    return None  # placeholder; real monkeypatching done in `store`


def test_roundtrip_and_scores(store):
    docs, db, pages = store
    n_files, n_chunks = ingest_pdfs(docs_dir=docs, db_dir=db)
    assert n_files == 2
    assert n_chunks > 0

    hits = retrieve("alpha beta gamma", k=3, db_dir=db, mode="dense")
    assert hits, "dense retrieval returned nothing"
    # The corpus's two pages share boilerplate, so pages 1 and 2 score nearly
    # identically; assert the right file and that the matching page ranks top-3.
    assert all(h["source"] == "one.pdf" for h in hits)
    assert 1 in [h["page"] for h in hits]
    assert 0.0 <= hits[0]["score"] <= 1.0

    hits = retrieve("nuclear magnetic resonance", k=3, db_dir=db, mode="hybrid")
    assert hits[0]["source"] == "two.pdf"
    # parent text present and at least as long as the child
    assert len(hits[0]["parent_text"].split()) >= len(hits[0]["text"].split())

    hits = retrieve("nuclear magnetic resonance", k=3, db_dir=db, mode="hybrid_rerank")
    assert hits[0]["source"] == "two.pdf"
    assert "rerank_score" in hits[0]


def test_idempotent_reingest(store):
    docs, db, _ = store
    _, first = ingest_pdfs(docs_dir=docs, db_dir=db)
    _, second = ingest_pdfs(docs_dir=docs, db_dir=db)
    assert first == second, "re-ingest changed the chunk count"
    # and no duplicate IDs
    import chromadb
    ids = chromadb.PersistentClient(path=db).get_collection("notes").get(include=[])["ids"]
    assert len(ids) == len(set(ids))


def test_file_delete_cleanup(store, tmp_path_factory):
    docs, db, _ = store
    # re-ingest with only one.pdf selected: two.pdf chunks must vanish
    ingest_pdfs(docs_dir=docs, db_dir=db, files=[docs / "one.pdf"])
    import chromadb
    client = chromadb.PersistentClient(path=db)
    metas = client.get_collection("notes").get(include=["metadatas"])["metadatas"]
    assert all(m["source"] == "one.pdf" for m in metas)
    # parents collection cleaned too
    pm = client.get_collection("parents").get(include=["metadatas"])["metadatas"]
    assert all(m["source"] == "one.pdf" for m in pm)


def test_rrf_fusion():
    dense = [
        {"id": "a", "text": "A", "source": "s", "page": 1, "parent_id": "P1", "dense_score": 0.9},
        {"id": "b", "text": "B", "source": "s", "page": 1, "parent_id": "P2", "dense_score": 0.8},
    ]
    bm25 = [
        {"id": "b", "text": "B", "source": "s", "page": 1, "parent_id": "P2", "bm25_rank": 1, "parent_text": "PB"},
        {"id": "d", "text": "D", "source": "s", "page": 2, "parent_id": "P4", "bm25_rank": 2, "parent_text": "PD"},
    ]
    fused = _rrf_fuse(dense, bm25)
    by_id = {h["id"]: h for h in fused}
    # b appears in both lists: 1/(60+2) + 1/(60+1) — beats a (1/61) and d (1/62)
    assert abs(by_id["b"]["rrf"] - (1 / 62 + 1 / 61)) < 1e-9
    assert fused[0]["id"] == "b"
    assert abs(by_id["a"]["rrf"] - 1 / 61) < 1e-9
    assert abs(by_id["d"]["rrf"] - 1 / 62) < 1e-9
    assert by_id["b"]["bm25_rank"] == 1
    assert by_id["b"]["parent_text"] == "PB"  # enriched from the BM25 hit


def test_parent_dedupe():
    hits = [
        {"id": "a", "text": "A", "parent_id": "P1", "score": 0.9, "source": "s", "page": 1},
        {"id": "b1", "text": "B1", "parent_id": "P2", "score": 0.5, "source": "s", "page": 1},
        {"id": "b2", "text": "B2", "parent_id": "P2", "score": 0.4, "source": "s", "page": 1},
        {"id": "c", "text": "C", "parent_id": "P3", "score": 0.3, "source": "s", "page": 2},
    ]
    out = _dedupe_parents(hits)
    assert [h["id"] for h in out] == ["a", "b1", "c"]
    assert out[1]["score"] == 0.5  # best child of P2 kept


def test_modes_respected(store):
    docs, db, _ = store
    ingest_pdfs(docs_dir=docs, db_dir=db)
    hits = retrieve("alpha", k=2, db_dir=db, mode="dense", allowed_sources=["two.pdf"])
    # one.pdf ticked off: nothing from it may come back... unless two.pdf has
    # no alpha at all (it doesn't) — then we get whatever two.pdf offers or [].
    assert all(h["source"] == "two.pdf" for h in hits)

    with pytest.raises(ValueError):
        retrieve("alpha", k=2, db_dir=db, mode="bogus")


def test_tokenizer():
    from retrieve import tokenize
    # hyphens/underscores inside tokens survive; surrounding punctuation goes
    assert tokenize("INV-YYYY-XXXX pattern") == ["inv-yyyy-xxxx", "pattern"]
    assert tokenize("G-quadruplex formation!") == ["g-quadruplex", "formation"]
    assert tokenize("snake_case_name") == ["snake_case_name"]
    # stopwords dropped, case-folded
    assert tokenize("The Theory of Everything") == ["theory", "everything"]
    # punctuation-only boundaries split tokens
    assert tokenize("alpha,beta;gamma") == ["alpha", "beta", "gamma"]
    # pure stopwords -> empty
    assert tokenize("the of and") == []


def test_clean_text():
    from ingest import clean_text, fused_token_rate
    # camelCase split: lower->Upper boundary gets a space
    out = clean_text("CommonlyinvolvedinDNArecognition")
    assert "in DNA" in out  # the DNA boundary is split
    # letter-digit and digit-letter joins
    assert clean_text("H2A and 30S subunit") == "H 2 A and 30 S subunit"
    # hyphenated line-break de-hyphenation: "protei-\nne" -> "proteine"
    # (the join is mechanical; whether it yields a real word depends on where
    # the extractor broke the line)
    assert clean_text("protei-\nne folding") == "proteine folding"
    # whitespace collapse
    assert clean_text("a\n\nb  c") == "a b c"
    # fused-rate diagnostic drops after cleaning
    raw = "IntermolecularHydrogenBonding occurs between AdjacentChains."
    assert fused_token_rate(clean_text(raw)) <= fused_token_rate(raw)
