"""Tests for heading-aware chunking (Step 3): detection, section splitting,
window fallback, metadata, mode-tagged IDs, and mode-switch/repair cleanup.

Run: python -m pytest tests/test_heading_chunking.py -v
No network; throwaway Chroma dir via conftest. Real-PDF detection is checked
against docs/*.pdf when present (skipped otherwise).
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest  # noqa: E402
from ingest import (  # noqa: E402
    CHUNKING_HEADING, CHUNKING_WINDOW, _chunk_id, _line_is_heading,
    _normalize_chunking, _with_section, chunking_mode, detect_page_headings,
    ingest_pdfs, plan_page_chunks, split_sections,
)

# --- Mode selection ------------------------------------------------------------

def test_default_chunking_is_window(monkeypatch):
    monkeypatch.delenv("CHUNKING", raising=False)
    assert chunking_mode() == CHUNKING_WINDOW == "window"


def test_chunking_env_switches_to_heading(monkeypatch):
    monkeypatch.setenv("CHUNKING", "heading")
    assert chunking_mode() == "heading"


def test_unknown_chunking_falls_back_to_window(monkeypatch):
    monkeypatch.setenv("CHUNKING", "bogus-mode")
    assert chunking_mode() == "window"
    assert _normalize_chunking("bogus") == "window"
    assert _normalize_chunking(None) == "window"   # env is bogus too
    monkeypatch.setenv("CHUNKING", "HEADING")      # case-insensitive
    assert _normalize_chunking(None) == "heading"
    assert _normalize_chunking(" window ") == "window"


# --- Heading detection ---------------------------------------------------------

class _FakePage:
    """Minimal PyMuPDF page double: get_text('dict') with hand-built spans."""

    def __init__(self, lines):
        # lines: [(text, size, flags), ...] — one span per line, body = lines[0]
        self._lines = lines

    def get_text(self, kind):
        assert kind == "dict"
        return {"blocks": [{"type": 0, "lines": [
            {"spans": [{"text": t, "size": float(s), "flags": int(f),
                        "font": "Helvetica"}]}
            for t, s, f in self._lines]}]}


def test_bigger_font_line_is_heading():
    page = _FakePage([("Body text line one.", 10.0, 0),
                      ("Big Section Title", 16.0, 0)])
    assert detect_page_headings(page) == ["Big Section Title"]


def test_numbered_and_colon_labels_detected_at_body_size():
    page = _FakePage([("Body text line one.", 12.0, 0),
                      ("1. Introduction", 12.0, 16),      # bold + numbered
                      ("Features:", 12.0, 0)])            # colon label
    heads = detect_page_headings(page)
    assert "1. Introduction" in heads
    assert "Features:" in heads


def test_plain_body_lines_are_not_headings():
    page = _FakePage([("Just an ordinary sentence of body text.", 10.0, 0),
                      ("Another ordinary sentence here.", 10.0, 0)])
    assert detect_page_headings(page) == []


def test_table_cell_labels_are_not_headings():
    # Checklist table headers: short 'Phase 1' / 'Phase Name' cells must not
    # become section boundaries (regression: Tagverse checklist PDF).
    assert not _line_is_heading("Phase 1", 10.0, 16, 10.0)
    assert not _line_is_heading("Phase Name", 10.0, 16, 10.0)
    assert not _line_is_heading("PHASE", 10.0, 16, 10.0)
    # ...but a real prefixed heading still is.
    assert _line_is_heading("Phase 1: Foundation, Multi-Tenant Setup",
                            10.0, 16, 10.0)
    assert _line_is_heading("SECTION A – 2 MARK QUESTIONS", 10.0, 16, 10.0)


def test_oversized_line_is_not_heading():
    long_line = "1. " + " ".join(["word"] * 40)   # > MAX_HEADING_CHARS
    assert not _line_is_heading(long_line, 10.0, 16, 10.0)


def test_headingless_page_returns_empty():
    # Pages with no text layer (OCR-only) yield [] -> window fallback.
    class _NoText:
        def get_text(self, kind):
            return {"blocks": []}
    assert detect_page_headings(_NoText()) == []


# --- Section splitting ---------------------------------------------------------

PAGE = ("Intro words before any heading. "
        "Alpha Section First body has several words of content here. "
        "Beta Section Second body also has several words of content there.")


def test_split_sections_splits_at_headings():
    parts = split_sections(PAGE, ["Alpha Section", "Beta Section"])
    assert len(parts) == 3
    assert parts[0][0] is None                      # untitled lead-in
    assert parts[0][1].startswith("Intro words")
    assert parts[1][0] == "Alpha Section"
    assert parts[1][1].startswith("Alpha Section")
    assert parts[2][0] == "Beta Section"
    assert parts[2][1].startswith("Beta Section")
    # No section may contain another section's heading.
    assert "Beta Section" not in parts[1][1]


def test_split_sections_without_headings_is_window_fallback():
    assert split_sections(PAGE, None) == [(None, PAGE)]
    assert split_sections(PAGE, []) == [(None, PAGE)]


def test_split_sections_skips_heading_missing_from_text():
    # Cleaning may move/remove a heading; a heading not found is skipped,
    # the rest still split.
    parts = split_sections(PAGE, ["Alpha Section", "Not In The Text"])
    assert [t for t, _ in parts] == [None, "Alpha Section"]


def test_adjacent_table_headings_do_not_create_empty_sections():
    # "Phase | Phase Name | Phase 1" cells sit <3 words apart: the first two
    # are dropped as boundaries, and their words fold into the first real
    # section instead of becoming a 1-3 word sliver chunk.
    text = "Phase Phase Name Phase 1 Foundation setup body words continue here."
    parts = split_sections(text, ["Phase", "Phase Name", "Phase 1"])
    assert len(parts) == 1
    title, body = parts[0]
    assert title == "Phase 1"
    assert len(body.split()) >= 6          # no sliver: all words retained
    assert "Phase Phase Name" in body      # dropped cells' words not lost


def test_title_only_stub_is_folded_into_previous_section():
    # A heading at the very end of a page (content continues next page) must
    # not become a chunk that is only a title.
    text = "Body words of the first section here. Ending Label"
    parts = split_sections(text, ["Body words", "Ending Label"])
    assert all(len(body.split()) > len((title or "").split())
               for title, body in parts)
    assert not any(body.strip() == "Ending Label" for _, body in parts)


# --- plan_page_chunks ----------------------------------------------------------

def test_window_mode_matches_fixed_windows():
    # CHUNKING=window must behave exactly like the original fixed windows.
    parents, children = plan_page_chunks(PAGE, None, 6, 2, 10)
    assert [t for t, _, _ in parents] == list(ingest.chunk_words_page(PAGE, 10))
    assert [t for t, _, _ in children] == list(ingest.chunk_words(PAGE, 6, 2))
    assert all(sec is None for _, sec, _ in parents + children)


def test_heading_mode_chunks_never_cross_a_boundary():
    parents, children = plan_page_chunks(
        PAGE, ["Alpha Section", "Beta Section"], 6, 2, 10)
    for _text, section, _start in parents + children:
        if section == "Alpha Section":
            assert "Beta Section" not in _text
        if section == "Beta Section":
            assert "Alpha Section" not in _text


def test_heading_mode_children_carry_section_title():
    _parents, children = plan_page_chunks(
        PAGE, ["Alpha Section"], 6, 2, 10)
    alpha = [t for t, sec, _ in children if sec == "Alpha Section"]
    assert alpha, "expected children inside the Alpha section"
    # Every chunk of the section starts with its title (BM25/embed see it).
    assert all(t.startswith("Alpha Section") for t in alpha)


def test_with_section_prepends_only_when_missing():
    assert _with_section("Sec", "body text") == "Sec: body text"
    assert _with_section("Sec", "Sec already first") == "Sec already first"
    assert _with_section("Q?", "answer") == "Q? answer"
    assert _with_section(None, "text") == "text"


# --- IDs -----------------------------------------------------------------------

def test_chunk_id_includes_mode_tag():
    a = _chunk_id("doc.pdf", 1, 0, "same text", CHUNKING_WINDOW)
    b = _chunk_id("doc.pdf", 1, 0, "same text", CHUNKING_HEADING)
    assert a != b, "same position under different modes must differ"
    assert "-win-" in a and "-head-" in b
    # Same mode + same text -> stable (idempotent re-ingest).
    assert a == _chunk_id("doc.pdf", 1, 0, "same text", CHUNKING_WINDOW)


# --- Ingest integration --------------------------------------------------------

HEAD_PAGE = ("Alpha Section " + " ".join(["alpha words here"] * 20) + " "
             "Beta Section " + " ".join(["beta words there"] * 20))


@pytest.fixture()
def heading_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.pdf").write_bytes(b"%PDF-1.4 stub")
    return docs


def _run_ingest(docs, db, monkeypatch, *, chunking, headings=None):
    import ingest as ingest_mod
    orig_pages, orig_heads = ingest_mod.load_pages, ingest_mod.load_headings
    ingest_mod.load_pages = lambda pdf_path, use_ocr=None: iter([(1, HEAD_PAGE)])
    ingest_mod.load_headings = lambda pdf_path: {1: headings or []}
    try:
        return ingest_pdfs(docs_dir=docs, db_dir=db, chunking=chunking,
                           use_ocr=False)
    finally:
        ingest_mod.load_pages, ingest_mod.load_headings = orig_pages, orig_heads


def _child_rows(db):
    import chromadb
    col = chromadb.PersistentClient(path=db).get_collection("notes")
    got = col.get(include=["metadatas"])
    return list(zip(got["ids"], got["metadatas"]))


def test_heading_ingest_stamps_chunking_and_section_metadata(
        heading_docs, tmp_path, monkeypatch):
    db = str(tmp_path / "db_head")
    _run_ingest(heading_docs, db, monkeypatch, chunking="heading",
                headings=["Alpha Section"])
    rows = _child_rows(db)
    assert rows, "expected chunks"
    sections = {m.get("section", "") for _i, m in rows}
    assert "Alpha Section" in sections
    assert all(m.get("chunking") == "heading" for _i, m in rows)


def test_window_ingest_keeps_window_metadata(heading_docs, tmp_path, monkeypatch):
    db = str(tmp_path / "db_win")
    _run_ingest(heading_docs, db, monkeypatch, chunking="window",
                headings=["Alpha Section"])
    rows = _child_rows(db)
    assert rows
    assert all(m.get("chunking") == "window" for _i, m in rows)
    assert all(m.get("section", "") == "" for _i, m in rows)


def test_headingless_page_falls_back_to_window_chunks(
        heading_docs, tmp_path, monkeypatch):
    # No detected headings (OCR page / no usable signal) -> identical text
    # chunks to window mode (only the mode tag in the ID differs).
    db_h, db_w = str(tmp_path / "db_h"), str(tmp_path / "db_w")
    _run_ingest(heading_docs, db_h, monkeypatch, chunking="heading", headings=[])
    _run_ingest(heading_docs, db_w, monkeypatch, chunking="window", headings=[])
    import chromadb
    def texts(db):
        col = chromadb.PersistentClient(path=db).get_collection("notes")
        return sorted(col.get(include=["documents"])["documents"])
    assert texts(db_h) == texts(db_w)


def test_mode_switch_replaces_old_mode_chunks(heading_docs, tmp_path, monkeypatch):
    """Re-ingesting under another CHUNKING must leave zero stale chunks of the
    previous mode (IDs are mode-tagged, so the sweep removes them)."""
    db = str(tmp_path / "db_switch")
    _run_ingest(heading_docs, db, monkeypatch, chunking="window",
                headings=["Alpha Section"])
    win_ids = [i for i, _m in _child_rows(db) if "-win-" in i]
    assert win_ids, "precondition: window chunks present"

    _run_ingest(heading_docs, db, monkeypatch, chunking="heading",
                headings=["Alpha Section"])
    rows = _child_rows(db)
    assert rows, "heading chunks must exist after the switch"
    assert all("-head-" in i for i, _m in rows), "stale window chunks remained"
    assert not any("-win-" in i for i, _m in rows)


def test_repair_under_heading_mode_rebuilds_same_ids(
        heading_docs, tmp_path, monkeypatch):
    """App's Repair index (delete both collections + re-ingest) under heading
    mode yields the same chunk IDs as a fresh ingest."""
    import chromadb
    db = str(tmp_path / "db_repair")

    def run():
        return _run_ingest(heading_docs, db, monkeypatch, chunking="heading",
                           headings=["Alpha Section"])

    run()
    fresh = set(_child_rows(db)[i][0] for i in range(len(_child_rows(db))))
    client = chromadb.PersistentClient(path=db)
    for name in ("notes", "parents"):
        client.delete_collection(name)
    import store
    store.reset_client()
    run()
    rows = _child_rows(db)
    repaired = {i for i, _m in rows}
    assert repaired == fresh
    assert all("-head-" in i for i in repaired)


def test_retrieve_exposes_section_field(heading_docs, tmp_path, monkeypatch):
    from retrieve import retrieve
    db = str(tmp_path / "db_sec")
    _run_ingest(heading_docs, db, monkeypatch, chunking="heading",
                headings=["Alpha Section"])
    hits = retrieve("alpha words", k=3, db_dir=db, mode="hybrid")
    assert hits, "expected at least one hit"
    assert all("section" in h for h in hits)
    assert any(h["section"] == "Alpha Section" for h in hits)


# --- Real PDFs -----------------------------------------------------------------

def test_real_pdfs_have_headings_or_flag_zero():
    """B.3 sanity check: every PDF in docs/ either yields headings or is a
    legitimate zero-heading file (then window fallback applies)."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    pdfs = sorted(docs.glob("*.pdf"))
    if not pdfs:
        pytest.skip("no PDFs in docs/")
    for pdf in pdfs:
        heads = ingest.load_headings(pdf)
        if not heads:
            # Acceptable only if the file has no text layer at all.
            import pymupdf
            doc = pymupdf.open(pdf)
            has_text = any((p.get_text() or "").strip() for p in doc)
            doc.close()
            assert not has_text, f"{pdf.name}: text layer present but no headings"
