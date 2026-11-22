"""Tests for OCR: image-only PDF ingestion + retrieval, text-layer PDFs not
OCR'd, OCR cache hits, graceful no-Tesseract behavior.

Run: python -m pytest tests/test_ocr.py -v
Builds a tiny image-only PDF (text rendered onto an image with Pillow) and a
text-layer PDF (PyMuPDF insert_text); throwaway docs/ and Chroma dir, never
touches chroma_db/. OCR tests skip if Tesseract isn't installed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocr  # noqa: E402
import ingest  # noqa: E402
import pymupdf  # noqa: E402
from ingest import ingest_pdfs  # noqa: E402
from retrieve import retrieve  # noqa: E402

OCR_WORD = "quokka burrow"  # unique keyword to retrieve the OCR'd page
TEXT_WORD = "platypus den"  # keyword for the text-layer page

pytestmark_ocr = pytest.mark.skipif(
    not ocr.is_tesseract_available(),
    reason="Tesseract not installed",
)


def make_image_pdf(path: Path, text: str) -> None:
    """A 1-page PDF whose page is just an image of `text` (no text layer)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 80), text, fill="black")
    img.save(path.parent / "_tmp_ocr_img.png")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(path.parent / "_tmp_ocr_img.png"))
    doc.save(path)
    doc.close()
    (path.parent / "_tmp_ocr_img.png").unlink()


def make_text_pdf(path: Path, text: str) -> None:
    """A 1-page PDF with a real text layer."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=14)
    doc.save(path)
    doc.close()


@pytest.fixture(scope="module")
def ocr_store(tmp_path_factory):
    """Image-only PDF + text-layer PDF, ingested into a throwaway store."""
    docs = tmp_path_factory.mktemp("docs_ocr")
    make_image_pdf(docs / "scanned.pdf", f"{OCR_WORD} habitat studies. " * 4)
    make_text_pdf(docs / "textlayer.pdf", f"{TEXT_WORD} burrow ecology. " * 4)
    db = str(tmp_path_factory.mktemp("db_ocr"))
    files, total = ingest_pdfs(docs_dir=docs, db_dir=db)
    return docs, db, files, total


class TestOcr:
    @pytestmark_ocr
    def test_image_pdf_ingests_with_ocr_and_retrievable(self, ocr_store):
        docs, db, files, total = ocr_store
        assert files == 2
        assert total > 0
        # The scanned page is OCR'd: metadata says so.
        import chromadb
        metas = chromadb.PersistentClient(path=db).get_collection("notes").get(
            include=["metadatas"])["metadatas"]
        scanned = [m for m in metas if m["source"] == "scanned.pdf"]
        assert scanned, "scanned.pdf produced no chunks"
        assert all(m["ocr"] for m in scanned)
        textlayer = [m for m in metas if m["source"] == "textlayer.pdf"]
        assert textlayer
        assert not any(m["ocr"] for m in textlayer), "text-layer PDF was OCR'd"
        # And it's retrievable by a keyword from the OCR'd text.
        hits = retrieve(f"{OCR_WORD} habitat", k=3, db_dir=db, mode="dense")
        assert hits and hits[0]["source"] == "scanned.pdf"
        assert hits[0]["ocr"] is True

    @pytestmark_ocr
    def test_reingest_cache_hit_no_rerun(self, ocr_store, monkeypatch):
        docs, db, _, _ = ocr_store
        import ingest
        calls = []
        orig = ocr.cached_ocr_page

        def counting_cached(pdf_path, page_num, page):
            calls.append((pdf_path.name, page_num))
            return orig(pdf_path, page_num, page)

        monkeypatch.setattr(ocr, "cached_ocr_page", counting_cached)
        _, first = ingest_pdfs(docs_dir=docs, db_dir=db)
        _, second = ingest_pdfs(docs_dir=docs, db_dir=db)
        assert first == second, "re-ingest changed the chunk count"
        # First run re-OCRed (cache empty); second run must be a cache hit.
        assert [c for c in calls if c[0] == "scanned.pdf"] == []

    def test_no_tesseract_graceful(self, tmp_path, monkeypatch):
        """With Tesseract unavailable: scanned pages are skipped, text pages
        still ingest, no crash, warning printed."""
        import io
        from contextlib import redirect_stderr

        make_image_pdf(tmp_path / "scan.pdf", f"{OCR_WORD} only pixels")
        make_text_pdf(tmp_path / "txt.pdf", f"{TEXT_WORD} plain text layer")
        db = str(tmp_path / "db")

        class FakeEngine:
            name = "fake"
            def available(self) -> bool:
                return False
            def ocr_page(self, page, dpi=300, lang="eng"):
                raise AssertionError("should not be called")
            def ocr_image(self, image, lang="eng"):
                raise AssertionError("should not be called")

        # Engine selection lives in ingest (re-exported from ocr): patch both.
        fake = FakeEngine()
        monkeypatch.setattr(ocr, "get_engine", lambda name=None: fake)
        monkeypatch.setattr(ingest, "get_engine", lambda name=None: fake)
        monkeypatch.setattr(ocr, "is_tesseract_available", lambda: False)
        err = io.StringIO()
        with redirect_stderr(err):
            files, total = ingest_pdfs(docs_dir=tmp_path, db_dir=db)
        assert files == 2
        # Only the text-layer page has content; the scanned page was skipped.
        import chromadb
        metas = chromadb.PersistentClient(path=db).get_collection("notes").get(
            include=["metadatas"])["metadatas"]
        assert all(m["source"] == "txt.pdf" for m in metas)
        assert "OCR skipped" in err.getvalue()
        # Image-only document with no OCR available: nothing crashes, and a
        # re-ingest is still clean.
        files2, _ = ingest_pdfs(docs_dir=tmp_path, db_dir=db)
        assert files2 == 2

    @pytestmark_ocr
    def test_ocr_page_and_image_file(self, tmp_path):
        """ocr_page renders via PyMuPDF; ocr_image_file handles standalone images."""
        from PIL import Image, ImageDraw

        make_image_pdf(tmp_path / "one.pdf", "habitat studies direct call")
        page = pymupdf.open(tmp_path / "one.pdf")[0]
        text = ocr.ocr_page(page)
        assert "habitat" in text

        img_path = tmp_path / "note.png"
        img = Image.new("RGB", (400, 120), "white")
        ImageDraw.Draw(img).text((20, 40), "habitat image note", fill="black")
        img.save(img_path)
        assert "habitat" in ocr.ocr_image_file(img_path)
