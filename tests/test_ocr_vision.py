"""Tests for the vision-OCR layer: engine selection, auto escalation, cache
keys per engine, vision fallback on API failure, page cap, metadata.

Run: python -m pytest tests/test_ocr_vision.py -v
No network calls, no real API usage: all engines are fakes. Throwaway
docs/ + Chroma dir; never touches chroma_db/ or the real ocr_cache/.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest  # noqa: E402
import ocr  # noqa: E402
import pymupdf  # noqa: E402
from ingest import ingest_pdfs  # noqa: E402

HANDWRITTEN = "cursive quoll tracks by the creek"  # keyword to retrieve


class FakeEngine(ocr.OcrEngine):
    """Scriptable fake: available(), ocr_image() -> text; records calls."""

    def __init__(self, name: str, model: str, text: str, available: bool = True,
                 fail: bool = False):
        self.name = name
        self._model = model
        self._text = text
        self._available = available
        self._fail = fail
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        return self._available

    def ocr_image(self, image, lang="eng"):
        self.calls += 1
        if self._fail:
            raise RuntimeError("simulated API outage")
        return self._text

    def ocr_page(self, page, dpi=300, lang="eng"):
        return self.ocr_image(None, lang=lang)


def make_image_pdf(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw

    img_path = path.parent / "_tmp.png"
    img = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(img).text((30, 80), text, fill="black")
    img.save(img_path)
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(path)
    doc.close()
    img_path.unlink()


@pytest.fixture()
def isolated_cache(tmp_path, monkeypatch):
    """Point ocr_cache/ at a throwaway dir and reset ingest's per-run state."""
    monkeypatch.setattr(ocr, "OCR_CACHE_DIR", tmp_path / "ocr_cache")
    monkeypatch.setattr(ingest, "_run_engine", "")
    monkeypatch.setattr(ingest, "_vision_budget", [ingest.OCR_MAX_VISION_PAGES])
    yield tmp_path / "ocr_cache"


def test_cache_key_differs_per_engine(isolated_cache, monkeypatch):
    """Same file+page, different engine/model -> distinct cache entries."""
    doc_dir = isolated_cache.parent
    make_image_pdf(doc_dir / "h.pdf", HANDWRITTEN)
    fake_t = FakeEngine("tesseract", "", "printed words")
    fake_c = FakeEngine("claude", "vision-1", HANDWRITTEN)
    page = pymupdf.open(doc_dir / "h.pdf")[0]

    t1 = ocr.cached_ocr_page(doc_dir / "h.pdf", 1, page, engine=fake_t)
    c1 = ocr.cached_ocr_page(doc_dir / "h.pdf", 1, page, engine=fake_c)
    t2 = ocr.cached_ocr_page(doc_dir / "h.pdf", 1, page, engine=fake_t)
    assert (t1, c1, t2) == ("printed words", HANDWRITTEN, "printed words")
    assert fake_t.calls == 1 and fake_c.calls == 1  # each cached under its own key

    import json
    cache = json.loads(next(isolated_cache.glob("*.json")).read_text(encoding="utf-8"))
    keys = set(cache)
    assert "1|tesseract|" in keys and "1|claude|vision-1" in keys

    # Changing the model invalidates the old model's entry.
    fake_c2 = FakeEngine("claude", "vision-2", "new model text")
    assert ocr.cached_ocr_page(doc_dir / "h.pdf", 1, page, engine=fake_c2) == "new model text"
    assert fake_c2.calls == 1


def test_auto_escalates_low_confidence_pages(isolated_cache, monkeypatch):
    """auto: high-confidence tesseract output stays; low-confidence escalates."""
    doc_dir = isolated_cache.parent
    make_image_pdf(doc_dir / "auto.pdf", HANDWRITTEN)
    fake_t = FakeEngine("tesseract", "", "blurry words")
    fake_t.mean_word_confidence = lambda image, lang="eng": 25.0  # below threshold
    fake_v = FakeEngine("claude", "vision-1", HANDWRITTEN)
    auto = ocr.AutoEngine(tesseract=fake_t, vision=fake_v)

    out = auto.ocr_image(object())
    assert out == HANDWRITTEN and fake_v.calls == 1
    assert auto.last_engine_used is fake_v

    # High confidence + dictionary-looking text: no escalation.
    fake_t2 = FakeEngine("tesseract", "", "the theory of everything is for")
    fake_t2.mean_word_confidence = lambda image, lang="eng": 90.0
    fake_v2 = FakeEngine("claude", "vision-1", "should not be called")
    auto2 = ocr.AutoEngine(tesseract=fake_t2, vision=fake_v2)
    out2 = auto2.ocr_image(object())
    assert out2 == "the theory of everything is for" and fake_v2.calls == 0
    assert auto2.last_engine_used is fake_t2


def test_vision_api_failure_falls_back_to_tesseract(isolated_cache, monkeypatch):
    """A failing vision engine must not crash ingestion: tesseract output is
    indexed and the page still retrievable."""
    doc_dir = isolated_cache.parent
    make_image_pdf(doc_dir / "down.pdf", HANDWRITTEN)
    db = str(isolated_cache.parent / "db")

    fake_v = FakeEngine("claude", "vision-1", "", fail=True)
    fake_t = FakeEngine("tesseract", "", "fallback printed text about " + HANDWRITTEN)
    for mod in (ocr, ingest):
        monkeypatch.setattr(mod, "build_engine",
                            lambda name: fake_v if name in ("claude", "ollama") else fake_t)
        monkeypatch.setattr(mod, "get_engine",
                            lambda name=None: fake_v if name == "claude" else
                            (fake_t if name == "tesseract" else fake_v))
    monkeypatch.setattr(ingest, "_run_engine", "claude")

    files, total = ingest_pdfs(docs_dir=doc_dir, db_dir=db, engine_name="claude")
    assert files == 1 and total > 0
    import chromadb
    metas = chromadb.PersistentClient(path=db).get_collection("notes").get(
        include=["metadatas"])["metadatas"]
    assert metas and all(m["ocr"] for m in metas)
    assert all(m["ocr_engine"] == "tesseract" for m in metas), "fallback engine not recorded"
    hits = __import__("retrieve").retrieve(HANDWRITTEN, k=2, db_dir=db, mode="dense")
    assert hits and hits[0]["source"] == "down.pdf"


def test_vision_page_cap_enforced(isolated_cache, monkeypatch):
    """Pages beyond OCR_MAX_VISION_PAGES fall back to Tesseract with a warning."""
    doc_dir = isolated_cache.parent
    n_pages = 3
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page(width=612, height=792)
        pix_path = doc_dir / f"_p{i}.png"
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (600, 200), "white")
        ImageDraw.Draw(img).text((30, 80), f"{HANDWRITTEN} page {i}", fill="black")
        img.save(pix_path)
        page.insert_image(page.rect, filename=str(pix_path))
        pix_path.unlink()
    doc.save(doc_dir / "capped.pdf")
    doc.close()
    db = str(isolated_cache.parent / "db")

    fake_v = FakeEngine("claude", "vision-1", HANDWRITTEN)
    fake_t = FakeEngine("tesseract", "", "tesseract text " + HANDWRITTEN)
    for mod in (ocr, ingest):
        monkeypatch.setattr(mod, "build_engine",
                            lambda name: fake_v if name in ("claude", "ollama") else fake_t)
        monkeypatch.setattr(mod, "get_engine",
                            lambda name=None: fake_v if name == "claude" else fake_t)
    monkeypatch.setattr(ingest, "_run_engine", "claude")
    monkeypatch.setenv("OCR_MAX_VISION_PAGES", "2")

    ingest_pdfs(docs_dir=doc_dir, db_dir=db, engine_name="claude")
    assert fake_v.calls == 2, "vision cap not enforced"
    assert fake_t.calls == 1
    import chromadb
    metas = chromadb.PersistentClient(path=db).get_collection("notes").get(
        include=["metadatas"])["metadatas"]
    engines = {m["page"]: m["ocr_engine"] for m in metas}
    assert set(engines.values()) == {"claude", "tesseract"}


def test_metadata_records_engine(isolated_cache, monkeypatch):
    doc_dir = isolated_cache.parent
    make_image_pdf(doc_dir / "meta.pdf", HANDWRITTEN)
    db = str(isolated_cache.parent / "db")

    fake_v = FakeEngine("claude", "vision-x", HANDWRITTEN + " transcript")
    for mod in (ocr, ingest):
        monkeypatch.setattr(mod, "get_engine", lambda name=None: fake_v)
    monkeypatch.setattr(ingest, "_run_engine", "claude")

    ingest_pdfs(docs_dir=doc_dir, db_dir=db, engine_name="claude")
    import chromadb
    metas = chromadb.PersistentClient(path=db).get_collection("notes").get(
        include=["metadatas"])["metadatas"]
    assert metas and all(m["ocr"] and m["ocr_engine"] == "claude" for m in metas)
