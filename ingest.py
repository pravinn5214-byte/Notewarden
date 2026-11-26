"""Steps 1-2 of the pipeline: load PDFs -> chunk -> embed -> store in Chroma.

Two levels of chunks:
- CHILD chunks (~120 words, 20 overlap) are indexed and used for matching.
- PARENT chunks (~500 words, page-aware, no overlap across pages) are stored
  only for context: when a child matches, its parent text is what the LLM reads.

Idempotent: every chunk gets a deterministic ID derived from its file, page,
index and text hash, so rerunning ingestion updates changed chunks in place
instead of creating duplicates. Chunks whose source disappeared (deleted file
or page) are removed from both collections.

Usage:
    1. Put your PDFs in the docs/ folder.
    2. Run: python ingest.py
"""
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterator

import chromadb  # noqa: F401  (kept for callers that import chromadb transitively)
import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ocr import (  # noqa: F401  (re-exported for callers/tests)
    VISION_ENGINE_NAMES, build_engine, cached_ocr_page, get_engine,
    is_tesseract_available, ocr_image_file,
)
from store import (  # noqa: F401  (re-exported for callers/tests)
    IndexCorruptError, LockHeldError, acquire_write_lock, db_path,
    get_client, health_check, release_write_lock, with_index_recovery,
)

load_dotenv()

DOCS_DIR = Path("docs")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DB_DIR = db_path()  # CHROMA_DIR env var, default chroma_db/
from store import COLLECTION, PARENT_COLLECTION  # single source of truth
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally
from chunking import (  # noqa: F401  (re-exported for callers/tests)
    CHILD_OVERLAP, CHILD_SIZE, CHUNKING_HEADING, CHUNKING_WINDOW,
    DEFAULT_CHUNKING, MAX_HEADING_CHARS, MAX_HEADING_WORDS, PARENT_SIZE,
    _COLON_LABEL_RE, _HEADING_SHAPE_RE, _ID_MODE, _MODES, _PREFIX_LABEL_RE,
    _chunk_id, _heading_marks, _line_is_heading, _maybe_wordninja,
    _page_body_size, _parent_index_for, _split_camel, _with_section,
    chunk_words, chunk_words_page, clean_text, detect_page_headings,
    fused_token_rate, load_headings, plan_page_chunks, split_sections,
)

# --- Chunking mode --------------------------------------------------------------
# CHUNKING=window        (default) fixed word windows — the original behaviour.
# CHUNKING=heading       split each page at its detected headings, so a chunk
#                        never crosses a section boundary; sections get
#                        `section` metadata.
# Read per ingest run (not at import) so the app, CLI and tests can switch
# modes without re-importing; unknown values fall back to window.
CHUNKING_WINDOW = "window"
CHUNKING_HEADING = "heading"
DEFAULT_CHUNKING = CHUNKING_WINDOW
_MODES = (CHUNKING_WINDOW, CHUNKING_HEADING)
# Short mode tag baked into every chunk ID so switching modes can never leave
# a stale chunk that looks live (and repair/re-ingest swaps cleanly).
_ID_MODE = {CHUNKING_WINDOW: "win", CHUNKING_HEADING: "head"}


def chunking_mode() -> str:
    """The active chunking mode ('window' or 'heading')."""
    mode = (os.environ.get("CHUNKING", "") or DEFAULT_CHUNKING).strip().lower()
    return mode if mode in _MODES else DEFAULT_CHUNKING


def _normalize_chunking(value: str | None) -> str:
    """None -> the CHUNKING env var; anything unknown -> window."""
    if value is None:
        return chunking_mode()
    value = value.strip().lower()
    return value if value in _MODES else DEFAULT_CHUNKING

# Below this many non-whitespace chars, a page counts as scanned (no text
# layer) and gets OCR'd instead. Overridable via OCR_MIN_CHARS in .env.
OCR_MIN_CHARS_DEFAULT = 30
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", OCR_MIN_CHARS_DEFAULT))
# Max pages sent to a vision OCR engine per ingest run; the rest fall back to
# Tesseract (cost control for OCR_ENGINE=claude/ollama/auto).
OCR_MAX_VISION_PAGES = int(os.environ.get("OCR_MAX_VISION_PAGES", "50"))

# --- Text cleanup, chunking & heading detection (moved to chunking.py) --------


# load_pages() records which pages it sourced via OCR here; ingest_pdfs reads
# it right after the call to tag chunk metadata. (A plain module global keeps
# load_pages' signature stable for callers/tests that stub it.)
_last_ocr_pages: set[int] = set()
# Per-run vision budget state: pages still allowed to use a vision engine, and
# the warning issued when the cap is hit.
_vision_budget: list[int] = [OCR_MAX_VISION_PAGES]
# Engine name resolved per run ("" for OCR-off / unavailable).
_run_engine: str = ""
# Records (page -> engine name that produced it) for metadata.
_last_page_engines: dict[int, str] = {}
# Default for load_pages(use_ocr=None); ingest_pdfs sets it per run.
_load_pages_use_ocr = True


def _env_ingest(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


def _ocr_document_page(pdf_path: Path, page_num: int, page, vision_pages: int) -> tuple[str, str, int]:
    """OCR one scanned page: (cleaned text, engine name, remaining vision budget).

    vision_pages is the per-run cap; pages beyond it fall back to Tesseract.
    """
    eng = get_engine(_run_engine) if _run_engine else None
    if eng is None:
        return "", "", vision_pages
    # A vision engine (or auto escalating to one) consumes the budget; pages
    # past the cap use plain Tesseract.
    uses_vision = eng.name in VISION_ENGINE_NAMES or eng.name == "auto"
    if uses_vision and vision_pages <= 0:
        print(f"WARNING: OCR_MAX_VISION_PAGES reached; {pdf_path.name} p.{page_num} "
              "falls back to Tesseract", file=sys.stderr)
        text = cached_ocr_page(pdf_path, page_num, page, engine=build_engine("tesseract"))
        return clean_text(text), "tesseract", vision_pages
    before = getattr(eng, "last_engine_used", None)
    try:
        text = cached_ocr_page(pdf_path, page_num, page, engine=eng)
        used = getattr(eng, "last_engine_used", None) or eng
    except Exception as exc:
        print(f"WARNING: {eng.name} OCR failed on {pdf_path.name} p.{page_num} "
              f"({exc}); falling back to Tesseract", file=sys.stderr)
        tess = build_engine("tesseract")
        if not tess.available():
            return "", "tesseract", vision_pages
        text = cached_ocr_page(pdf_path, page_num, page, engine=tess)
        used = tess
    if before is not used:
        vision_pages -= 1
    return clean_text(text), used.name, vision_pages


def _ocr_image_file_tracked(path: Path, vision_pages: int) -> tuple[str, str, int]:
    """OCR a standalone image upload: (cleaned text, engine name, remaining budget)."""
    from PIL import Image

    eng = get_engine(_run_engine) if _run_engine else None
    if eng is None:
        return "", "", vision_pages
    uses_vision = eng.name in VISION_ENGINE_NAMES or eng.name == "auto"
    if uses_vision and vision_pages <= 0:
        print(f"WARNING: OCR_MAX_VISION_PAGES reached; {path.name} falls back "
              "to Tesseract", file=sys.stderr)
        eng = build_engine("tesseract")
        return clean_text(eng.ocr_image(Image.open(path))), "tesseract", vision_pages
    before = getattr(eng, "last_engine_used", None)
    try:
        text = eng.ocr_image(Image.open(path))
        used = getattr(eng, "last_engine_used", None) or eng
    except Exception as exc:
        print(f"WARNING: {eng.name} OCR failed on {path.name} ({exc}); "
              "falling back to Tesseract", file=sys.stderr)
        tess = build_engine("tesseract")
        if not tess.available():
            return "", "tesseract", vision_pages
        text = tess.ocr_image(Image.open(path))
        used = tess
    if before is not used:
        vision_pages -= 1
    return clean_text(text), used.name, vision_pages


def load_pages(pdf_path: Path, use_ocr: bool | None = None) -> Iterator[tuple[int, str]]:
    """Yield (page_number, cleaned_text) for each page with extractable text.

    Extractor: PyMuPDF (chosen over pypdf after measuring fused-token rates —
    see module comment). Text goes through clean_text() before chunking, so
    dense, BM25 and parent text all operate on the same cleaned words.

    Pages whose extracted text is shorter than OCR_MIN_CHARS non-whitespace
    chars are treated as scanned and OCR'd instead (only those pages; pages
    with a good text layer are never OCR'd). use_ocr=False or a missing
    Tesseract install skips OCR — such pages are just left out.
    """
    global _last_ocr_pages, _last_page_engines
    if use_ocr is None:
        use_ocr = _load_pages_use_ocr
    _last_ocr_pages = set()
    _last_page_engines = {}
    ocr_ok = use_ocr and _run_engine != ""
    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text() or ""
        if not ocr_ok or len("".join(raw.split())) >= OCR_MIN_CHARS:
            text = clean_text(raw)
            if text:
                yield page_num, text
        else:
            _last_ocr_pages.add(page_num)
            text, engine_used, budget = _ocr_document_page(pdf_path, page_num, page, _vision_budget[0])
            _vision_budget[0] = budget
            _last_page_engines[page_num] = engine_used
            yield page_num, text


def ingest_pdfs(
    docs_dir: Path = DOCS_DIR,
    db_dir: str = DB_DIR,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
    parent_size: int = PARENT_SIZE,
    progress_cb: Callable[[str], None] | None = None,
    files: list[Path] | None = None,
    use_ocr: bool = True,
    engine_name: str | None = None,
    chunking: str | None = None,
) -> tuple[int, int]:
    """Ingest the given documents (default: every PDF/image in docs_dir).

    Builds the child collection (embedded, used for retrieval) and the parent
    collection (context text, looked up by parent_id). The index is made to
    match the selection exactly: chunks from files not in `files` are removed,
    so deselected files stop contributing to answers.
    Returns (files_processed, child_chunks_stored).

    chunking: 'window' (default), 'heading' or 'window_titled' (experimental);
    None reads the CHUNKING env var. Heading mode splits pages at detected
    headings (see plan_page_chunks) and stamps `chunking`/`section` into every
    chunk's metadata; window_titled keeps window boundaries and only prefixes
    child text with the nearest heading (no `section`); because the mode is
    part of each chunk ID, re-ingesting under another mode replaces the old
    mode's chunks instead of mixing them.

    Image files (.png/.jpg/.jpeg) are treated as 1-page documents and OCR'd.
    For PDFs, only pages with less than OCR_MIN_CHARS of extracted text are
    OCR'd (when use_ocr and Tesseract are available); chunk metadata records
    `ocr: true/false` per chunk.

    progress_cb, if given, is called with a short status string per PDF
    (used by the Streamlit UI as a progress indicator).
    """
    mode = _normalize_chunking(chunking)
    docs = sorted(
        p for p in (
            files if files is not None
            else list(docs_dir.glob("*.pdf")) + [p for ext in IMAGE_EXTENSIONS for p in docs_dir.glob(ext)]
        )
        if p.is_file()
    )
    if not docs:
        raise FileNotFoundError(f"No PDFs or images found in {docs_dir}/ - add some first.")

    global _run_engine, _vision_budget
    engine_name = engine_name or _env_ingest("OCR_ENGINE", "tesseract")
    engine_ok = False
    if use_ocr:
        try:
            engine_ok = get_engine(engine_name).available()
        except Exception:
            engine_ok = False
    use_ocr = use_ocr and engine_ok
    # Resolve the engine once per run; "auto" is stored as-is (it resolves
    # per page) but still needs Tesseract-or-vision to be available.
    _run_engine = engine_name if use_ocr else ""
    _vision_budget = [int(_env_ingest("OCR_MAX_VISION_PAGES", str(OCR_MAX_VISION_PAGES)))]
    if not use_ocr:
        warn = ("OCR skipped: no OCR backend found. Scanned pages will be "
                "ignored. Install Tesseract (see README 'Setting up OCR'), e.g. on "
                "Windows the UB Mannheim build from "
                "https://github.com/UB-Mannheim/tesseract/wiki, or "
                "`brew install tesseract` / `sudo apt install tesseract-ocr`, "
                "or set OCR_ENGINE=claude with ANTHROPIC_API_KEY.")
        if progress_cb:
            progress_cb("WARNING: " + warn)
        else:
            print(f"WARNING: {warn}", file=sys.stderr)
    elif engine_name in VISION_ENGINE_NAMES or engine_name == "auto":
        est = sum(
            1 for d in docs if d.suffix.lower() in IMAGE_EXTENSIONS
        ) + sum(
            1 for d in docs if d.suffix.lower() == ".pdf"
            for p in pymupdf.open(d)
            if len("".join((p.get_text() or "").split())) < OCR_MIN_CHARS
        )
        if est:
            msg = (f"~{est} page(s) will use the vision model ({engine_name}); "
                   f"cap {OCR_MAX_VISION_PAGES}/run (OCR_MAX_VISION_PAGES).")
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

    # Diagnostic: fused-token % before vs after cleanup (target < 1% after).
    for pdf in docs:
        if pdf.suffix.lower() != ".pdf":
            continue
        try:
            raw = " ".join((p.get_text() or "") for p in pymupdf.open(pdf))
            before, after = fused_token_rate(raw), fused_token_rate(clean_text(raw))
            msg = f"{pdf.name}: fused tokens {before:.2%} -> {after:.2%}"
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)
        except Exception:
            pass  # diagnostic only

    client = get_client(db_dir)
    lock = acquire_write_lock()
    try:
        collection = with_index_recovery(
            lambda: client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}))
        parents = with_index_recovery(lambda: client.get_or_create_collection(PARENT_COLLECTION))
    except BaseException:
        release_write_lock(lock)
        raise

    ids, texts, metas = [], [], []
    parent_ids, parent_texts = [], []
    live_ids: set[str] = set()       # child chunks that should exist after this run
    live_parent_ids: set[str] = set()
    ocr_pages_total = ocr_pages_count = 0
    for doc_path in docs:
        if progress_cb:
            progress_cb(f"Reading {doc_path.name}...")
        file_chunks = 0
        page_headings: dict[int, list[str]] = {}
        if mode == CHUNKING_HEADING and doc_path.suffix.lower() not in IMAGE_EXTENSIONS:
            # Heading-aware modes: detect per-page section headings from the
            # PDF's font/structure info. Pages with no text layer (OCR'd) or
            # with no usable headings are simply missing here -> window fallback.
            page_headings = load_headings(doc_path)
        if doc_path.suffix.lower() in IMAGE_EXTENSIONS:
            # An image upload is a 1-page document; it only has pixels, so
            # without OCR there is nothing to index.
            if use_ocr:
                text, engine_used, budget = _ocr_image_file_tracked(doc_path, _vision_budget[0])
                _vision_budget[0] = budget
                page_texts = [(1, text)]
                page_ocr_flags = {1: True}
                page_engines = {1: engine_used}
                ocr_pages_total += 1
                ocr_pages_count += 1
            else:
                page_texts, page_ocr_flags, page_engines = [], {}, {}
        else:
            page_texts = list(load_pages(doc_path))
            page_ocr_flags = {p: True for p in _last_ocr_pages}
            page_engines = dict(_last_page_engines)
            ocr_pages_total += len(page_texts)
            ocr_pages_count += len(_last_ocr_pages)
        for page_num, page_text in page_texts:
            # Parents and children for this page. Window mode cuts fixed word
            # windows over the page; heading mode cuts them per section (see
            # plan_page_chunks), so a chunk never crosses a heading.
            headings = page_headings.get(page_num, [])
            page_parents, page_children = plan_page_chunks(
                page_text, headings, child_size, child_overlap, parent_size)
            page_parent_ids = []
            parent_starts = []
            for j, (ptxt, _section, start_word) in enumerate(page_parents):
                pid = _chunk_id(doc_path.name, page_num, f"par{j}", ptxt, mode)
                page_parent_ids.append(pid)
                parent_starts.append(start_word)
                if pid not in live_parent_ids:
                    parent_ids.append(pid)
                    parent_texts.append(ptxt)
                    live_parent_ids.add(pid)

            for i, (chunk, section, start_word) in enumerate(page_children):
                cid = _chunk_id(doc_path.name, page_num, i, chunk, mode)
                # Which parent covers this child: children and parents are cut
                # from the same page/section text in page word order.
                pidx = _parent_index_for(start_word, parent_starts)
                ids.append(cid)
                texts.append(chunk)
                metas.append({
                    "source": doc_path.name,
                    "page": page_num,
                    "parent_id": page_parent_ids[pidx],
                    "ocr": bool(page_ocr_flags.get(page_num, False)),
                    "ocr_engine": page_engines.get(page_num, ""),
                    "chunking": mode,
                    "section": section or "",
                })
                live_ids.add(cid)
                file_chunks += 1
        if progress_cb:
            progress_cb(f"{doc_path.name}: {file_chunks} chunks")

    if ocr_pages_total:
        msg = f"{ocr_pages_count} of {ocr_pages_total} pages OCR'd"
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    # Re-embed only chunks the store doesn't already have. Existing chunks get
    # a metadata-only upsert too: if the schema changed (e.g. adding
    # parent_id), old rows must not keep stale metadata just because their
    # content-derived ID is unchanged.
    existing = set(collection.get(ids=list(live_ids), include=[])["ids"]) if live_ids else set()
    new_idx = [i for i, cid in enumerate(ids) if cid not in existing]
    old_idx = [i for i, cid in enumerate(ids) if cid in existing]
    if old_idx:
        collection.upsert(
            ids=[ids[i] for i in old_idx],
            documents=[texts[i] for i in old_idx],
            metadatas=[metas[i] for i in old_idx],
        )
    if new_idx:
        if progress_cb:
            progress_cb(f"Embedding {len(new_idx)} new/changed chunks...")
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode([texts[i] for i in new_idx], show_progress_bar=False).tolist()
        collection.upsert(
            ids=[ids[i] for i in new_idx],
            documents=[texts[i] for i in new_idx],
            embeddings=embeddings,
            metadatas=[metas[i] for i in new_idx],
        )

    # Upsert parents that are new or whose text changed (parents are cheap:
    # document only, no embedding).
    existing_parents = set(parents.get(ids=list(live_parent_ids), include=[])["ids"]) if live_parent_ids else set()
    new_parent_idx = [i for i, pid in enumerate(parent_ids) if pid not in existing_parents]
    if new_parent_idx:
        parents.upsert(
            ids=[parent_ids[i] for i in new_parent_idx],
            documents=[parent_texts[i] for i in new_parent_idx],
            metadatas=[{"source": pid.rsplit("-p", 1)[0]} for pid in (parent_ids[i] for i in new_parent_idx)],
        )

    # Remove chunks whose file/page no longer exists (deleted or edited PDFs).
    stale = [cid for cid in collection.get(include=[])["ids"] if cid not in live_ids]
    if stale:
        collection.delete(ids=stale)
    stale_parents = [pid for pid in parents.get(include=[])["ids"] if pid not in live_parent_ids]
    if stale_parents:
        parents.delete(ids=stale_parents)
    if progress_cb:
        progress_cb(f"Removed {len(stale)} stale child / {len(stale_parents)} stale parent chunk(s)")

    release_write_lock(lock)
    return len(docs), collection.count()


def heading_report(docs_dir: Path = DOCS_DIR, limit: int = 30) -> str:
    """Per-file heading-detection summary for sanity-checking real PDFs.

    Lists the first `limit` headings per file and flags files with zero
    detected headings (they fall back to window chunking) or with an
    implausible density of more than one heading per MIN_WORDS_PER_HEADING
    words. Read-only: only the PDF text layer is inspected (no OCR).
    """
    MIN_WORDS_PER_HEADING = 50
    out = ["# Heading detection report", "",
           f"Rule: bigger-than-body font, numbered/prefixed labels, short "
           f"bold sub-headings and 'label:' lines · flag threshold: "
           f"< {MIN_WORDS_PER_HEADING} words per heading", ""]
    for pdf in sorted(docs_dir.glob("*.pdf")):
        try:
            doc = pymupdf.open(pdf)
        except Exception as exc:
            out += [f"## {pdf.name}", "", f"Could not open: {exc}", ""]
            continue
        found: list[tuple[int, str]] = []
        words = 0
        text_pages = 0
        for pno, page in enumerate(doc, start=1):
            raw = page.get_text() or ""
            if raw.strip():
                text_pages += 1
                words += len(raw.split())
            found.extend((pno, h) for h in detect_page_headings(page))
        page_count = doc.page_count
        doc.close()

        n = len(found)
        per = words / n if n else 0.0
        out += [f"## {pdf.name}", "",
                f"- {page_count} pages ({text_pages} with a text layer), "
                f"{n} heading(s), {words} words"
                + (f", 1 per {per:.0f} words" if n else "")]
        if n == 0:
            out.append("- **FLAG: NO HEADINGS DETECTED — this file falls back "
                       "to window chunking**")
        elif per < MIN_WORDS_PER_HEADING:
            out.append(f"- **FLAG: implausibly many headings — 1 per "
                       f"{per:.0f} words (< {MIN_WORDS_PER_HEADING})**")
        out.append("")
        for pno, h in found[:limit]:
            out.append(f"- p{pno}: {h}")
        if n > limit:
            out.append(f"- … {n - limit} more")
        out.append("")
    return "\n".join(out)


def main() -> None:
    if "--headings" in sys.argv:
        print(heading_report())
        return
    try:
        health_check()
    except IndexCorruptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Fix: delete the index directory and run this command again "
              "(OCR is cached, so pages won't be re-OCR'd).", file=sys.stderr)
        sys.exit(1)
    try:
        files, total = ingest_pdfs()
    except LockHeldError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{files} document(s) processed, {total} child chunks stored in {DB_DIR}/")


if __name__ == "__main__":
    main()
