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
CHILD_SIZE = 120   # words per child chunk
CHILD_OVERLAP = 20
PARENT_SIZE = 500  # words per parent chunk (page-aware)

# --- Chunking mode --------------------------------------------------------------
# CHUNKING=window  (default) fixed word windows — the original behaviour.
# CHUNKING=heading split each page at its detected headings, so a chunk never
#                  crosses a section boundary; sections get `section` metadata.
# Read per ingest run (not at import) so the app, CLI and tests can switch
# modes without re-importing; unknown values fall back to window.
CHUNKING_WINDOW = "window"
CHUNKING_HEADING = "heading"
DEFAULT_CHUNKING = CHUNKING_WINDOW
# Short mode tag baked into every chunk ID so switching modes can never leave
# a stale chunk that looks live (and repair/re-ingest swaps cleanly).
_ID_MODE = {CHUNKING_WINDOW: "win", CHUNKING_HEADING: "head"}


def chunking_mode() -> str:
    """The active chunking mode ('window' or 'heading')."""
    mode = (os.environ.get("CHUNKING", "") or DEFAULT_CHUNKING).strip().lower()
    return mode if mode in (CHUNKING_WINDOW, CHUNKING_HEADING) else DEFAULT_CHUNKING


def _normalize_chunking(value: str | None) -> str:
    """None -> the CHUNKING env var; anything unknown -> window."""
    if value is None:
        return chunking_mode()
    value = value.strip().lower()
    return value if value in (CHUNKING_WINDOW, CHUNKING_HEADING) else DEFAULT_CHUNKING

# Below this many non-whitespace chars, a page counts as scanned (no text
# layer) and gets OCR'd instead. Overridable via OCR_MIN_CHARS in .env.
OCR_MIN_CHARS_DEFAULT = 30
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", OCR_MIN_CHARS_DEFAULT))
# Max pages sent to a vision OCR engine per ingest run; the rest fall back to
# Tesseract (cost control for OCR_ENGINE=claude/ollama/auto).
OCR_MAX_VISION_PAGES = int(os.environ.get("OCR_MAX_VISION_PAGES", "50"))

# --- Text cleanup -------------------------------------------------------------
# pypdf often drops inter-word spaces ("CommonlyinvolvedinDNA"), which wrecks
# BM25 tokenization. PyMuPDF's extractor does not, so it is the primary
# extractor (measured on 170.pdf: 2.16% fused tokens with pypdf vs 0.01% with
# PyMuPDF). clean_text() below is the fallback/secondary cleanup applied to
# whatever the extractor returns: splits camelCase and letter-digit joins,
# and (if wordninja is installed) word-splits long lowercase tokens.

try:
    import wordninja
except ImportError:
    wordninja = None


def _split_camel(token: str) -> str:
    """Insert spaces at lower->Upper boundaries and letter->digit joins."""
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", token)   # camelCase
    token = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", token)      # letter123
    token = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", token)      # 123letter
    return token


def _maybe_wordninja(token: str) -> str:
    """Wordninja only on long all-lowercase tokens (e.g. 'antibodybinding')."""
    if wordninja is None or len(token) <= 18 or not token.isalpha() or not token.islower():
        return token
    return " ".join(wordninja.split(token))


def clean_text(text: str) -> str:
    """Normalize extracted PDF text: de-hyphenate line breaks, split fused
    tokens, collapse whitespace."""
    # Hyphenation across line breaks: "protei-\nnn" -> "protein"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    words = []
    for tok in text.split():
        parts = _split_camel(tok).split()
        words.extend(_maybe_wordninja(p) for p in parts)
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def fused_token_rate(text: str) -> float:
    """Fraction of alphabetic tokens longer than 18 chars (fused-word signal)."""
    toks = text.split()
    if not toks:
        return 0.0
    alpha = [w for w in toks if w.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for w in alpha if len(w) > 18) / len(alpha)


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


def chunk_words(text: str, size: int = CHILD_SIZE, overlap: int = CHILD_OVERLAP) -> Iterator[str]:
    """Split text into overlapping word windows."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk size ({size})")
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def chunk_words_page(text: str, size: int = PARENT_SIZE) -> Iterator[str]:
    """Split one page's text into non-overlapping windows of `size` words."""
    words = text.split()
    for start in range(0, len(words), size):
        yield " ".join(words[start:start + size])


def _chunk_id(source: str, page: int, index: int, text: str,
              mode: str = CHUNKING_WINDOW) -> str:
    """Deterministic ID; the text hash makes edits to a page produce a new
    ID (old one gets cleaned up), identical reruns produce the same one.

    `mode` is part of the ID ('win'/'head'): re-ingesting under a different
    CHUNKING setting writes a different ID for the same page position, so the
    stale-chunk sweep below removes the other mode's chunks instead of leaving
    two chunkings mixed in one index.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    tag = _ID_MODE.get(mode, mode)
    return f"{source}-p{page}-{tag}-c{index}-{digest}"


# --- Heading detection ----------------------------------------------------------
# A line counts as a heading when it is visually or structurally set apart from
# the body text: noticeably bigger font, a short bold/numbered/prefixed label,
# or a short label line ending in ':'. Everything is judged per page against
# that page's body size (char-weighted), so a page with no size contrast still
# gets its numbered/bold labels, and a page with no usable signal returns []
# (-> window fallback for that page).

_HEADING_SHAPE_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[.)]?\s+\S"                        # 1. ... / 3.2 ... / 3) ...
    r"|\(\s*(?:[ivx]+|\d+)\s*\)\s*\S"                 # (i) ... / (1) ...
    r")",
    re.IGNORECASE,
)
_PREFIX_LABEL_RE = re.compile(                        # SECTION A ... / Phase 2: ...
    r"^(?:phase|section|chapter|unit|part|topic|module)\b[^\n]{0,80}",
    re.IGNORECASE,
)
_COLON_LABEL_RE = re.compile(r"^\S[^:\n]{0,70}:\s*$")
MAX_HEADING_WORDS = 16
MAX_HEADING_CHARS = 90


def _page_body_size(lines: list[tuple[float, int, str, str]]) -> float:
    """Font size of the body text: the size covering the most characters.
    (A plain median would be skewed by pages with many tiny bullet markers.)"""
    weight: dict[float, int] = {}
    for size, _flags, _font, text in lines:
        if size:
            weight[size] = weight.get(size, 0) + len(text)
    return max(weight, key=weight.get) if weight else 0.0


def _line_is_heading(text: str, size: float, flags: int, body: float,
                     font: str = "") -> bool:
    """Is this single line a section heading? (see module comment)"""
    t = " ".join(text.split())
    n_words = len(t.split())
    letters = sum(1 for c in t if c.isalpha())
    if letters < 2 or not 1 <= n_words <= MAX_HEADING_WORDS or len(t) > MAX_HEADING_CHARS:
        return False
    bold = bool(flags & 16) or "bold" in font.lower()
    if body and size >= body * 1.15:
        return True                                    # bigger than the body
    if n_words <= 8 and _COLON_LABEL_RE.match(t):
        return True                                    # "Structural features:"
    if _HEADING_SHAPE_RE.match(t) and (bold or (body and size > body)):
        return True                                    # numbered / (i) labels
    if (_PREFIX_LABEL_RE.match(t) and (bold or (body and size > body))
            and (n_words >= 3 or ":" in t)):
        return True                                    # SECTION A ... / Phase 2: ...
    if bold and body and size > body and n_words >= 2:
        return True                                    # emphasised sub-heading
    return False


def detect_page_headings(page) -> list[str]:
    """Heading lines of one PyMuPDF page, in reading order.

    Returns [] for pages with no text layer (scanned pages that went through
    OCR only yield a plain string) — callers fall back to window chunking for
    such pages, which is the documented OCR/headingless fallback.
    """
    try:
        info = page.get_text("dict")
    except Exception:
        return []
    lines: list[tuple[float, int, str, str]] = []
    for block in info.get("blocks", []) or []:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []) or []:
            spans = [sp for sp in line.get("spans", []) if (sp.get("text") or "").strip()]
            if not spans:
                continue
            text = "".join(sp.get("text", "") for sp in spans)
            dominant = max(spans, key=lambda sp: len(sp.get("text", "")))
            lines.append((
                float(dominant.get("size") or 0),
                int(dominant.get("flags") or 0),
                str(dominant.get("font") or ""),
                text,
            ))
    body = _page_body_size(lines)
    out = []
    for size, flags, font, text in lines:
        if _line_is_heading(text, size, flags, body, font=font):
            out.append(" ".join(text.split()))
    return out


def load_headings(pdf_path: Path) -> dict[int, list[str]]:
    """Detected headings per page (1-based) for one PDF.

    Pages whose text layer is missing/empty (they get OCR'd by load_pages)
    are simply absent here — window chunking covers them.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return {}
    out: dict[int, list[str]] = {}
    try:
        for pno, page in enumerate(doc, start=1):
            heads = detect_page_headings(page)
            if heads:
                out[pno] = heads
    finally:
        doc.close()
    return out


def split_sections(page_text: str, headings: list[str] | None
                   ) -> list[tuple[str | None, str]]:
    """Split a page's cleaned text at its headings.

    Returns [(section_title | None, section_text), ...] where section_text
    starts at its heading (so the title is already inside it). A heading that
    cleaning moved/removed is skipped, and a page with no usable heading comes
    back as one untitled section — i.e. exactly the window chunking.
    """
    if not headings:
        return [(None, page_text)]
    marks: list[tuple[int, str]] = []
    cursor = 0
    for h in headings:
        h = clean_text(h)
        if len(h) < 2:
            continue
        idx = page_text.find(h, cursor)
        if idx < 0:
            continue
        marks.append((idx, h))
        cursor = idx + len(h)
    if not marks:
        return [(None, page_text)]
    # A heading followed (almost) immediately by the next one is a label in a
    # table/list, not a section — e.g. the "Phase | Phase Name | Phase 1"
    # header cells of a checklist table. Drop it as a boundary only; its words
    # stay in the text, so nothing is lost but the cut.
    kept: list[tuple[int, str]] = []
    for i, (start, title) in enumerate(marks):
        if i + 1 < len(marks):
            between = page_text[start + len(title):marks[i + 1][0]]
            if len(between.split()) < 3:
                continue
        kept.append((start, title))
    marks = kept
    if not marks:
        return [(None, page_text)]
    sections: list[tuple[str | None, str]] = []
    lead = page_text[:marks[0][0]].strip() if marks[0][0] > 0 else ""
    short_lead = ""
    if lead:
        if len(lead.split()) <= 3:
            short_lead = lead   # too small to stand alone; fold in below
        else:
            sections.append((None, lead))
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(page_text)
        body = page_text[start:end].strip()
        if not body:
            continue
        # A heading with no content after it (page ended right there) is not a
        # section of its own — fold it into the previous one instead of
        # emitting a chunk that is only a title.
        if sections and len(body.split()) <= len(title.split()):
            ptitle, pbody = sections[-1]
            sections[-1] = (ptitle, pbody + " " + body)
        else:
            sections.append((title, body))
    # Same idea in the other direction: a lead-in too short to stand alone
    # (usually table cells dropped by the adjacency rule above, which uses the
    # same 3-word scale) joins the first section rather than becoming a
    # 1-3 word chunk of its own.
    if short_lead:
        if sections:
            title0, body0 = sections[0]
            sections[0] = (title0, short_lead + " " + body0)
        else:
            sections.append((None, short_lead))
    return sections or [(None, page_text)]


def _with_section(title: str | None, text: str) -> str:
    """Prepend the section title to a chunk that doesn't already start with
    it (only a section's first chunk contains the heading itself), so BM25 and
    the embedder see the section name on every chunk of the section."""
    if not title or not text:
        return text
    if text.lower().startswith(title.lower()):
        return text
    sep = " " if title.endswith((":", "?", "!", ".")) else ": "
    return f"{title}{sep}{text}"


def plan_page_chunks(page_text: str, headings: list[str] | None,
                     child_size: int = CHILD_SIZE,
                     child_overlap: int = CHILD_OVERLAP,
                     parent_size: int = PARENT_SIZE,
                     ) -> tuple[list[tuple[str, str | None, int]],
                                list[tuple[str, str | None, int]]]:
    """Parents and children for one page, in page word order.

    Each item is (text, section_title | None, start_word) where start_word is
    the offset of the chunk's first word within the page — children are mapped
    to the parent that contains them.

    Window mode (no headings): identical to the old fixed word windows.
    Heading mode: the page is split into sections first, then windows are cut
    inside each section, so no chunk ever crosses a heading boundary.
    """
    sections = split_sections(page_text, headings)
    parents: list[tuple[str, str | None, int]] = []
    children: list[tuple[str, str | None, int]] = []
    word_cursor = 0
    for title, text in sections:
        for j, ptxt in enumerate(chunk_words_page(text, parent_size)):
            parents.append((_with_section(title, ptxt), title,
                            word_cursor + j * parent_size))
        for i, ctxt in enumerate(chunk_words(text, child_size, child_overlap)):
            children.append((_with_section(title, ctxt), title,
                             word_cursor + i * (child_size - child_overlap)))
        word_cursor += len(text.split())
    return parents, children


def _parent_index_for(start_word: int, parent_starts: list[int]) -> int:
    """Index of the parent window containing `start_word` (last one that has
    started at or before it). For contiguous window mode this is exactly the
    old `start_word // parent_size`."""
    idx = 0
    for i, s in enumerate(parent_starts):
        if s <= start_word:
            idx = i
        else:
            break
    return idx



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

    chunking: 'window' (default) or 'heading'; None reads the CHUNKING env
    var. Heading mode splits pages at detected headings (see plan_page_chunks)
    and stamps `chunking`/`section` into every chunk's metadata; because the
    mode is part of each chunk ID, re-ingesting under another mode replaces
    the old mode's chunks instead of mixing them.

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
            # Heading mode: detect per-page section headings from the PDF's
            # font/structure info. Pages with no text layer (OCR'd) or with no
            # usable headings are simply missing here -> window fallback.
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
