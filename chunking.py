"""Text chunking, heading detection, and section splitting for the RAG pipeline.

Extracted from ingest.py (Step 6 of the optimization pass); all names are
re-exported from ingest.py via ``from chunking import ...`` so callers that
already import from ingest keep working without change.

Three chunking modes (chosen via the CHUNKING env var in ingest.py):
  window   — fixed word windows (default, original behaviour).
  heading  — split each page at its detected section headings; chunks never
             cross a heading boundary; `section` metadata on every chunk.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterator

try:
    import wordninja
except ImportError:
    wordninja = None

# ---------------------------------------------------------------------------
# Constants (also imported by ingest.py)
# ---------------------------------------------------------------------------
CHILD_SIZE = 120    # words per child chunk
CHILD_OVERLAP = 20
PARENT_SIZE = 500   # words per parent chunk (page-aware)

CHUNKING_WINDOW = "window"
CHUNKING_HEADING = "heading"
DEFAULT_CHUNKING = CHUNKING_WINDOW
_MODES = (CHUNKING_WINDOW, CHUNKING_HEADING)
# Short mode tag baked into every chunk ID so switching modes can never leave
# a stale chunk that looks live.
_ID_MODE = {CHUNKING_WINDOW: "win", CHUNKING_HEADING: "head"}

MAX_HEADING_WORDS = 16
MAX_HEADING_CHARS = 90

# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

def _split_camel(token: str) -> str:
    """Insert spaces at lower->Upper boundaries and letter-digit joins."""
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


# ---------------------------------------------------------------------------
# Word-window chunking
# ---------------------------------------------------------------------------

def chunk_words(text: str, size: int = CHILD_SIZE,
                overlap: int = CHILD_OVERLAP) -> Iterator[str]:
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


# ---------------------------------------------------------------------------
# Chunk IDs
# ---------------------------------------------------------------------------

def _chunk_id(source: str, page: int, index: int | str, text: str,
              mode: str = CHUNKING_WINDOW) -> str:
    """Deterministic ID; the text hash makes edits produce a new ID.

    `mode` is part of the ID ('win'/'head'): re-ingesting under a different
    CHUNKING setting writes a different ID, so the stale-chunk sweep removes
    the other mode's chunks instead of leaving two chunkings mixed in one index.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    tag = _ID_MODE.get(mode, mode)
    return f"{source}-p{page}-{tag}-c{index}-{digest}"


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

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


def _page_body_size(lines: list[tuple[float, int, str, str]]) -> float:
    """Font size of the body text: the size covering the most characters."""
    weight: dict[float, int] = {}
    for size, _flags, _font, text in lines:
        if size:
            weight[size] = weight.get(size, 0) + len(text)
    return max(weight, key=weight.get) if weight else 0.0


def _line_is_heading(text: str, size: float, flags: int, body: float,
                     font: str = "") -> bool:
    """Is this single line a section heading?"""
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

    Returns [] for pages with no text layer — callers fall back to window
    chunking for such pages.
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


def load_headings(pdf_path) -> dict[int, list[str]]:
    """Detected headings per page (1-based) for one PDF."""
    import pymupdf
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


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _heading_marks(page_text: str, headings: list[str] | None,
                   ) -> list[tuple[int, str]]:
    """Usable heading positions (char offset, title) inside a page's text."""
    if not headings:
        return []
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
    # A heading followed almost immediately by the next one is a label in a
    # table/list, not a section — drop it as a boundary only.
    kept: list[tuple[int, str]] = []
    for i, (start, title) in enumerate(marks):
        if i + 1 < len(marks):
            between = page_text[start + len(title):marks[i + 1][0]]
            if len(between.split()) < 3:
                continue
        kept.append((start, title))
    return kept


def split_sections(page_text: str, headings: list[str] | None,
                   ) -> list[tuple[str | None, str]]:
    """Split a page's cleaned text at its headings.

    Returns [(section_title | None, section_text), ...].
    """
    if not headings:
        return [(None, page_text)]
    marks = _heading_marks(page_text, headings)
    if not marks:
        return [(None, page_text)]
    sections: list[tuple[str | None, str]] = []
    lead = page_text[:marks[0][0]].strip() if marks[0][0] > 0 else ""
    short_lead = ""
    if lead:
        if len(lead.split()) <= 3:
            short_lead = lead
        else:
            sections.append((None, lead))
    for i, (start, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(page_text)
        body = page_text[start:end].strip()
        if not body:
            continue
        if sections and len(body.split()) <= len(title.split()):
            ptitle, pbody = sections[-1]
            sections[-1] = (ptitle, pbody + " " + body)
        else:
            sections.append((title, body))
    if short_lead:
        if sections:
            title0, body0 = sections[0]
            sections[0] = (title0, short_lead + " " + body0)
        else:
            sections.append((None, short_lead))
    return sections or [(None, page_text)]


def _with_section(title: str | None, text: str) -> str:
    """Prepend the section title to a chunk that doesn't already start with it."""
    if not title or not text:
        return text
    if text.lower().startswith(title.lower()):
        return text
    sep = " " if title.endswith((":", "?", "!", ".")) else ": "
    return f"{title}{sep}{text}"


# ---------------------------------------------------------------------------
# Plan chunks (parents + children) for one page
# ---------------------------------------------------------------------------

def plan_page_chunks(
    page_text: str,
    headings: list[str] | None,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
    parent_size: int = PARENT_SIZE,
) -> tuple[list[tuple[str, str | None, int]],
           list[tuple[str, str | None, int]]]:
    """Parents and children for one page, in page word order.

    Each item is (text, section_title | None, start_word).

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
    """Index of the parent window containing `start_word`."""
    idx = 0
    for i, s in enumerate(parent_starts):
        if s <= start_word:
            idx = i
        else:
            break
    return idx
