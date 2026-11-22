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
import re
from pathlib import Path
from typing import Callable, Iterator

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "notes"          # child chunks (embedded)
PARENT_COLLECTION = "parents"  # parent chunks (not embedded, context only)
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally
CHILD_SIZE = 120   # words per child chunk
CHILD_OVERLAP = 20
PARENT_SIZE = 500  # words per parent chunk (page-aware)

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


def load_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, cleaned_text) for each page with extractable text.

    Extractor: PyMuPDF (chosen over pypdf after measuring fused-token rates —
    see module comment). Text goes through clean_text() before chunking, so
    dense, BM25 and parent text all operate on the same cleaned words.
    """
    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text())
        if text:
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


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic ID; the text hash makes edits to a page produce a new
    ID (old one gets cleaned up), identical reruns produce the same one."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}-p{page}-c{index}-{digest}"


def ingest_pdfs(
    docs_dir: Path = DOCS_DIR,
    db_dir: str = DB_DIR,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
    parent_size: int = PARENT_SIZE,
    progress_cb: Callable[[str], None] | None = None,
    files: list[Path] | None = None,
) -> tuple[int, int]:
    """Ingest the given PDFs (default: every PDF in docs_dir).

    Builds the child collection (embedded, used for retrieval) and the parent
    collection (context text, looked up by parent_id). The index is made to
    match the selection exactly: chunks from files not in `files` are removed,
    so deselected files stop contributing to answers.
    Returns (files_processed, child_chunks_stored).

    progress_cb, if given, is called with a short status string per PDF
    (used by the Streamlit UI as a progress indicator).
    """
    pdfs = sorted(files) if files is not None else sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}/ - add some first.")

    # Diagnostic: fused-token % before vs after cleanup (target < 1% after).
    for pdf in pdfs:
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

    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    parents = client.get_or_create_collection(PARENT_COLLECTION)  # no embeddings needed

    ids, texts, metas = [], [], []
    parent_ids, parent_texts = [], []
    live_ids: set[str] = set()       # child chunks that should exist after this run
    live_parent_ids: set[str] = set()
    for pdf in pdfs:
        if progress_cb:
            progress_cb(f"Reading {pdf.name}...")
        file_chunks = 0
        for page_num, page_text in load_pages(pdf):
            # Parents for this page: up to `parent_size` words each, no overlap
            # across pages, so a parent never spans two pages.
            page_parents = list(chunk_words_page(page_text, parent_size))
            page_parent_ids = []
            for j, ptxt in enumerate(page_parents):
                pid = _chunk_id(pdf.name, page_num, f"par{j}", ptxt)
                page_parent_ids.append(pid)
                if pid not in live_parent_ids:
                    parent_ids.append(pid)
                    parent_texts.append(ptxt)
                    live_parent_ids.add(pid)

            for i, chunk in enumerate(chunk_words(page_text, child_size, child_overlap)):
                cid = _chunk_id(pdf.name, page_num, i, chunk)
                # Which parent covers this child: children are built from the
                # same page text, so find the parent window containing the
                # child's first word.
                start_word = i * (child_size - child_overlap)
                pidx = min(start_word // parent_size, len(page_parent_ids) - 1)
                ids.append(cid)
                texts.append(chunk)
                metas.append({
                    "source": pdf.name,
                    "page": page_num,
                    "parent_id": page_parent_ids[pidx],
                })
                live_ids.add(cid)
                file_chunks += 1
        if progress_cb:
            progress_cb(f"{pdf.name}: {file_chunks} chunks")

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

    return len(pdfs), collection.count()


def main() -> None:
    files, total = ingest_pdfs()
    print(f"{files} PDF(s) processed, {total} child chunks stored in {DB_DIR}/")


if __name__ == "__main__":
    main()
