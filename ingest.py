"""Steps 1-2 of the pipeline: load PDFs -> chunk -> embed -> store in Chroma.

Idempotent: every chunk gets a deterministic ID derived from its file, page
and index, so rerunning ingestion updates changed chunks in place instead of
creating duplicates. Chunks whose source disappeared (deleted file or page)
are removed from the store.

Usage:
    1. Put your PDFs in the docs/ folder.
    2. Run: python ingest.py
"""
import hashlib
from pathlib import Path
from typing import Callable, Iterator

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "notes"
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs locally
CHUNK_SIZE = 300  # words per chunk (a rough stand-in for tokens)
OVERLAP = 50      # words shared between neighbouring chunks


def load_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) for each page that has extractable text."""
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield page_num, text


def chunk_words(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> Iterator[str]:
    """Split text into overlapping word windows."""
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk size ({size})")
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Deterministic ID; the text hash makes edits to a page produce a new
    ID (old one gets cleaned up), identical reruns produce the same one."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}-p{page}-c{index}-{digest}"


def ingest_pdfs(
    docs_dir: Path = DOCS_DIR,
    db_dir: str = DB_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    progress_cb: Callable[[str], None] | None = None,
    files: list[Path] | None = None,
) -> tuple[int, int]:
    """Ingest the given PDFs (default: every PDF in docs_dir).

    The index is made to match the selection exactly: chunks from files not
    in `files` are removed, so deselected files stop contributing to answers.
    Returns (files_processed, chunks_stored).

    progress_cb, if given, is called with a short status string per PDF
    (used by the Streamlit UI as a progress indicator).
    """
    pdfs = sorted(files) if files is not None else sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}/ - add some first.")

    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    live_ids: set[str] = set()  # chunks that should exist after this run
    for pdf in pdfs:
        if progress_cb:
            progress_cb(f"Reading {pdf.name}...")
        file_chunks = 0
        for page_num, page_text in load_pages(pdf):
            for i, chunk in enumerate(chunk_words(page_text, chunk_size, overlap)):
                cid = _chunk_id(pdf.name, page_num, i, chunk)
                ids.append(cid)
                texts.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})
                live_ids.add(cid)
                file_chunks += 1
        if progress_cb:
            progress_cb(f"{pdf.name}: {file_chunks} chunks")

    # Re-embed only chunks the store doesn't already have.
    existing = set(collection.get(ids=list(live_ids), include=[])["ids"]) if live_ids else set()
    new_idx = [i for i, cid in enumerate(ids) if cid not in existing]
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

    # Remove chunks whose file/page no longer exists (deleted or edited PDFs).
    stale = [cid for cid in collection.get(include=[])["ids"] if cid not in live_ids]
    if stale:
        collection.delete(ids=stale)
    if progress_cb:
        progress_cb(f"Removed {len(stale)} stale chunk(s)")

    return len(pdfs), collection.count()


def main() -> None:
    files, total = ingest_pdfs()
    print(f"{files} PDF(s) processed, {total} chunks stored in {DB_DIR}/")


if __name__ == "__main__":
    main()
