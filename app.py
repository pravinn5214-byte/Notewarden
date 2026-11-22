"""Streamlit UI for the RAG Study Assistant.

Sidebar: upload PDFs, process documents, tune chunk size and top-k.
Main: chat with grounded answers; each answer has an expandable Sources panel.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from generate import FALLBACK_MESSAGE, OLLAMA_DEFAULT_MODEL, generate_answer
from ingest import IMAGE_EXTENSIONS, ingest_pdfs
from ocr import is_tesseract_available
from retrieve import DEFAULT_MODE, MODES, invalidate_bm25_cache, retrieve

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"


def check_store_ready() -> list[str]:
    """Return a human-readable reason the app can't answer yet, or [] if ready."""
    if not DOCS_DIR.exists() or not (list(DOCS_DIR.glob("*.pdf"))
                                     + [p for ext in IMAGE_EXTENSIONS for p in DOCS_DIR.glob(ext)]):
        return ["No PDFs or images yet. Upload your notes in the sidebar, then click "
                "**Process documents**."]
    import chromadb
    try:
        count = chromadb.PersistentClient(path=DB_DIR).get_collection("notes").count()
    except Exception:
        return ["Documents aren't indexed yet. Click **Process documents** in the "
                "sidebar."]
    if count == 0:
        return ["The index is empty. Click **Process documents** in the sidebar."]
    return []


# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"assistant", "text": ..., "sources": [...]}

# --- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Study Assistant")

    upload_types = ["pdf", *IMAGE_EXTENSIONS]
    uploaded = st.file_uploader(
        "Upload PDFs or images (scanned notes)", type=upload_types,
        accept_multiple_files=True,
    )
    if uploaded and st.button("Save uploads to docs/"):
        DOCS_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            (DOCS_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploaded)} file(s) to docs/. Click Process documents.")

    # The uploader resets on refresh (Streamlit widget behavior), so docs/ on
    # disk is the source of truth for what's in your library.
    st.subheader("Your documents")
    docs = sorted(
        list(DOCS_DIR.glob("*.pdf"))
        + [p for ext in IMAGE_EXTENSIONS for p in DOCS_DIR.glob(ext)]
    ) if DOCS_DIR.exists() else []
    if not docs:
        st.caption("No documents yet.")

    selected: list[Path] = []
    for doc_path in docs:
        col_check, col_name, col_del = st.columns([0.1, 0.75, 0.15])
        checked = col_check.checkbox(f"Use {doc_path.name}", value=True, key=f"sel_{doc_path.name}", label_visibility="collapsed")
        icon = "🖼" if doc_path.suffix.lower() in IMAGE_EXTENSIONS else "📄"
        col_name.markdown(f"{icon} `{doc_path.name}`")
        if col_del.button("🗑", key=f"del_{doc_path.name}", help=f"Delete {doc_path.name}"):
            doc_path.unlink()
            with st.spinner("Updating index..."):
                try:
                    ingest_pdfs()  # drops the deleted file's chunks
                    invalidate_bm25_cache()
                    st.success(f"Deleted {doc_path.name}")
                except FileNotFoundError:
                    pass  # nothing left to index
            st.rerun()
        if checked:
            selected.append(doc_path)

    st.divider()

    tesseract_ok = is_tesseract_available()
    if not tesseract_ok:
        st.warning(
            "Tesseract not found — OCR is disabled and scanned/image pages "
            "will be skipped. Install it ([instructions](https://github.com/UB-Mannheim/tesseract/wiki) "
            "for Windows; `brew install tesseract` on macOS, "
            "`sudo apt install tesseract-ocr` on Linux), or set `TESSERACT_CMD` in `.env`."
        )
    use_ocr = st.checkbox("Use OCR for scanned pages", value=True, disabled=not tesseract_ok)

    engine_options = ["tesseract", "claude", "ollama", "auto"]
    ocr_engine = st.selectbox(
        "OCR engine",
        engine_options,
        index=engine_options.index(os.environ.get("OCR_ENGINE", "tesseract"))
        if os.environ.get("OCR_ENGINE", "tesseract") in engine_options else 0,
        help="tesseract = local, fast, free, best on printed text · "
             "claude = Anthropic vision model (best on handwriting, page images "
             "are sent to the API) · ollama = local vision model · "
             "auto = tesseract, escalating low-confidence pages to OCR_VISION_ENGINE",
    )
    if ocr_engine == "claude":
        st.caption("☁️ Page images are sent to the Anthropic API (needs ANTHROPIC_API_KEY).")
    elif ocr_engine in ("ollama", "auto"):
        st.caption("💻 Runs locally — nothing leaves your machine.")
    else:
        st.caption("💻 Runs locally — nothing leaves your machine.")

    chunk_size = st.slider("Child chunk size (words)", 80, 300, 120, step=20)
    parent_size = st.slider("Parent chunk size (words)", 200, 1000, 500, step=50)
    top_k = st.slider("Top-k chunks to retrieve", 1, 10, 4)
    mode = st.selectbox(
        "Retrieval mode",
        MODES,
        index=MODES.index(os.environ.get("RETRIEVAL_MODE", DEFAULT_MODE))
        if os.environ.get("RETRIEVAL_MODE", DEFAULT_MODE) in MODES else MODES.index(DEFAULT_MODE),
        help="dense = embeddings only (baseline) · hybrid = dense + BM25 keyword "
             "search fused with RRF · hybrid_rerank = hybrid + cross-encoder "
             "re-ranking (answers use the larger parent chunk)",
    )

    if st.button("🗑️ Process documents", use_container_width=True):
        if not selected:
            st.error("No documents selected. Tick the checkboxes next to the files you want to use.")
        else:
            progress = st.progress(0.0, text="Starting...")
            try:
                state = [0.0]  # mutable: closure at module scope can't use nonlocal

                def cb(status: str) -> None:
                    state[0] = min(0.95, state[0] + 0.15)
                    progress.progress(state[0], text=status)

                with st.spinner("OCRing scanned pages..." if use_ocr and tesseract_ok else None):
                    files, total = ingest_pdfs(
                        child_size=chunk_size, parent_size=parent_size,
                        progress_cb=cb, files=selected, use_ocr=use_ocr,
                        engine_name=ocr_engine,
                    )
                invalidate_bm25_cache()
                progress.progress(1.0, text=f"Done: {files} file(s), {total} chunks")
                st.success(f"Indexed {files} file(s) → {total} child chunks.")
                st.info("If you changed chunk size, re-run processing to rebuild "
                        "the index with the new setting.")
            except Exception as exc:
                progress.empty()
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    provider = os.environ.get("LLM_PROVIDER", "claude")
    st.caption(f"LLM provider: **{provider}**"
               + (f" ({os.environ.get('OLLAMA_MODEL', OLLAMA_DEFAULT_MODEL)})" if provider == "ollama" else ""))

# --- Main: chat ----------------------------------------------------------
st.title("💬 Chat with your notes")

blockers = check_store_ready()
for b in blockers:
    st.warning(b)

# Replaying history keeps earlier answers (with their Sources sections) visible.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    badge = f" · `OCR · {s['ocr_engine']}`" if s.get("ocr") and s.get("ocr_engine") else (" · `OCR`" if s.get("ocr") else "")
                    st.markdown(f"**{s['source']}** — page {s['page']}{badge} — score `{s['score']:.2f}`")
                    st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

if question := st.chat_input("Ask about your notes...", disabled=bool(blockers)):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # Retrieve deeper than k, then keep only chunks from the files
            # currently ticked, so answers only draw on the selection.
            hits = retrieve(
                question, k=top_k * 4, mode=mode,
                allowed_sources=[p.name for p in selected],
            )
            allowed = {p.name for p in selected}
            hits = [h for h in hits if h["source"] in allowed][:top_k]
        except RuntimeError as exc:  # missing/empty store
            st.error(str(exc))
            st.stop()

        with st.spinner("Thinking..." if mode == "dense" else "Re-ranking & thinking..."):
            try:
                answer = generate_answer(question, hits)
            except RuntimeError as exc:  # missing key, Ollama down, API error
                st.error(f"LLM error: {exc}")
                st.stop()

        st.markdown(answer)
        with st.expander("Sources"):
            for s in hits:
                badge = f" · `OCR · {s['ocr_engine']}`" if s.get("ocr") and s.get("ocr_engine") else (" · `OCR`" if s.get("ocr") else "")
                parts = [f"**{s['source']}** — page {s['page']}{badge}",
                         f"dense `{s.get('dense_score', float('nan')):.2f}`"]
                if "bm25_rank" in s:
                    parts.append(f"BM25 rank {s['bm25_rank']}")
                if "rerank_score" in s:
                    parts.append(f"rerank `{s['rerank_score']:.2f}`")
                st.markdown(" | ".join(parts))
                st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "text": answer, "sources": hits if answer != FALLBACK_MESSAGE else hits}
    )
