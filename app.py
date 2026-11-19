"""Streamlit UI for the RAG Study Assistant.

Sidebar: upload PDFs, process documents, tune chunk size and top-k.
Main: chat with grounded answers; each answer has an expandable Sources panel.
"""
import os
from pathlib import Path

import streamlit as st

from generate import FALLBACK_MESSAGE, OLLAMA_DEFAULT_MODEL, generate_answer
from ingest import ingest_pdfs
from retrieve import retrieve

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"


def check_store_ready() -> list[str]:
    """Return a human-readable reason the app can't answer yet, or [] if ready."""
    if not DOCS_DIR.exists() or not list(DOCS_DIR.glob("*.pdf")):
        return ["No PDFs yet. Upload your notes in the sidebar, then click "
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

    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Save uploads to docs/"):
        DOCS_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            (DOCS_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploaded)} file(s) to docs/. Click Process documents.")

    # The uploader resets on refresh (Streamlit widget behavior), so docs/ on
    # disk is the source of truth for what's in your library.
    st.subheader("Your documents")
    pdfs = sorted(DOCS_DIR.glob("*.pdf")) if DOCS_DIR.exists() else []
    if not pdfs:
        st.caption("No documents yet.")

    selected: list[Path] = []
    for pdf in pdfs:
        col_check, col_name, col_del = st.columns([0.1, 0.75, 0.15])
        checked = col_check.checkbox(f"Use {pdf.name}", value=True, key=f"sel_{pdf.name}", label_visibility="collapsed")
        col_name.markdown(f"📄 `{pdf.name}`")
        if col_del.button("🗑", key=f"del_{pdf.name}", help=f"Delete {pdf.name}"):
            pdf.unlink()
            with st.spinner("Updating index..."):
                try:
                    ingest_pdfs()  # drops the deleted file's chunks
                    st.success(f"Deleted {pdf.name}")
                except FileNotFoundError:
                    pass  # nothing left to index
            st.rerun()
        if checked:
            selected.append(pdf)

    st.divider()

    chunk_size = st.slider("Chunk size (words)", 100, 1000, 300, step=50)
    top_k = st.slider("Top-k chunks to retrieve", 1, 10, 4)

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

                files, total = ingest_pdfs(chunk_size=chunk_size, progress_cb=cb, files=selected)
                progress.progress(1.0, text=f"Done: {files} file(s), {total} chunks")
                st.success(f"Indexed {files} file(s) → {total} chunks.")
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
                    st.markdown(f"**{s['source']}** — page {s['page']} — score `{s['score']:.2f}`")
                    st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

if question := st.chat_input("Ask about your notes...", disabled=bool(blockers)):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # Retrieve deeper than k, then keep only chunks from the files
            # currently ticked, so answers only draw on the selection.
            hits = retrieve(question, k=top_k * 4)
            allowed = {p.name for p in selected}
            hits = [h for h in hits if h["source"] in allowed][:top_k]
        except RuntimeError as exc:  # missing/empty store
            st.error(str(exc))
            st.stop()

        with st.spinner("Thinking..."):
            try:
                answer = generate_answer(question, hits)
            except RuntimeError as exc:  # missing key, Ollama down, API error
                st.error(f"LLM error: {exc}")
                st.stop()

        st.markdown(answer)
        with st.expander("Sources"):
            for s in hits:
                st.markdown(f"**{s['source']}** — page {s['page']} — score `{s['score']:.2f}`")
                st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "text": answer, "sources": hits if answer != FALLBACK_MESSAGE else hits}
    )
