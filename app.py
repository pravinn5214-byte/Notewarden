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
from retrieve import DEFAULT_MODE, MODES, invalidate_bm25_cache, retrieve, retrieve_multi
from decompose import Decomposition, decompose
from agent import answer_with_agent
from store import (  # noqa: F401  (re-exported for callers/tests)
    IndexCorruptError, LockHeldError, acquire_write_lock, db_path,
    get_client, get_collections, health_check, release_write_lock,
    reset_client, with_index_recovery,
)

DOCS_DIR = Path("docs")
DB_DIR = db_path()  # CHROMA_DIR env var, default chroma_db/


@st.cache_resource
def _cached_collections(db_dir: str):
    """Streamlit-cached handle on the two collections.

    Cleared (st.cache_resource.clear(), via _refresh_collections) after every
    write so the next read reopens through store.get_client() and never holds
    a stale HNSW segment.
    """
    return get_collections(db_dir)


def _refresh_collections() -> None:
    st.cache_resource.clear()
    reset_client()


def check_store_ready() -> list[str]:
    """Return a human-readable reason the app can't answer yet, or [] if ready."""
    if not DOCS_DIR.exists() or not (list(DOCS_DIR.glob("*.pdf"))
                                     + [p for ext in IMAGE_EXTENSIONS for p in DOCS_DIR.glob(ext)]):
        return ["No PDFs or images yet. Upload your notes in the sidebar, then click "
                "**Process documents**."]
    try:
        health_check()
    except Exception:
        return ["The index looks out of sync. Click **Repair index** in the "
                "sidebar."]
    try:
        client = get_client()
        count = with_index_recovery(lambda: client.get_collection("notes").count())
    except Exception:
        return ["Documents aren't indexed yet. Click **Process documents** in the "
                "sidebar."]
    if count == 0:
        return ["The index is empty. Click **Process documents** in the sidebar."]
    return []


# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"|"assistant", "text": ..., "sources": [...]}
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = uuid.uuid4().hex[:12]  # browser-session tag for query_log

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
                    _refresh_collections()
                    st.success(f"Deleted {doc_path.name}")
                except LockHeldError:
                    st.warning("The app (or another ingest) is running; stop it first.")
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
    if mode == "dense":
        st.caption("Dense mode can miss one topic in multi-topic questions. "
                   "Hybrid is recommended.")
    decompose_enabled = st.checkbox(
        "Split multi-topic questions",
        value=os.environ.get("DECOMPOSE", "on").lower() != "off",
        help="Splits multi-topic questions into one search per topic so each "
             "gets its own evidence.",
    )
    if decompose_enabled:
        st.caption("Adds one short LLM call for multi-part questions.")
    agent_mode = st.checkbox(
        "Agentic mode",
        value=os.environ.get("AGENT_MODE", "on").lower() != "off",
        help="Retries retrieval when results are weak and asks a clarifying "
             "question when stuck (max one retry and one clarification per turn).",
    )
    if agent_mode:
        st.caption("One extra LLM call only when retrieval is weak.")

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
            except LockHeldError:
                progress.empty()
                st.warning("The app (or another ingest) is running; stop it first.")
            except IndexCorruptError:
                progress.empty()
                st.error("The index looks out of sync. Click **Repair index** in the sidebar.")
            except Exception as exc:
                progress.empty()
                st.error(f"Ingestion failed: {exc}")
            else:
                invalidate_bm25_cache()
                _refresh_collections()

    st.divider()

    if st.button("🛠️ Repair index", use_container_width=True,
                 help="Deletes and rebuilds both collections from docs/ (OCR cache is reused, scanned pages are not re-OCR'd)"):
        with st.spinner("Rebuilding index from docs/..."):
            try:
                lock = acquire_write_lock()
            except LockHeldError:
                st.warning("The app (or another ingest) is running; stop it first.")
            else:
                try:
                    client = get_client()
                    for name in ("notes", "parents"):
                        try:
                            with_index_recovery(lambda n=name: client.delete_collection(n))
                        except Exception:
                            pass  # collection didn't exist yet
                    reset_client()
                    files, total = ingest_pdfs(use_ocr=True)
                    invalidate_bm25_cache()
                    _refresh_collections()
                    st.success(f"Index repaired: {files} file(s), {total} child chunks.")
                except LockHeldError:
                    st.warning("The app (or another ingest) is running; stop it first.")
                except IndexCorruptError as exc:
                    st.error(f"Repair failed: {exc}")
                except Exception as exc:
                    st.error(f"Repair failed: {exc}")
                finally:
                    release_write_lock(locals().get("lock"))

    st.divider()
    provider = os.environ.get("LLM_PROVIDER", "claude")
    st.caption(f"LLM provider: **{provider}**"
               + (f" ({os.environ.get('OLLAMA_MODEL', OLLAMA_DEFAULT_MODEL)})" if provider == "ollama" else ""))

    # Query log: the 10 weakest recent queries, to spot retrieval gaps.
    with st.expander("🔍 Query log", expanded=False):
        try:
            from query_log import enabled, rows_since
            if not enabled():
                st.caption("Query logging is off (QUERY_LOG=off).")
            else:
                rows = rows_since()
                logged = [r for r in rows if r["top_dense_score"] is not None][:10]
                if not logged:
                    st.caption("No queries logged yet.")
                else:
                    for r in logged:
                        q = r["original_query"] or ""
                        if len(q) > 80:
                            q = q[:77] + "..."
                        st.markdown(f"`{r['top_dense_score']:.2f}` · {q}")
                        bits = [f"{r['ts'][:16]}", r["retrieval_mode"] or "-"]
                        if r["gated_out"]:
                            bits.append("gated out")
                        if r["clarified"]:
                            bits.append("clarified")
                        if (r["retries"] or 0) > 0:
                            bits.append(f"{r['retries']} retry")
                        st.caption(" · ".join(bits))
                    st.caption(f"{len(rows)} query(ies) logged · `python log_report.py` for the full report")
        except Exception as exc:
            st.caption(f"Query log unavailable: {exc}")

def _render_sources(sources: list[dict], subqueries: list[str] | None = None,
                    supported: set | None = None) -> None:
    """Sources expander body: grouped by sub-question when decomposed."""
    with st.expander("Sources"):
        if subqueries and len(subqueries) > 1:
            supported = supported if supported is not None else set(range(len(subqueries)))
            by_sub: dict[int, list] = {}
            for s in sources:
                by_sub.setdefault(s.get("subquery_index", 0), []).append(s)
            for i, sub in enumerate(subqueries):
                st.markdown(f"**Sub-question {i + 1}:** {sub}")
                if i not in supported:
                    st.caption("no relevant notes found")
                for s in by_sub.get(i, []):
                    _source_line(s)
        else:
            for s in sources:
                _source_line(s)


def _source_line(s: dict) -> None:
    badge = f" · `OCR · {s['ocr_engine']}`" if s.get("ocr") and s.get("ocr_engine") else (" · `OCR`" if s.get("ocr") else "")
    st.markdown(f"**{s['source']}** — page {s['page']}{badge} — score `{s['score']:.2f}`")
    st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


def _render_trace(trace: dict) -> None:
    """Collapsed 'How I searched' expander with the agent trace (no document text)."""
    if not trace or not trace.get("steps"):
        return
    with st.expander("How I searched"):
        for entry in trace.get("steps", []):
            name = entry.get("step", "?")
            bits = [f"**{name}**"]
            bits += [f"{k}: {v}" for k, v in entry.items()
                     if k != "step" and v not in (None, False)]
            st.markdown(" · ".join(bits))
        if trace.get("queries_tried"):
            st.caption("Queries tried: " + " → ".join(trace["queries_tried"]))
        if trace.get("clarify"):
            st.caption(f"Clarified because: {trace['clarify'].get('reason', 'unknown')}")


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
            _render_sources(msg["sources"], msg.get("subqueries"), msg.get("supported"))

if question := st.chat_input("Ask about your notes...", disabled=bool(blockers)):
    import time as _time
    _plain_turn_start = _time.monotonic()
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        allowed_names = [p.name for p in selected]
        history_msgs = [m for m in st.session_state.messages
                        if m["role"] in ("user", "assistant")][-8:]
        if agent_mode:
            try:
                result = answer_with_agent(
                    question, history=history_msgs, mode=mode, top_k=top_k,
                    files=allowed_names,
                )
            except IndexCorruptError:
                st.error("The index looks out of sync. Click **Repair index** in the sidebar.")
                st.stop()
            except RuntimeError as exc:  # missing/empty store
                st.error(str(exc))
                st.stop()
            st.markdown(result.answer_or_question)
            if result.sources:
                _render_sources(result.sources,
                                result.trace.get("decomposition_subqueries"),
                                result.trace.get("supported"))
            _render_trace(result.trace)
            if result.action == "answer":
                st.session_state.messages.append(
                    {"role": "assistant", "text": result.answer_or_question,
                     "sources": result.sources})
            else:
                st.session_state.messages.append(
                    {"role": "assistant", "text": result.answer_or_question,
                     "clarify": True})
            st.stop()
        with st.spinner("Splitting question..." if decompose_enabled else None):
            decomposition = decompose(question) if decompose_enabled else None
        try:
            if decomposition is not None and len(decomposition.subqueries) > 1:
                hits, supported = retrieve_multi(
                    decomposition.subqueries, mode=mode, top_k=top_k,
                    files=allowed_names,
                )
            else:
                hits = retrieve(
                    question, k=top_k, mode=mode,
                    allowed_sources=allowed_names,
                )
                supported = {0} if hits else set()
        except IndexCorruptError:
            st.error("The index looks out of sync. Click **Repair index** in the sidebar.")
            st.stop()
        except RuntimeError as exc:  # missing/empty store
            st.error(str(exc))
            st.stop()

        with st.spinner("Thinking..." if mode == "dense" else "Re-ranking & thinking..."):
            try:
                answer = generate_answer(question, hits, decomposition=decomposition)
            except RuntimeError as exc:  # missing key, Ollama down, API error
                st.error(f"LLM error: {exc}")
                st.stop()
        st.markdown(answer)
        _render_sources(hits, decomposition.subqueries if decomposition else None, supported)

        # Query log (best-effort, never breaks the app).
        import time as _time
        from query_log import log_query, results_payload, top_dense, gated
        try:
            subs = decomposition.subqueries if decomposition else []
            from generate import MIN_SCORE as _gate
            log_query(
                session_id=st.session_state.get("session_id", ""),
                original_query=question,
                retrieval_mode=mode,
                top_k=top_k,
                decompose_used=bool(subs and len(subs) > 1),
                agent_mode=False,
                sub_queries=subs if len(subs) > 1 else None,
                results=results_payload(hits),
                top_dense_score=top_dense(hits),
                gated_out=gated(hits, _gate),
                unsupported_topics=[subs[i] for i in range(len(subs))
                                    if i not in supported] if len(subs) > 1 else [],
                llm_provider=os.environ.get("LLM_PROVIDER", "claude"),
                latency_ms=(_time.monotonic() - _plain_turn_start) * 1000.0,
                answer=answer,
            )
        except Exception:  # logging must never break the app
            pass

    st.session_state.messages.append(
        {"role": "assistant", "text": answer, "sources": hits,
         "subqueries": decomposition.subqueries if decomposition else None,
         "supported": supported if decomposition else None}
    )
