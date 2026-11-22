# RAG Study Assistant — Features & Roadmap

## ✅ Current features (implemented)

| Feature | Where | Notes |
|---|---|---|
| PDF upload & library | `app.py` sidebar | Upload multiple PDFs → saved to `docs/`; per-file delete (🗑) also removes its chunks from the index |
| Per-file selection | `app.py` sidebar | Checkboxes decide which files answers can draw on; BM25 respects the filter |
| PDF ingestion & chunking | `ingest.py` | PyMuPDF extraction + `clean_text()` (camelCase/letter-digit splits, de-hyphenation) → child/parent word windows; fused-token diagnostic on ingest |
| OCR for scanned pages & images | `ocr.py` + `ingest.py` | Pages with < `OCR_MIN_CHARS` (default 30) extracted chars are OCR'd; `.png`/`.jpg`/`.jpeg` uploads become 1-page OCR'd documents; `ocr: true/false` + `ocr_engine` chunk metadata + `OCR · <engine>` badge in Sources; swappable `OcrEngine` interface |
| OCR engines (incl. handwriting) | `ocr.py` | `tesseract` (local, free, print-quality) · `claude` (Anthropic vision API, best on cursive; `OCR_CLAUDE_MODEL`) · `ollama` (local vision model, `OCR_OLLAMA_MODEL`, respects `OLLAMA_HOST`) · `auto` (tesseract first; mean word confidence < `OCR_CONF_THRESHOLD` [60] or garbage text escalates to `OCR_VISION_ENGINE`); shared transcription prompt; retries w/ backoff; vision failure falls back to Tesseract |
| OCR engine cache | `ocr_cache/` | Cached per (file hash, page, engine, model) — switching engines never returns stale text; old tesseract-only entries stay valid |
| Vision cost control | `ingest.py` | Ingest prints "~N pages will use the vision model"; `OCR_MAX_VISION_PAGES` (default 50) per run, excess pages fall back to Tesseract with a warning; sidebar shows whether pages leave the machine |
| OCR engine comparison | `ocr_compare.py` | CER/WER + time per engine against a ground-truth .txt for one image/PDF page |
| Idempotent indexing | `ingest.py` | Content-derived chunk IDs: re-ingesting never duplicates; stale children & parents removed; metadata refreshed |
| Vector store | `chroma_db/` | ChromaDB persistent client, cosine distance (`notes` children + `parents` context) |
| Semantic retrieval | `retrieve.py` | `all-MiniLM-L6-v2` sentence-transformer; top-k chunks with similarity scores |
| Hybrid search | `retrieve.py` | BM25 (rank_bm25) over children + dense, fused with RRF (k=60), top-20 each; punctuation-aware tokenizer with stopwords |
| Re-ranking | `retrieve.py` | Cross-encoder `ms-marco-MiniLM-L-6-v2`; parents deduped, best child's score kept |
| Parent-child chunks | `ingest.py` | 120-word children (matched) + 500-word page-aware parents (LLM context); both content-derived IDs, both cleaned on delete |
| Retrieval modes | `app.py` sidebar | `dense` / `hybrid` / `hybrid_rerank` (default); also `RETRIEVAL_MODE` env var |
| Relevance gate | `generate.py` | Gates on dense cosine (mode-independent), `MIN_SCORE` = 0.3 |
| Sources panel | `app.py` | Shows dense score, BM25 rank, rerank score, file, page per result |
| Grounded generation | `generate.py` | LLM answers only from context, cites as `[file p.N]`; multi-topic questions answered per part (`**Part N — <topic>**` headings, each part cited or flagged as unsupported) |
| Multi-source top-k | `app.py` | When a question matches chunks from several selected files, hits are round-robined by source before the top-k cut, so one topic can't crowd out the other file's chunks |
| Two LLM providers | `generate.py` | `claude` (Anthropic API) or `ollama` (local, e.g. `llama3.2`), picked via `LLM_PROVIDER` |
| Chat UI | `app.py` | Streamlit chat with message history (per browser session), Sources expander per answer |
| Tunable parameters | `app.py` sidebar | Child/parent chunk size, top-k (1–10), retrieval mode |
| Retrieval evaluation | `evaluate.py` + `eval_questions.json` | 45 tagged questions (keyword/semantic); all 3 modes at k=3/5 with bootstrap 95% CIs; per-tag breakdown; results in `eval_results.md` |
| Parent-child ablation | `ablation.py` | LLM-as-judge check that parent vs child context contains the answer |
| Tests | `tests/test_retrieval.py`, `tests/test_ocr.py`, `tests/test_ocr_vision.py` | Round trip, idempotency, delete cleanup, RRF maths, parent dedupe, tokenizer, clean_text; OCR: image-only PDF ingests + is retrievable, text-layer PDF not OCR'd, cache hit, graceful no-Tesseract (skipped if Tesseract absent); vision: auto escalation, per-engine cache keys, API-failure fallback, page cap, engine metadata (all fakes, no network) |
| CLI pipeline | all modules | `python ingest.py`, `retrieve.py "q" --mode hybrid_rerank`, `generate.py "q"` — full pipeline without the UI |
| Config | `.env` | `LLM_PROVIDER`, `OLLAMA_MODEL`, `ANTHROPIC_API_KEY`, `RETRIEVAL_MODE`, `TESSERACT_CMD`, `OCR_LANG`, `OCR_MIN_CHARS`, `OCR_PREPROCESS`, `OCR_ENGINE`, `OCR_VISION_ENGINE`, `OCR_CONF_THRESHOLD`, `OCR_CLAUDE_MODEL`, `OCR_OLLAMA_MODEL`, `OCR_MAX_VISION_PAGES` |

## 🔮 What we could do next

### Persistence & history (your MongoDB idea — removed for now, can re-add)
- [ ] **Chat history persistence** — currently history lives in `st.session_state` and dies on refresh/restart. Options:
  - **MongoDB** (local or Atlas) + **Compass** GUI for browsing — sessions collection, per-session & delete-all buttons. Needs `pymongo` + a running `mongod`.
  - **SQLite** — zero-install, single file, good enough for one user.
  - Simplest: JSON files per session in `chat_history/`.
- [ ] Session switcher (continue an old conversation), rename sessions
- [ ] "Clear current chat" / "Delete all history" buttons

### Docker & deployment
- [ ] **Dockerfile** for the Streamlit app
- [ ] **docker-compose**: app + Ollama + (optional) MongoDB in one stack
- [ ] `OLLAMA_HOST=http://ollama:11434` config so containers talk to each other

### Retrieval quality
- [x] **Hybrid search** — BM25 keyword search fused with vector search (RRF, k=60) — `retrieve.py` `hybrid` mode
- [x] **Re-ranking** — cross-encoder `ms-marco-MiniLM-L-6-v2` on the fused top-20 — `hybrid_rerank` mode (default)
- [x] **Parent-child chunks** — 120-word children for matching, 500-word page-aware parents for LLM context — `ingest.py`
- [x] **OCR for scanned pages & images** — Tesseract + vision engines (Claude / Ollama) via `ocr.py` (`OcrEngine` interface), with `auto` escalation and engine comparison (`ocr_compare.py`)
- [ ] Better OCR models — plug a stronger vision LLM behind `ocr.py`'s `OcrEngine` interface
- [ ] Better chunking — respect headings/paragraph boundaries instead of fixed word windows
- [ ] Metadata filters (by file, page range) in the UI

### UX
- [ ] Show token/word count of the indexed docs
- [ ] Export a chat as Markdown
- [ ] Multiple named "notebooks" (separate Chroma collections per topic)
- [ ] Support other file types: DOCX, TXT, Markdown, PPTX

### Model & cost
- [ ] Show which model answered + rough token usage
- [ ] Streaming answers (Ollama & Anthropic both support it)
- [ ] Conversation memory — feed the last N turns back so follow-up questions work

### Ops
- [ ] Tests for ingest/retrieve round-trip (currently none)
- [ ] Logging of queries + retrieval scores to spot weak spots
