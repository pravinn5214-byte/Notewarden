# RAG Study Assistant

Chat with your own PDF notes. Ingest PDFs, retrieve the relevant passages with
hybrid vector + keyword search, re-rank them with a cross-encoder, and get
answers grounded in those notes — with page citations.

## How it works

```
docs/*.pdf -> ingest.py (PyMuPDF extraction + clean_text) -> ChromaDB "notes"   (CHILD chunks, ~120 words, embedded)
                                                          -> ChromaDB "parents" (PARENT chunks, ~500 words, context)
                              |
question -------------------> retrieve.py (mode: dense | hybrid | hybrid_rerank)
                              |    dense: cosine top-k
                              |    hybrid: dense + BM25 fused with RRF (k=60)
                              |    hybrid_rerank: hybrid + cross-encoder, dedup by parent
                              |
                    generate.py (Claude or Ollama)  <- parent text as context
                              |
                    answer with [file p.N] citations
```

- **Parent-child chunks**: small *child* chunks (default 120 words, 20 overlap)
  are indexed for matching; each carries `source`, `page` and `parent_id`.
  The matching child's *parent* chunk (default 500 words, page-aware) is what
  the LLM reads, so answers cite the right page but reason over coherent
  context. Sizes are configurable in the sidebar.
- **Text extraction**: PyMuPDF (chosen over pypdf after measurement: pypdf
  dropped inter-word spaces on 170.pdf, producing 2.16% fused tokens like
  `CommonlyinvolvedinDNA`; PyMuPDF yields 0.01%). `clean_text()` additionally
  splits camelCase/letter-digit joins, re-joins hyphenated line breaks, and
  collapses whitespace before chunking — dense, BM25 and parent text all use
  the same cleaned words. Ingestion prints the fused-token % before/after.
- **OCR (scanned pages & images)**: pages with (almost) no text layer — fewer
  than `OCR_MIN_CHARS` (default 30) non-whitespace chars — are rendered at 300
  DPI with PyMuPDF and OCR'd with Tesseract (`ocr.py`), as are standalone
  `.png`/`.jpg`/`.jpeg` uploads (1-page documents). OCR text goes through the
  same `clean_text()` and chunking, so page numbers and citations stay
  correct; chunk metadata records `ocr: true/false` and the Sources panel
  shows an `OCR` badge. OCR text is cached per (file hash, page) under
  `ocr_cache/`, so re-ingesting never re-OCRs an unchanged page.
  **Quality note**: Tesseract is good on printed/scanned text but poor on
  cursive handwriting. `ocr.py` is built around a small `OcrEngine` interface,
  so a stronger backend (a vision-capable LLM or TrOCR) can be swapped in
  later without touching the rest of the pipeline.
- **BM25 tokenizer**: lowercase, punctuation-stripped, keeps intra-token
  hyphens/underscores (`INV-YYYY-XXXX`, `G-quadruplex` survive), drops a small
  stopword list. Same tokenizer for index and queries.
- **Hybrid search**: BM25 (rank_bm25) over the child chunks — built from the
  Chroma contents and cached — fused with dense retrieval using Reciprocal
  Rank Fusion (k=60), taking top-20 from each.
- **Re-ranking**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  re-scores the fused top-20; the top-k is cut from that. Results are
  deduplicated by parent: each parent appears at most once, with its best
  child's score.
- **Retrieval modes**: `dense` (original baseline), `hybrid`, `hybrid_rerank`
  (default). Pick in the sidebar or set `RETRIEVAL_MODE`.
- **Idempotent ingestion**: chunk IDs are content-derived, so rerunning
  ingestion never duplicates chunks — it only embeds new/changed chunks,
  refreshes metadata, and removes stale ones (children *and* parents) from
  deleted/edited files.
- **Grounding**: the LLM is instructed to answer only from the provided
  context and to say "I couldn't find that in your notes" otherwise. The
  relevance gate (`MIN_SCORE`, default 0.3) uses the **dense cosine score** of
  the best candidate in every mode — cosine is calibrated (0..1, related text
  ≈ 0.3+), while RRF scores peak near 0.03 and cross-encoder scores are often
  negative, so gating on either would reject good results.
- **Multi-topic questions**: a question touching several topics/files is
  answered per part (`**Part 1 — <topic>**, **Part 2 — …**`), and the app
  round-robins retrieval hits by source before the top-k cut, so one topic
  can't crowd the other out. Prefer `hybrid`/`hybrid_rerank` over `dense` for
  such questions — dense averaging loses the keyword-only topic.

## Setup

Requires Python 3.10+.

```
python -m pip install -r requirements.txt
```

Set the provider (default is `claude`):

```
# Anthropic API
set LLM_PROVIDER=claude
set ANTHROPIC_API_KEY=sk-ant-...

# or a local model via Ollama (https://ollama.com)
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=llama3.1        # optional, this is the default
```

(On bash use `export` instead of `set`.)

### Setting up OCR

OCR (for scanned pages and image notes) needs the Tesseract binary — the
Python side (`pytesseract`, `Pillow`) comes from `requirements.txt`.

**Windows**
1. Install the [UB Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki).
2. In the installer, under *Additional language data*, tick **English**
   (and any other languages you read).
3. Add the install folder (default `C:\Program Files\Tesseract-OCR`) to PATH,
   or set the binary path in `.env`:
   ```
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
   (If it's at the default path, the app finds it even without PATH.)
4. Verify in a new terminal: `tesseract --version`

**macOS**

```
brew install tesseract
```

**Linux (Debian/Ubuntu)**

```
sudo apt install tesseract-ocr
```

The app detects Tesseract at startup; if it's missing you get a one-time
warning and scanning/image pages are skipped (the rest still ingests).

Optional `.env` settings: `OCR_LANG` (default `eng`; use e.g. `eng+deu` for
multiple languages), `OCR_MIN_CHARS` (page treated as scanned below this many
extracted chars), `OCR_PREPROCESS` (`contrast` or `threshold` for faint
scans).

### Handwritten notes

Tesseract is good on printed/scanned text but weak on cursive. For
handwritten pages, pick a vision engine:

```
# in .env
OCR_ENGINE=claude          # or ollama, or auto
OCR_CLAUDE_MODEL=claude-sonnet-4-20250514
OCR_OLLAMA_MODEL=llama3.2-vision   # then: ollama pull llama3.2-vision
```

| Engine | Quality on handwriting | Cost | Privacy | Speed |
|---|---|---|---|---|
| `tesseract` | Poor on cursive | Free, local | Local | Fast (~1 s/page) |
| `claude` | Best (handles cursive, math, diagrams) | API $ per page | Page images sent to Anthropic | ~3-8 s/page |
| `ollama` | Good with a vision model pulled (e.g. `llama3.2-vision`) | Free, local | Local | ~5-20 s/page (GPU-dependent) |
| `auto` | Best of both: Tesseract keeps confident printed pages, escalates shaky ones to `OCR_VISION_ENGINE` | Mostly free | Mostly local | Fast on printed, slow on escalated pages |

- **`auto` mode** runs Tesseract first and computes mean word confidence
  (`pytesseract.image_to_data`). Pages scoring below `OCR_CONF_THRESHOLD`
  (default 60) or producing mostly non-dictionary text are re-transcribed by
  the vision engine in `OCR_VISION_ENGINE` (claude|ollama).
- **Caching**: OCR text is cached per (file hash, page, engine, model) under
  `ocr_cache/` — switching engines never returns another engine's stale text.
- **Cost control**: `OCR_MAX_VISION_PAGES` (default 50) caps how many pages
  per ingest run use a vision engine; the rest fall back to Tesseract with a
  warning. The ingest prints "~N pages will use the vision model" up front.
- **Failures**: vision calls retry with backoff on rate limits/transient
  errors; a page that still fails falls back to Tesseract. Ingestion never
  crashes on an API outage.
- **Engine choice in the app**: pick it in the sidebar; a caption there tells
  you whether pages leave the machine.

**Tips for photographing notes** (applies to every engine, most of all the
vision ones):
- Good, even light; avoid shadows across the page.
- Shoot the page flat and straight-on; one page per image.
- High contrast: dark pen on light paper; avoid glare.
- Fill the frame with the page; higher resolution beats zooming.

**Measuring your handwriting**: `ocr_compare.py` scores engines against a
ground-truth transcription of one page:

```
python ocr_compare.py my_notes.png truth.txt                  # all engines
python ocr_compare.py scanned.pdf truth.txt --page 3 --engines claude,ollama
```

It prints CER/WER (lower is better) and time per page.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `tesseract is not installed or it is not in your PATH` (pytesseract error) | Install Tesseract (above) or set `TESSERACT_CMD` in `.env` to the full binary path |
| Missing language data / `Please make sure the TESSDATA_PREFIX environment variable is set correctly` | The language pack wasn't installed. Re-run the installer (Windows) and tick the language, or install e.g. `tesseract-ocr-deu` (Linux); point `TESSDATA_PREFIX` at the `tessdata` folder if it lives elsewhere |
| Handwriting recognized poorly | Expected — Tesseract is weak on cursive. Set `OCR_ENGINE=claude` (or `auto`) in `.env`, or try `OCR_PREPROCESS=threshold` for faint scans |
| `ANTHROPIC_API_KEY is not set` with `OCR_ENGINE=claude` | Set the key in `.env`, or switch to `OCR_ENGINE=ollama` (local) |
| `Ollama vision OCR failed` / model not found | `ollama pull llama3.2-vision` (or set `OCR_OLLAMA_MODEL` to a vision model you have), and check Ollama is running |
| Vision costs too high on big scans | Lower `OCR_MAX_VISION_PAGES`, or use `OCR_ENGINE=auto` so only shaky pages hit the API |

## Run

```
streamlit run app.py
```

Then in the sidebar: upload PDFs (they're saved to `docs/`), click
**Process documents**, and ask questions in the chat. Each answer has an
expandable **Sources** panel showing file, page, dense score, BM25 rank and
cross-encoder score (the latter two only in the hybrid modes).

### Command-line usage

```
python ingest.py                                 # index docs/ without the UI
python retrieve.py "what is normalization?"      # default mode (hybrid_rerank)
python retrieve.py "what is normalization?" --mode dense -k 6
python generate.py "what is normalization?"      # full RAG answer in the terminal
```

## Evaluation

`evaluate.py` compares retrieval modes — dense vs hybrid vs hybrid_rerank — at
k=3 and k=5 on hit rate@k and MRR, with bootstrap 95% confidence intervals and
a keyword/semantic per-tag breakdown:

```
python evaluate.py              # or: python evaluate.py 120 300
python evaluate.py --ablation   # + parent-vs-child LLM-judge ablation
```

It builds an index in `eval_chroma_db_<size>/` (your app index is untouched)
for each child chunk size, runs every question from `eval_questions.json` in
every mode, and saves tables to `eval_results.md`. Latest headline (45
questions, child size 120): hybrid_rerank MRR 0.856 vs dense 0.667 at k=3 —
significant by paired bootstrap (diff +0.189, CI excludes 0).

Add your own questions to `eval_questions.json`:

```json
{"question": "...", "expected_source": "my_notes.pdf", "expected_page": 7, "tag": "keyword"}
```

## Tests

```
python -m pytest tests/test_retrieval.py -v
```

Covers the ingest→retrieve round trip, idempotent re-ingest, file-delete
cleanup, RRF fusion maths, parent dedupe, the BM25 tokenizer and clean_text —
on a synthetic corpus, never touching your real `chroma_db/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No vector store found" / index empty | Click **Process documents** (or run `python ingest.py`) |
| "ANTHROPIC_API_KEY is not set" | Set the key, or switch to `LLM_PROVIDER=ollama` |
| "Ollama is not running" | Start Ollama (`ollama serve`) and pull the model (`ollama pull llama3.1`) |
| Answers ignore notes / cite nothing | Lower `MIN_SCORE` in `generate.py`, or upload more relevant PDFs |
| Retrieval finds wrong pages | Run `python evaluate.py`, compare modes, and try a different child chunk size |
| PDF text is garbled / fused words | Ingestion now uses PyMuPDF + `clean_text()` (fused-token % printed during ingest). If a PDF still glues words, check the diagnostic output |
