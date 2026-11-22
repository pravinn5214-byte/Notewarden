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
