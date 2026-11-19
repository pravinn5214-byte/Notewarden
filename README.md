# RAG Study Assistant

Chat with your own PDF notes. Ingest PDFs, retrieve the relevant passages with
vector search, and get answers grounded in those notes — with page citations.

## How it works

```
docs/*.pdf -> ingest.py -> ChromaDB (persistent, cosine)
                              |
question -------------------> retrieve.py (top-k chunks + scores)
                              |
                    generate.py (Claude or Ollama)
                              |
                    answer with [file p.N] citations
```

- **Chunking**: each PDF page is split into overlapping word windows
  (default 300 words, 50 overlap). Every chunk carries `source` (file name)
  and `page` metadata.
- **Idempotent ingestion**: chunk IDs are content-derived, so rerunning
  ingestion never duplicates chunks — it only embeds new/changed chunks and
  removes stale ones.
- **Grounding**: the LLM is instructed to answer only from the provided
  context and to say "I couldn't find that in your notes" otherwise. If the
  best retrieval score is below `MIN_SCORE` (default 0.3), the LLM is skipped
  entirely.

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
expandable **Sources** panel showing file, page and similarity score.

### Command-line usage

```
python ingest.py                                 # index docs/ without the UI
python retrieve.py "what is normalization?"      # inspect raw retrieval
python generate.py "what is normalization?"      # full RAG answer in the terminal
```

## Evaluation

`evaluate.py` measures retrieval quality across chunk sizes 300/500/800:

```
python evaluate.py              # or: python evaluate.py 300 500
```

It rebuilds the index in `eval_chroma_db/` (your app index is untouched) for
each size, runs every question from `eval_questions.json`, and prints:

- **hit rate@k** — fraction of questions where the expected (file, page)
  appears in the top-k chunks
- **MRR** — mean of 1/rank of the first matching chunk (0 if absent)

Add your own questions to `eval_questions.json`:

```json
{"question": "...", "expected_source": "my_notes.pdf", "expected_page": 7}
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "No vector store found" / index empty | Click **Process documents** (or run `python ingest.py`) |
| "ANTHROPIC_API_KEY is not set" | Set the key, or switch to `LLM_PROVIDER=ollama` |
| "Ollama is not running" | Start Ollama (`ollama serve`) and pull the model (`ollama pull llama3.1`) |
| Answers ignore notes / cite nothing | Lower `MIN_SCORE` in `generate.py`, or upload more relevant PDFs |
| Retrieval finds wrong pages | Run `python evaluate.py` and try a different chunk size |

## Nice-to-haves (not yet implemented)

- Hybrid search: BM25 keyword search fused with vector search
- Re-ranking of retrieved chunks
