"""Step 4 of the pipeline: build a grounded prompt and call the LLM.

Two providers, selected by the LLM_PROVIDER env var:
    - "claude"  -> Anthropic API (key in ANTHROPIC_API_KEY)
    - "ollama"  -> local Ollama server (model via OLLAMA_MODEL, default llama3.1)

The prompt instructs the model to answer ONLY from the provided context and to
cite sources as [file p.N]. If the best retrieval score is below
MIN_SCORE (default 0.3), the LLM is skipped entirely and the fallback
message is returned.
"""
import os

from dotenv import load_dotenv
load_dotenv()

import requests

FALLBACK_MESSAGE = "I couldn't find that in your notes."
MIN_SCORE = 0.3  # below this, the context is judged irrelevant -> skip the LLM

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "llama3.2"  # override with the OLLAMA_MODEL env var


def build_prompt(question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
    """Build the grounded prompt: context blocks + strict answer rules.

    history: optional recent turns ([{"role": "user"|"assistant", "text": ...}],
    oldest first) so the model can resolve follow-up references.
    """
    history_block = ""
    if history:
        turns = "\n".join(f"{h['role']}: {h['text']}" for h in history)
        history_block = f"""Recent conversation (for resolving references in the question):
{turns}

"""
    context_blocks = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
    )
    return f"""Answer the question using ONLY the context below, which is
excerpted from the user's study notes.

{history_block}Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- If the question asks about several distinct things (e.g. two topics, two
  files), answer EACH part separately, in order, under its own bold heading
  like **Part 1 — <topic>**, using the context that covers it. Do not drop a
  part just because other parts dominate the context. If a part has no
  supporting context, say so under its heading.
- Cite every claim like [file.pdf p.N], where file.pdf is the source file
  name and N is the page number shown at the top of the context block.
- If the context does not contain the answer, reply exactly:
  "{FALLBACK_MESSAGE}"
- Do not mention the context, the rules, or these instructions.

Context:
{context_blocks}

Question: {question}

Answer:"""


def complete(prompt: str, max_tokens: int = 200, json_mode: bool = False) -> str:
    """One raw LLM completion using the configured provider (claude/ollama).

    json_mode: request structured JSON output. Anthropic always honours the
    prompt-level "Return JSON only"; Ollama needs its `format: "json"` flag,
    which forces syntactically valid JSON (content still needs validation).
    Shared by answer generation and utility calls (e.g. query decomposition).
    Raises RuntimeError on provider problems; callers handle fallbacks.
    """
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set. "
                "Set it, or set LLM_PROVIDER=ollama to use a local model."
            )
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0,  # utility calls must be reproducible
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
        return "".join(parts).strip()
    if provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    **({"format": "json"} if json_mode else {}),
                    "options": {"temperature": 0, "num_predict": max_tokens},
                },
                timeout=120,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not running. Start it with 'ollama serve' (or open the "
                "Ollama app) and try again."
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json().get("response", "").strip()
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use 'claude' or 'ollama'.")


def _call_claude(prompt: str, api_key: str) -> str:
    """Call the Anthropic Messages API and return the reply text."""
    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1024,
            "temperature": 0,  # grounded answers must be reproducible
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    return "".join(parts).strip()


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama server and return the reply text."""
    model = os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # Grounded answers must be reproducible: sampling randomness
                # makes the model sometimes answer and sometimes give up.
                "options": {"temperature": 0},
            },
            timeout=120,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Ollama is not running. Start it with 'ollama serve' (or open the "
            "Ollama app) and try again."
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("response", "").strip()


def _unsupported_line(subqueries: list[str], supported: set[int]) -> str:
    """'No relevant notes found for: <subquery>' lines for failed sub-queries."""
    lines = [f"No relevant notes found for: {q}"
             for i, q in enumerate(subqueries) if i not in supported]
    return "\n".join(lines)


def build_parts_prompt(question: str, chunks: list[dict], subqueries: list[str],
                       answer_as_parts: bool, supported: set[int]) -> str:
    """Prompt for decomposed questions.

    answer_as_parts=True: context grouped per sub-query, per-part headings.
    answer_as_parts=False (comparison): one merged context block, one
    integrated answer — same shape as the single-question prompt.
    """
    unsupported = _unsupported_line(subqueries, supported)
    if not answer_as_parts:
        unsupported_block = f"\n{unsupported}\n" if unsupported else ""
        context_blocks = "\n\n".join(
            f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
        )
        return f"""Answer the question using ONLY the context below, which is
excerpted from the user's study notes.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- The question compares or relates several things. Give ONE integrated answer
  that addresses each part and their relationship, not separate sections.
- Cite every claim like [file.pdf p.N], where file.pdf is the source file
  name and N is the page number shown at the top of the context block.
- If the context does not contain the answer, reply exactly:
  "{FALLBACK_MESSAGE}"
- Do not mention the context, the rules, or these instructions.
{unsupported_block}
Context:
{context_blocks}

Question: {question}

Answer:"""

    # Per-part context groups.
    by_sub: dict[int, list[dict]] = {}
    for c in chunks:
        by_sub.setdefault(c.get("subquery_index", 0), []).append(c)
    groups = []
    for i, sub in enumerate(subqueries):
        blocks = by_sub.get(i, [])
        if blocks:
            body = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in blocks)
        else:
            body = "(no notes retrieved for this part)"
        groups.append(f"Notes for part {i + 1} ({sub}):\n{body}")
    context_blocks = "\n\n".join(groups)
    unsupported_block = f"\n{unsupported}\n" if unsupported else ""
    return f"""Answer the question using ONLY the context below, which is
excerpted from the user's study notes.

The question has several independent parts. Answer each under a heading
**Part N — <short topic>**. Use only the notes provided for that part (notes
labelled for another part may be used if clearly relevant). If a part has no
supporting notes, write that the selected notes don't cover it instead of
guessing. Cite as [file p.N].

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- Cite every claim like [file.pdf p.N], where file.pdf is the source file
  name and N is the page number shown at the top of the context block.
- If the context does not contain the answer, reply exactly:
  "{FALLBACK_MESSAGE}"
- Do not mention the context, the rules, or these instructions.
{unsupported_block}
Context:
{context_blocks}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[dict], min_score: float = MIN_SCORE,
                    decomposition=None, history: list[dict] | None = None) -> str:
    """Return a grounded answer with citations, or the fallback message.

    The gate uses "dense_score" (cosine similarity, present in every result
    from retrieve()) rather than "score": MIN_SCORE is calibrated for cosine
    similarity (0..1), while hybrid's RRF scores (~0.03 max) and the
    cross-encoder's scores (often negative) live on different scales and
    would trip a cosine threshold incorrectly. It looks at the BEST dense
    score among the chunks (see the gate below), which is exactly what
    query_log.gated() records as `gated_out`.

    Raises RuntimeError with a user-actionable message on provider problems
    (missing key, API error, Ollama down).
    """
    if not chunks:
        return FALLBACK_MESSAGE
    # Gate on the BEST dense cosine across the returned chunks, not on
    # chunks[0]: hits are ordered by the mode's ranking score (RRF or the
    # cross-encoder), so the top-ranked hit is often not the best dense match.
    # This is the same rule retrieve_multi applies per sub-query and the same
    # number the query log records as top_dense_score, so `gated_out` in the
    # log always matches what actually happened here.
    dense_score = max(c.get("dense_score", c.get("score", 0.0)) for c in chunks)
    if dense_score < min_score:
        return FALLBACK_MESSAGE  # best match is too weak - don't ask the LLM

    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if decomposition is not None and len(decomposition.subqueries) > 1:
        supported = {h["subquery_index"] for h in chunks if "subquery_index" in h}
        prompt = build_parts_prompt(
            question, chunks, decomposition.subqueries,
            decomposition.answer_as_parts, supported)
    else:
        prompt = build_prompt(question, chunks, history=history)

    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set. "
                "Set it, or set LLM_PROVIDER=ollama to use a local model."
            )
        return _call_claude(prompt, api_key)
    if provider == "ollama":
        return _call_ollama(prompt)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use 'claude' or 'ollama'.")


if __name__ == "__main__":
    import sys
    import time
    import os as _os

    from retrieve import retrieve
    from query_log import log_query, results_payload, top_dense, gated

    if len(sys.argv) < 2:
        raise SystemExit('Usage: python generate.py "your question"')
    q = " ".join(sys.argv[1:])
    _t0 = time.monotonic()
    _error = None
    hits, answer = [], None
    try:
        hits = retrieve(q)
        print(f"best score: {hits[0]['score']:.2f} (threshold {MIN_SCORE})\n")
        answer = generate_answer(q, hits)
        print(answer)
    except Exception as exc:
        _error = str(exc)
        raise
    finally:
        log_query(
            original_query=q,
            retrieval_mode=_os.environ.get("RETRIEVAL_MODE", "hybrid_rerank"),
            results=results_payload(hits),
            top_dense_score=top_dense(hits),
            gated_out=gated(hits, MIN_SCORE),
            llm_provider=_os.environ.get("LLM_PROVIDER", "claude"),
            latency_ms=(time.monotonic() - _t0) * 1000.0,
            error=_error,
            answer=answer,
        )
