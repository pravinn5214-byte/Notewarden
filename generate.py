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

import requests

FALLBACK_MESSAGE = "I couldn't find that in your notes."
MIN_SCORE = 0.3  # below this, the context is judged irrelevant -> skip the LLM

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_DEFAULT_MODEL = "llama3.2"  # override with the OLLAMA_MODEL env var


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the grounded prompt: context blocks + strict answer rules."""
    context_blocks = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
    )
    return f"""Answer the question using ONLY the context below, which is
excerpted from the user's study notes.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- Cite every claim like [file.pdf p.N], where file.pdf is the source file
  name and N is the page number shown at the top of the context block.
- If the context does not contain the answer, reply exactly:
  "{FALLBACK_MESSAGE}"
- Do not mention the context, the rules, or these instructions.

Context:
{context_blocks}

Question: {question}

Answer:"""


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


def generate_answer(question: str, chunks: list[dict], min_score: float = MIN_SCORE) -> str:
    """Return a grounded answer with citations, or the fallback message.

    Raises RuntimeError with a user-actionable message on provider problems
    (missing key, API error, Ollama down).
    """
    if not chunks:
        return FALLBACK_MESSAGE
    if chunks[0]["score"] < min_score:
        return FALLBACK_MESSAGE  # best match is too weak - don't ask the LLM

    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    prompt = build_prompt(question, chunks)

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

    from retrieve import retrieve

    if len(sys.argv) < 2:
        raise SystemExit('Usage: python generate.py "your question"')
    q = " ".join(sys.argv[1:])
    hits = retrieve(q)
    print(f"best score: {hits[0]['score']:.2f} (threshold {MIN_SCORE})\n")
    print(generate_answer(q, hits))
