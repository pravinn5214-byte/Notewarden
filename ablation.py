"""Parent-child ablation: does returning the parent chunk (instead of the
matched child's own text) actually give the LLM better context?

For 20 questions, retrieve with hybrid_rerank twice — once with
return_parents=True (parent text) and once with return_parents=False (child
text) — and ask the configured LLM (LLM_PROVIDER, same as generate.py):
"does this context contain the answer to the question? yes/no".

Reports % contexts containing the answer for both variants. A high "yes" rate
for parents at similar or better rates than children justifies the parent
lookup; if parents underperform, that's a signal the parent window is too
coarse.

Usage:
    python ablation.py
    (called automatically by `python evaluate.py --ablation`)
"""
import os
import random

from dotenv import load_dotenv
load_dotenv()

from retrieve import retrieve

ABLATION_QUESTIONS = 20
SEED = 42


def _ask_llm(context: str, question: str) -> bool:
    """Ask the configured LLM whether the context answers the question."""
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    prompt = (
        "You are judging retrieval quality. Answer ONLY 'yes' or 'no'.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context[:3000]}\n\n"
        "Does this context contain the information needed to answer the question?"
    )
    if provider == "ollama":
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                  "prompt": prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=120,
        )
        answer = resp.json().get("response", "").strip().lower()
    elif provider == "claude":
        import requests
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 8,
                  "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        answer = "".join(b.get("text", "") for b in resp.json().get("content", [])).strip().lower()
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'.")
    return answer.startswith("yes")


def run(questions: list[dict]) -> str:
    """Run the ablation, return a markdown section."""
    rng = random.Random(SEED)
    subset = rng.sample(questions, min(ABLATION_QUESTIONS, len(questions)))

    parent_yes = child_yes = 0
    errors = 0
    for i, q in enumerate(subset):
        try:
            common = dict(k=3, mode="hybrid_rerank")
            hits_p = retrieve(q["question"], return_parents=True, **common)
            hits_c = retrieve(q["question"], return_parents=False, **common)
            ctx_p = "\n\n".join(h["text"] for h in hits_p)
            ctx_c = "\n\n".join(h["text"] for h in hits_c)
            parent_yes += _ask_llm(ctx_p, q["question"])
            child_yes += _ask_llm(ctx_c, q["question"])
            print(f"  [{i+1}/{len(subset)}] parent={'yes' if parent_yes and False else ''}", end="\r")
        except Exception as exc:
            errors += 1
            print(f"\n  ablation error on q{i+1}: {exc}")

    n = len(subset) - errors
    if n <= 0:
        return "## Parent-child ablation\n\nFailed — all questions errored (LLM unavailable?)."

    lines = [
        "## Parent-child ablation (hybrid_rerank, k=3, LLM-as-judge)",
        "",
        f"Judge: `{os.environ.get('LLM_PROVIDER', 'claude')}` · "
        f"{n} questions · question: 'does this context contain the answer? yes/no'",
        "",
        "| context | % contain answer |",
        "|---|---|",
        f"| parent text (~500 words) | {parent_yes / n:.0%} ({parent_yes}/{n}) |",
        f"| child text (~120 words) | {child_yes / n:.0%} ({child_yes}/{n}) |",
        "",
    ]
    if parent_yes >= child_yes:
        lines.append("Parents give the LLM at least as often answerable context — "
                     "the parent lookup is justified.")
    else:
        lines.append("Children outperform parents — consider a smaller PARENT_SIZE.")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from pathlib import Path
    qs = json.load(open(Path(__file__).parent / "eval_questions.json", encoding="utf-8"))
    print(run(qs))