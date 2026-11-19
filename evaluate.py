"""Retrieval evaluation: hit rate@k and MRR across chunk sizes.

Reads eval_questions.json, where each item has:
    question        - the query
    expected_source - file name the answer should come from
    expected_page   - page number the answer should come from

For each chunk size in CHUNK_SIZES it rebuilds the index, retrieves the
top-k chunks for every question, and checks whether any chunk matches the
expected (source, page). It then prints a comparison table.

Usage:
    python evaluate.py            # default: sizes 300, 500, 800
    python evaluate.py 300 500    # custom sizes
"""
import json
from pathlib import Path

from retrieve import retrieve

from ingest import ingest_pdfs

EVAL_FILE = Path("eval_questions.json")
DB_DIR = "eval_chroma_db"  # separate store so the app's index is untouched
CHUNK_SIZES = [300, 500, 800]
OVERLAP_RATIO = 50 / 300  # keep the same overlap ratio as the default settings
TOP_K = 10  # retrieve deep enough for stable MRR; report @k and MRR from this


def load_questions(path: Path = EVAL_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hit_and_rank(hits: list[dict], source: str, page: int) -> int | None:
    """Rank (1-based) of the first chunk matching (source, page), or None."""
    for rank, hit in enumerate(hits, start=1):
        if hit["source"] == source and hit["page"] == page:
            return rank
    return None


def evaluate_size(questions: list[dict], chunk_size: int, k: int = TOP_K) -> dict:
    """Ingest at one chunk size and score all questions. Returns the metrics."""
    overlap = max(1, round(chunk_size * OVERLAP_RATIO))
    ingest_pdfs(db_dir=DB_DIR, chunk_size=chunk_size, overlap=overlap)

    hits_total = 0
    reciprocal_ranks: list[float] = []
    per_question: list[tuple[str, int | None]] = []
    for q in questions:
        results = retrieve(q["question"], k=k, db_dir=DB_DIR)
        rank = hit_and_rank(results, q["expected_source"], q["expected_page"])
        if rank is not None:
            hits_total += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
        per_question.append((q["question"], rank))

    n = len(questions)
    return {
        "chunk_size": chunk_size,
        "hit_rate@k": hits_total / n,
        "mrr": sum(reciprocal_ranks) / n,
        "per_question": per_question,
        "k": k,
    }


def main() -> None:
    sizes = [int(a) for a in __import__("sys").argv[1:]] or CHUNK_SIZES
    questions = load_questions()
    print(f"{len(questions)} questions, top-{TOP_K} retrieval\n")

    results = [evaluate_size(questions, size) for size in sizes]

    # Comparison table
    print(f"{'chunk size':>10} | {'hit rate@k':>10} | {'MRR':>6}")
    print("-" * 34)
    for r in results:
        print(f"{r['chunk_size']:>10} | {r['hit_rate@k']:>10.2%} | {r['mrr']:>6.3f}")

    # Per-question detail for the best size (by MRR)
    best = max(results, key=lambda r: r["mrr"])
    print(f"\nPer-question ranks at chunk size {best['chunk_size']} "
          f"(None = not found in top-{best['k']}):")
    for question, rank in best["per_question"]:
        print(f"  rank {str(rank):>4}  {question}")


if __name__ == "__main__":
    main()
