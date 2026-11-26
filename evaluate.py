"""Retrieval evaluation: hit rate@k and MRR across retrieval modes with
per-tag breakdowns and bootstrap confidence intervals.

Reads eval_questions.json, where each item has:
    question        - the query
    expected_source - file name the answer should come from
    expected_page   - page number the answer should come from
    tag             - "keyword" (exact terms) or "semantic" (paraphrase)

For each child chunk size it builds an index in eval_chroma_db_<size>/
(your app index is untouched), runs every question in every mode at k=3 and
k=5, then reports:
  - overall hit rate@k / MRR per mode, with bootstrap 95% CIs (1000 resamples)
  - the same split by tag, so keyword vs semantic strengths are visible
  - a parent-child ablation (see ablation.py) when run with --ablation

CIs: differences are only meaningful when their intervals don't overlap;
otherwise the result is marked "not significant".

Usage:
    python evaluate.py                # default child size
    python evaluate.py 120 300        # multiple sizes
    python evaluate.py --ablation     # add the parent-child ablation

Results are saved to eval_results.md.
"""
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from ingest import CHILD_SIZE, chunking_mode, ingest_pdfs
from retrieve import MODES, retrieve, retrieve_multi
from generate import FALLBACK_MESSAGE

EVAL_FILE = Path("eval_questions.json")
RESULTS_FILE = Path("eval_results.md")
# Separate store so the app's index is untouched. EVAL_DB_DIR switches the
# store (e.g. EVAL_DB_DIR=eval_chroma_db_heading) so two chunking modes can be
# measured without either one evicting the other's index.
DB_DIR = os.environ.get("EVAL_DB_DIR", "eval_chroma_db")
KS = [3, 5]
OVERLAP = 20  # child overlap
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
TAGS = ("keyword", "semantic")


def load_questions(path: Path = EVAL_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hit_and_rank(hits: list[dict], source: str, page: int) -> int | None:
    """Rank (1-based) of the first chunk matching (source, page), or None."""
    for rank, hit in enumerate(hits, start=1):
        if hit["source"] == source and hit["page"] == page:
            return rank
    return None


def bootstrap_ci(values: list[float], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% CI of the mean via bootstrap resampling."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n) - 1]
    return lo, hi


def fmt_metric(values: list[float]) -> str:
    """'0.667 [0.52-0.79]' — mean with 95% bootstrap CI."""
    if not values:
        return "n/a"
    m = sum(values) / len(values)
    lo, hi = bootstrap_ci(values)
    return f"{m:.3f} [{lo:.2f}-{hi:.2f}]"


def evaluate_mode(questions: list[dict], mode: str, k: int, db_dir: str) -> dict[str, list[float]]:
    """Per-question 1/rank values, overall and per tag. Returns a dict with
    keys 'overall', 'keyword', 'semantic' mapping to lists of reciprocal
    ranks (0.0 = miss)."""
    out: dict[str, list[float]] = {t: [] for t in TAGS}
    out["overall"] = []
    for q in questions:
        results = retrieve(q["question"], k=k, db_dir=db_dir, mode=mode)
        rank = hit_and_rank(results, q["expected_source"], q["expected_page"])
        rr = 1 / rank if rank is not None else 0.0
        out["overall"].append(rr)
        out[q["tag"]].append(rr)
    return out


MULTITOPIC_FILE = Path("eval_multitopic.json")

# --paired: which stores to diff and where the table lands. Defaults match the
# eval runs that produced eval_results_heading.md (window vs heading, size 120).
PAIRED_BASE_DB = "eval_chroma_db_120"
PAIRED_TREATED_DB = "eval_chroma_db_heading_120"
PAIRED_TREATED_LABEL = "heading"
PAIRED_RESULTS_FILE = Path("eval_results_heading.md")


def paired_bootstrap_ci(diffs: list[int], n: int = BOOTSTRAP_N,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% CI of the mean of per-question differences, via the same bootstrap
    percentile convention as bootstrap_ci (same n/seed → comparable widths)."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(diffs)
    means = []
    for _ in range(n):
        sample = [rng.choice(diffs) for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def run_paired(base_db: str = PAIRED_BASE_DB,
               treated_db: str = PAIRED_TREATED_DB,
               treated_label: str = PAIRED_TREATED_LABEL) -> list[str]:
    """--paired: per-question hit@k difference (treated − window) with paired
    bootstrap CIs, per mode/k. Both stores must already be ingested (run
    evaluate.py with EVAL_DB_DIR set to each) and contain the same files."""
    questions = load_questions()
    lines = [
        f"## Paired bootstrap: {treated_label} − window (per-question hit@k)",
        "",
        f"Same {len(questions)} questions run against both stores "
        f"({base_db} vs {treated_db}), retrieval only. Each row bootstraps the "
        f"mean per-question difference ({treated_label} − window) with "
        f"{BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED} — paired, so question "
        'difficulty cancels out. "Hits" = questions with the expected '
        "(source, page) in the top k.",
        "",
        "| mode | k | window hits | " + treated_label + " hits | diff | 95% CI | excludes 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        for k in KS:
            w = [1.0 if r > 0 else 0.0
                 for r in evaluate_mode(questions, mode, k, base_db)["overall"]]
            t = [1.0 if r > 0 else 0.0
                 for r in evaluate_mode(questions, mode, k, treated_db)["overall"]]
            diffs = [b - a for a, b in zip(w, t)]
            lo, hi = paired_bootstrap_ci(diffs)
            row = (f"| {mode} | {k} | {sum(w):.0f}/{len(w)} ({sum(w)/len(w):.3f}) "
                   f"| {sum(t):.0f}/{len(t)} ({sum(t)/len(t):.3f}) "
                   f"| {sum(diffs)/len(diffs):+.3f} | [{lo:+.3f}, {hi:+.3f}] "
                   f"| {'yes' if lo > 0 or hi < 0 else 'no'} |")
            lines.append(row)
            print(f"  {mode:>14} k={k}: {row}")
    return lines


def _save_paired(lines: list[str], path: Path = PAIRED_RESULTS_FILE) -> None:
    """Replace everything from the '## Paired bootstrap' heading on (or append
    when the file has no such section yet)."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## Paired bootstrap"
    if marker in text:
        text = text[:text.index(marker)] + "\n".join(lines) + "\n"
    else:
        text = text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n" if text \
            else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def both_covered(hits: list[dict], topics: list[dict], k: int) -> bool:
    """True when every topic's keyword appears in a hit from its expected
    source within the top-k."""
    top = hits[:k]
    for topic in topics:
        ok = any(
            h["source"] == topic.get("expected_source") and
            topic.get("keyword", "").lower() in h["text"].lower()
            for h in top)
        if not ok:
            return False
    return True


def evaluate_multitopic(mode: str, k: int, db_dir: str, decompose_on: bool) -> float:
    """Rate of two-topic questions where BOTH topics are covered in the top-k."""
    questions = load_questions(MULTITOPIC_FILE)
    covered = 0
    for q in questions:
        if decompose_on:
            from decompose import decompose
            dec = decompose(q["question"])
            subs = dec.subqueries
            if len(subs) > 1:
                hits, _ = retrieve_multi(subs, mode=mode, top_k=k, db_dir=db_dir)
            else:
                hits = retrieve(q["question"], k=k, db_dir=db_dir, mode=mode)
        else:
            hits = retrieve(q["question"], k=k, db_dir=db_dir, mode=mode)
        if both_covered(hits, q["topics"], k):
            covered += 1
    return covered / len(questions)


def run_multitopic(sizes: list[int], write: bool) -> None:
    """--multitopic: 'both topics covered in top-k' rate, decompose off vs on."""
    from decompose import clear_cache
    rows: list[str] = ["| mode | k | decompose off | decompose on |", "|---|---|---|---|"]
    for child_size in sizes:
        db_dir = f"{DB_DIR}_{child_size}"
        print(f"Ingesting at child size {child_size} into {db_dir}/ ...")
        ingest_pdfs(child_size=child_size, child_overlap=OVERLAP, db_dir=db_dir)
        for mode in MODES:
            for k in KS:
                clear_cache()
                off = evaluate_multitopic(mode, k, db_dir, decompose_on=False)
                clear_cache()
                on = evaluate_multitopic(mode, k, db_dir, decompose_on=True)
                print(f"  {mode:>14} k={k}: off={off:.0%} on={on:.0%}")
                rows.append(f"| {mode} | {k} | {off:.0%} | {on:.0%} |")
    report = "\n".join(["## Multi-topic coverage — decompose off vs on", "", *rows])
    print(f"\n{report}")
    if write:
        out = Path("eval_results_multitopic.md")
        out.write_text(report + "\n", encoding="utf-8")
        print(f"Saved to {out}")


AGENT_FILE = Path("eval_agent.json")


def run_agent_eval(db_dir: str, llm=None) -> dict:
    """--agent: recovered-after-retry rate, clarify precision, unnecessary-
    clarify rate, and answer-rate on unanswerable questions."""
    from agent import answer_with_agent

    questions = load_questions(AGENT_FILE)
    stats = {
        "n": len(questions),
        "recovered_after_retry": 0, "retry_used": 0,
        "clarified": 0, "should_clarify": 0,
        "clarified_correctly": 0, "unnecessary_clarify": 0,
        "unanswerable_answered": 0, "unanswerable_n": 0,
        "answered_strong": 0,
    }
    for q in questions:
        result = answer_with_agent(q["question"], history=[], mode="hybrid_rerank",
                                   top_k=4, db_dir=db_dir, llm=llm)
        qtype = q.get("type", "answerable")
        retry = bool(result.trace.get("retry"))
        clarified = result.action == "clarify"
        if retry:
            stats["retry_used"] += 1
        if clarified:
            stats["clarified"] += 1
        if qtype == "vague":
            stats["should_clarify"] += 1
            if clarified:
                stats["clarified_correctly"] += 1
        elif clarified:
            stats["unnecessary_clarify"] += 1
        if qtype == "unanswerable":
            stats["unanswerable_n"] += 1
            if not clarified and result.answer_or_question != FALLBACK_MESSAGE:
                stats["unanswerable_answered"] += 1
    return stats


def print_agent_report(stats: dict) -> None:
    n = max(1, stats["n"])
    un_n = max(1, stats["unanswerable_n"])
    lines = [
        "## Agent evaluation",
        "",
        f"- Questions: {stats['n']}",
        f"- Retries used: {stats['retry_used']}",
        f"- Recovered after retry: not directly observable offline; retry rate {stats['retry_used'] / n:.0%}",
        f"- Clarify precision: {stats['clarified_correctly']}/{stats['should_clarify']} vague questions clarified"
        f" ({stats['clarified_correctly'] / max(1, stats['should_clarify']):.0%})",
        f"- Unnecessary clarifies: {stats['unnecessary_clarify']} of {stats['n']} ({stats['unnecessary_clarify'] / n:.0%})",
        f"- Unanswerable questions answered anyway: {stats['unanswerable_answered']}/{stats['unanswerable_n']}"
        f" ({stats['unanswerable_answered'] / un_n:.0%} — should be 0%)",
    ]
    report = "\n".join(lines)
    print(report)
    return report


def run_agent(sizes: list[int], write: bool, llm=None) -> None:
    for child_size in sizes:
        db_dir = f"{DB_DIR}_{child_size}"
        print(f"Ingesting at child size {child_size} into {db_dir}/ ...")
        ingest_pdfs(child_size=child_size, child_overlap=OVERLAP, db_dir=db_dir)
        stats = run_agent_eval(db_dir, llm=llm)
        report = print_agent_report(stats)
        if write:
            out = Path("eval_results_agent.md")
            out.write_text(report + "\n", encoding="utf-8")
            print(f"Saved to {out}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ablation = "--ablation" in sys.argv
    if "--agent" in sys.argv:
        sizes = [int(a) for a in args] or [CHILD_SIZE]
        run_agent(sizes, write="--write" in sys.argv)
        return
    if "--paired" in sys.argv:
        # Window vs treated (default heading) per-question hit difference with
        # paired bootstrap CIs. Stores must already be ingested.
        _save_paired(run_paired())
        return
    if "--multitopic" in sys.argv:
        sizes = [int(a) for a in args] or [CHILD_SIZE]
        run_multitopic(sizes, write="--write" in sys.argv)
        return
    sizes = [int(a) for a in args] or [CHILD_SIZE]
    questions = load_questions()
    print(f"{len(questions)} questions ({sum(1 for q in questions if q['tag']=='keyword')} keyword / "
          f"{sum(1 for q in questions if q['tag']=='semantic')} semantic), modes: {', '.join(MODES)}, k: {KS}\n")

    tables: list[str] = []
    for child_size in sizes:
        db_dir = f"{DB_DIR}_{child_size}"
        print(f"Ingesting at child size {child_size} into {db_dir}/ ...")
        ingest_pdfs(child_size=child_size, child_overlap=OVERLAP, db_dir=db_dir)

        rows: list[tuple[str, int, dict[str, list[float]]]] = []
        for mode in MODES:
            for k in KS:
                rr = evaluate_mode(questions, mode, k, db_dir)
                rows.append((mode, k, rr))
                print(f"  {mode:>14} k={k}: overall={fmt_metric(rr['overall'])}")

        md = [
            f"## Mode comparison — child size {child_size}",
            "",
            "Values: mean [bootstrap 95% CI]. CI overlap ⇒ not significant.",
            "",
            "| mode | k | overall hit/MRR | keyword | semantic |",
            "|---|---|---|---|---|",
        ]
        for mode, k, rr in rows:
            md.append(f"| {mode} | {k} | {fmt_metric(rr['overall'])} | "
                      f"{fmt_metric(rr['keyword'])} | {fmt_metric(rr['semantic'])} |")
        tables.append("\n".join(md))

    if ablation:
        import ablation
        tables.append(ablation.run(questions))

    header = [
        "# Retrieval evaluation results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{len(questions)} questions · extractor: PyMuPDF + clean_text · "
        f"chunking: {chunking_mode()} · "
        f"child overlap {OVERLAP} · RRF k=60 · reranker: ms-marco-MiniLM-L-6-v2 · "
        f"{BOOTSTRAP_N} bootstrap resamples (seed {BOOTSTRAP_SEED})",
        "",
    ]
    RESULTS_FILE.write_text("\n".join(header) + "\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()