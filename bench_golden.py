"""Golden-master benchmark for the RAG Study Assistant retrieval pipeline.

Records, for every question × mode × k combination, the ordered chunk IDs
and scores, plus p50 / p95 retrieval latency (warm and cold).  The output is
saved to golden/baseline.json (or a path given via --output).

Usage
-----
  # Record the baseline (run once against the real index):
  python bench_golden.py

  # Check that the current code produces identical results:
  python bench_golden.py --check [--tolerance 1e-6]

The --check mode exits with code 0 on success, 1 on any ID/score difference,
printing a human-readable diff and the latency delta per combination.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# --- Constants ---------------------------------------------------------------
EVAL_JSON = Path("eval_questions.json")
GOLDEN_DIR = Path("golden")
DEFAULT_OUTPUT = GOLDEN_DIR / "baseline.json"
MODES = ("dense", "hybrid", "hybrid_rerank")
KS = (3, 5)
WARMUP_REPS = 2    # runs before timing to warm caches
TIMING_REPS = 5    # runs for latency measurement
DEFAULT_TOL = 1e-6

# Fields kept for the golden record (not "text" / "parent_text" — those are
# the LLM context and not relevant for retrieval identity checks).
RECORD_FIELDS = ("id", "source", "page", "dense_score", "score")


# --- Helpers -----------------------------------------------------------------

def _load_questions() -> list[dict]:
    if not EVAL_JSON.exists():
        sys.exit(f"ERROR: {EVAL_JSON} not found")
    qs = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    if not qs:
        sys.exit("ERROR: eval_questions.json is empty")
    return qs


def _slim(hit: dict) -> dict:
    """Keep only the fields that define retrieval identity."""
    return {f: hit[f] for f in RECORD_FIELDS if f in hit}


def _retrieve_one(question: str, mode: str, k: int) -> list[dict]:
    from retrieve import retrieve
    return retrieve(question, k=k, mode=mode)


def _measure_latency(question: str, mode: str, k: int) -> dict:
    """
    Warm-start latency: measure after WARMUP_REPS calls (models + Chroma loaded).
    Cold-start latency: reset all module-level caches then measure one call.
    Returns {"warm_p50_ms": float, "warm_p95_ms": float,
             "cold_ms": float}.
    """
    from retrieve import _get_embedder, _get_reranker, invalidate_bm25_cache
    from store import reset_client

    # Warm: ensure models are loaded, then time TIMING_REPS calls.
    _get_embedder()
    if mode == "hybrid_rerank":
        _get_reranker()
    for _ in range(WARMUP_REPS):
        _retrieve_one(question, mode, k)

    warm_times: list[float] = []
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        _retrieve_one(question, mode, k)
        warm_times.append((time.perf_counter() - t0) * 1000)

    warm_p50 = statistics.median(warm_times)
    warm_p95 = statistics.quantiles(warm_times, n=20)[18]  # ~95th pctile over 5 samples

    # Cold: drop ALL module-level caches (client, collections, BM25) then
    # time a single call.  We drop retrieve's globals via reset_client which
    # triggers _drop_cached_collections.
    invalidate_bm25_cache()
    reset_client()
    t0 = time.perf_counter()
    _retrieve_one(question, mode, k)
    cold_ms = (time.perf_counter() - t0) * 1000

    return {"warm_p50_ms": warm_p50, "warm_p95_ms": warm_p95, "cold_ms": cold_ms}


# --- Record mode -------------------------------------------------------------

def record(output: Path) -> None:
    questions = _load_questions()
    output.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "meta": {
            "questions": len(questions),
            "modes": list(MODES),
            "ks": list(KS),
            "tolerance": DEFAULT_TOL,
        },
        "entries": [],
    }

    total = len(questions) * len(MODES) * len(KS)
    done = 0
    for q in questions:
        question = q["question"]
        for mode in MODES:
            for k in KS:
                done += 1
                print(f"\r[{done}/{total}] {mode} k={k}  {question[:50]:<50}", end="", flush=True)
                hits = _retrieve_one(question, mode, k)
                entry = {
                    "question": question,
                    "mode": mode,
                    "k": k,
                    "hits": [_slim(h) for h in hits],
                }
                results["entries"].append(entry)

    print()
    # Latency: measure for the first question, all modes, k=3 only (representative).
    print("Measuring latency (warm + cold) …")
    latencies = {}
    q0 = questions[0]["question"]
    for mode in MODES:
        lat = _measure_latency(q0, mode, k=3)
        latencies[mode] = lat
        print(f"  {mode}: warm p50={lat['warm_p50_ms']:.1f}ms  "
              f"p95={lat['warm_p95_ms']:.1f}ms  cold={lat['cold_ms']:.1f}ms")
    results["latency"] = latencies

    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBaseline saved → {output}  ({done} entries)")


# --- Check mode --------------------------------------------------------------

def check(golden_path: Path, tol: float) -> int:
    if not golden_path.exists():
        print(f"ERROR: {golden_path} not found — run without --check first")
        return 1

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    lat_deltas: list[str] = []

    total = len(golden["entries"])
    done = 0
    for entry in golden["entries"]:
        question = entry["question"]
        mode = entry["mode"]
        k = entry["k"]
        done += 1
        print(f"\r[{done}/{total}] checking {mode} k={k}  {question[:45]:<45}", end="", flush=True)
        actual_hits = _retrieve_one(question, mode, k)
        actual = [_slim(h) for h in actual_hits]
        expected = entry["hits"]

        if len(actual) != len(expected):
            failures.append(
                f"{mode} k={k} | {question[:60]!r} | "
                f"length mismatch: got {len(actual)}, want {len(expected)}"
            )
            continue
        for rank, (a, e) in enumerate(zip(actual, expected)):
            if a.get("id") != e.get("id"):
                failures.append(
                    f"{mode} k={k} rank={rank+1} | {question[:60]!r} | "
                    f"ID mismatch: got {a.get('id')!r}, want {e.get('id')!r}"
                )
            else:
                for field in ("dense_score", "score"):
                    av = a.get(field, 0.0)
                    ev = e.get(field, 0.0)
                    if abs(av - ev) > tol:
                        failures.append(
                            f"{mode} k={k} rank={rank+1} | {question[:60]!r} | "
                            f"{field}: got {av:.8f}, want {ev:.8f} (Δ={abs(av-ev):.2e})"
                        )

    print()

    # Latency delta (informational, never a failure).
    if "latency" in golden:
        print("\nLatency delta (current vs baseline):")
        q0 = golden["entries"][0]["question"]
        for mode in MODES:
            lat = _measure_latency(q0, mode, k=3)
            base = golden["latency"].get(mode, {})
            def _d(key: str) -> str:
                b = base.get(key, 0)
                c = lat.get(key, 0)
                sign = "+" if c >= b else ""
                return f"{sign}{c - b:+.1f}ms (now {c:.1f}ms)"
            lat_deltas.append(
                f"  {mode}: warm_p50 {_d('warm_p50_ms')}  "
                f"warm_p95 {_d('warm_p95_ms')}  cold {_d('cold_ms')}"
            )
        for line in lat_deltas:
            print(line)

    if failures:
        print(f"\n❌  {len(failures)} FAILURE(S):")
        for f in failures[:40]:
            print(" ", f)
        if len(failures) > 40:
            print(f"  … and {len(failures) - 40} more")
        return 1

    print(f"\n✅  Golden master: all {total} entries match (tolerance {tol:.0e})")
    return 0


# --- Pytest suite timing (called from check mode summary) --------------------

def time_pytest() -> float:
    """Run the full pytest suite and return elapsed seconds."""
    import subprocess
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short"],
        capture_output=False,
    )
    elapsed = time.perf_counter() - t0
    if r.returncode not in (0, 5):  # 5 = no tests collected (shouldn't happen)
        print(f"WARNING: pytest exited with code {r.returncode}")
    return elapsed


def time_reingest() -> float:
    """Re-ingest docs/ and return elapsed seconds."""
    from ingest import ingest_pdfs
    from store import reset_client
    t0 = time.perf_counter()
    ingest_pdfs()
    reset_client()
    return time.perf_counter() - t0


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-master retrieval benchmark")
    parser.add_argument("--check", action="store_true",
                        help="Check current results against the saved baseline")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Path to baseline JSON (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOL,
                        help=f"Score tolerance for --check (default: {DEFAULT_TOL:.0e})")
    parser.add_argument("--time-pytest", action="store_true",
                        help="Time the full pytest suite and print the result")
    parser.add_argument("--time-reingest", action="store_true",
                        help="Time a re-ingest of docs/ and print the result")
    args = parser.parse_args()

    if args.time_pytest:
        t = time_pytest()
        print(f"pytest suite: {t:.1f}s")
        return

    if args.time_reingest:
        t = time_reingest()
        print(f"re-ingest: {t:.1f}s")
        return

    if args.check:
        sys.exit(check(args.output, args.tolerance))
    else:
        record(args.output)


if __name__ == "__main__":
    main()
