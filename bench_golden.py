"""Golden-master benchmark for the RAG Study Assistant retrieval pipeline.

Usage
-----
# Record baseline (run once against the real index):
    python bench_golden.py

# Check a later run against the recorded baseline:
    python bench_golden.py --check

# Override output path:
    python bench_golden.py [--check] --out golden/my_run.json

What is recorded
----------------
For every combination of (question × mode × k) the golden master stores:
  - ordered child chunk IDs
  - ordered dense_score values (to 1e-6 precision)
  - ordered final score values (to 1e-6 precision)

Latency is recorded per (mode, k):
  - warm: median of WARM_REPS calls on the SAME question (model already loaded)
  - cold: a fresh Python subprocess that answers one question (model load included)

Also recorded: pytest suite wall-clock time and re-ingest wall-clock time.

--check mode
------------
Fails (exit 1) if any ID order or score differs from the saved baseline.
Prints a delta table for latencies (new minus old, in ms).
Score tolerance: 1e-6 absolute (floating-point rounding only).

Requires
--------
  - A populated chroma_db/ index (run ingest.py first).
  - eval_questions.json (45 questions).
  - retrieve.py, store.py, ingest.py on PYTHONPATH (run from repo root).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUESTIONS_FILE = Path("eval_questions.json")
GOLDEN_DIR = Path("golden")
DEFAULT_OUT = GOLDEN_DIR / "baseline.json"
MODES = ("dense", "hybrid", "hybrid_rerank")
KS = (3, 5)
WARM_REPS = 5          # repetitions for warm-latency median
SCORE_TOL = 1e-6       # absolute tolerance for score comparison

# Which single question to use for cold-start latency probes (index 0)
COLD_QUESTION_IDX = 0


def _load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _retrieve_one(question: str, mode: str, k: int) -> list[dict]:
    """Return retrieve() results for one (question, mode, k) combo."""
    from retrieve import retrieve
    return retrieve(question, k=k, mode=mode)


def _run_all(questions: list[dict]) -> dict:
    """Run retrieval for all (q × mode × k) and record IDs + scores + latency."""
    from retrieve import retrieve

    # Warm up the models once before timing (they are module-level singletons)
    q0 = questions[COLD_QUESTION_IDX]["question"]
    for mode in MODES:
        for k in KS:
            retrieve(q0, k=k, mode=mode)

    results: dict = {"questions": {}, "latency_warm_ms": {}, "latency_cold_ms": {}}

    for qi, item in enumerate(questions):
        q = item["question"]
        qkey = str(qi)
        results["questions"][qkey] = {}
        for mode in MODES:
            for k in KS:
                key = f"{mode}_k{k}"
                # Record IDs and scores
                hits = retrieve(q, k=k, mode=mode)
                results["questions"][qkey][key] = {
                    "ids":         [h["id"] for h in hits],
                    "dense_score": [round(h.get("dense_score", 0.0), 8) for h in hits],
                    "score":       [round(h.get("score", 0.0), 8) for h in hits],
                }

    # Warm-latency: WARM_REPS calls per (mode, k) on question 0
    for mode in MODES:
        for k in KS:
            key = f"{mode}_k{k}"
            times = []
            for _ in range(WARM_REPS):
                t0 = time.perf_counter()
                retrieve(q0, k=k, mode=mode)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            mid = WARM_REPS // 2
            results["latency_warm_ms"][key] = round(times[mid], 2)

    # Cold-latency: subprocess per (mode, k) — includes model load.
    # Pass HF_HUB_OFFLINE so subprocesses don't retry HuggingFace lookups.
    _sub_env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for mode in MODES:
        for k in KS:
            key = f"{mode}_k{k}"
            script = (
                f"import time, sys\n"
                f"sys.path.insert(0, '.')\n"
                f"t0=time.perf_counter()\n"
                f"from retrieve import retrieve\n"
                f"retrieve({q0!r}, k={k}, mode={mode!r})\n"
                f"print(round((time.perf_counter()-t0)*1000,2))\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
                env=_sub_env,
            )
            if proc.returncode != 0:
                print(f"WARNING: cold-start subprocess failed for {key}:", proc.stderr[:200])
                results["latency_cold_ms"][key] = None
            else:
                results["latency_cold_ms"][key] = float(proc.stdout.strip())

    return results


def _run_pytest_timer() -> float:
    """Run the full pytest suite and return wall-clock seconds."""
    _env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        capture_output=True, env=_env,
    )
    return round(time.perf_counter() - t0, 2)


def _run_reingest_timer() -> float:
    """Re-ingest docs/ and return wall-clock seconds (OCR cache reused)."""
    _env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, "ingest.py"],
        capture_output=True, env=_env,
    )
    return round(time.perf_counter() - t0, 2)


def _record(out_path: Path) -> None:
    """Run the full benchmark and write the golden file."""
    questions = _load_questions()
    print(f"Recording golden master for {len(questions)} questions × "
          f"{len(MODES)} modes × {len(KS)} k-values …")
    t0 = time.perf_counter()
    data = _run_all(questions)
    retrieval_total = round(time.perf_counter() - t0, 2)
    print(f"  retrieval pass done in {retrieval_total:.1f}s")

    print("  timing pytest suite …")
    data["suite_time_s"] = _run_pytest_timer()
    print(f"  pytest: {data['suite_time_s']:.1f}s")

    print("  timing re-ingest …")
    data["reingest_time_s"] = _run_reingest_timer()
    print(f"  re-ingest: {data['reingest_time_s']:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nBaseline written to {out_path}")
    _print_latency_table(data)


def _print_latency_table(data: dict) -> None:
    print("\nLatency (ms) — warm/cold:")
    header = f"{'combo':<24} {'warm_p50':>10} {'cold_p50':>10}"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        for k in KS:
            key = f"{mode}_k{k}"
            w = data["latency_warm_ms"].get(key)
            c = data["latency_cold_ms"].get(key)
            ws = f"{w:.1f}" if w is not None else "n/a"
            cs = f"{c:.1f}" if c is not None else "n/a"
            print(f"  {key:<22} {ws:>10} {cs:>10}")
    print(f"\nFull pytest suite : {data.get('suite_time_s', '?'):.1f}s")
    print(f"Re-ingest time    : {data.get('reingest_time_s', '?'):.1f}s")


def _check(baseline_path: Path) -> int:
    """Compare a fresh run against the recorded baseline.  Returns exit code."""
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    questions = _load_questions()
    print(f"Running retrieval check against {baseline_path} …")
    t0 = time.perf_counter()
    fresh = _run_all(questions)
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"  retrieval pass done in {elapsed:.1f}s")

    failures: list[str] = []
    for qkey, q_data in baseline["questions"].items():
        for combo, expected in q_data.items():
            actual_combo = fresh["questions"].get(qkey, {}).get(combo)
            if actual_combo is None:
                failures.append(f"Q{qkey}/{combo}: MISSING in fresh run")
                continue
            # IDs (order matters)
            if actual_combo["ids"] != expected["ids"]:
                failures.append(
                    f"Q{qkey}/{combo}: ID ORDER CHANGED\n"
                    f"  expected: {expected['ids']}\n"
                    f"  actual  : {actual_combo['ids']}"
                )
            # Scores (absolute tolerance SCORE_TOL)
            for score_key in ("dense_score", "score"):
                exp_s = expected.get(score_key, [])
                act_s = actual_combo.get(score_key, [])
                if len(exp_s) != len(act_s):
                    failures.append(f"Q{qkey}/{combo}/{score_key}: length differs "
                                    f"({len(exp_s)} vs {len(act_s)})")
                    continue
                for i, (e, a) in enumerate(zip(exp_s, act_s)):
                    if abs(e - a) > SCORE_TOL:
                        failures.append(
                            f"Q{qkey}/{combo}/{score_key}[{i}]: "
                            f"{e} → {a} (delta {a-e:+.2e})"
                        )

    if failures:
        print(f"\n❌  GOLDEN MASTER FAILED — {len(failures)} difference(s):\n")
        for f in failures[:40]:   # cap output
            print(" ", f)
        if len(failures) > 40:
            print(f"  … {len(failures) - 40} more (see full output)")
    else:
        print("\n✅  Golden master: all IDs and scores match.")

    # Latency delta table
    print("\nLatency delta (ms) — new minus baseline (warm/cold):")
    header = f"{'combo':<24} {'Δwarm':>10} {'Δcold':>10}"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        for k in KS:
            key = f"{mode}_k{k}"
            bw = baseline["latency_warm_ms"].get(key)
            bc = baseline["latency_cold_ms"].get(key)
            fw = fresh["latency_warm_ms"].get(key)
            fc = fresh["latency_cold_ms"].get(key)
            dw = f"{fw - bw:+.1f}" if fw is not None and bw is not None else "n/a"
            dc = f"{fc - bc:+.1f}" if fc is not None and bc is not None else "n/a"
            print(f"  {key:<22} {dw:>10} {dc:>10}")

    # Suite + re-ingest timing
    print("\nSuite / re-ingest timing:")
    for key in ("suite_time_s", "reingest_time_s"):
        b = baseline.get(key)
        # We don't re-run suite/ingest in --check; just show the baseline value.
        label = key.replace("_", " ").replace("time s", "time")
        print(f"  {label:<20}: baseline {b:.1f}s  (not re-run in --check)")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-master retrieval benchmark")
    parser.add_argument("--check", action="store_true",
                        help="Check fresh run against saved baseline; fail on difference")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Path to golden JSON file (default: {DEFAULT_OUT})")
    parser.add_argument("--no-suite", action="store_true",
                        help="Skip pytest suite timing (record mode only)")
    parser.add_argument("--no-reingest", action="store_true",
                        help="Skip re-ingest timing (record mode only)")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.check:
        if not out_path.exists():
            sys.exit(f"ERROR: baseline file not found: {out_path}")
        sys.exit(_check(out_path))

    # Record mode
    questions = _load_questions()
    print(f"Recording golden master for {len(questions)} questions × "
          f"{len(MODES)} modes × {len(KS)} k-values …")
    t0_total = time.perf_counter()
    data = _run_all(questions)
    retrieval_total = round(time.perf_counter() - t0_total, 2)
    print(f"  retrieval pass done in {retrieval_total:.1f}s")

    if not args.no_suite:
        print("  timing pytest suite …")
        data["suite_time_s"] = _run_pytest_timer()
        print(f"  pytest: {data['suite_time_s']:.1f}s")
    else:
        data["suite_time_s"] = None

    if not args.no_reingest:
        print("  timing re-ingest …")
        data["reingest_time_s"] = _run_reingest_timer()
        print(f"  re-ingest: {data['reingest_time_s']:.1f}s")
    else:
        data["reingest_time_s"] = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nBaseline written to {out_path}")
    _print_latency_table(data)


if __name__ == "__main__":
    main()
