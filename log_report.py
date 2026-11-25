"""Query-log report CLI: spot weak retrieval from logs/queries.db.

Usage:
    python log_report.py                # all-time report
    python log_report.py --since 7d     # last 7 days (also 24h, 30d, ...)
    python log_report.py --csv out.csv  # also dump the raw rows to CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from query_log import _load_json, log_path, rows_since

_SINCE_RE = re.compile(r"^(\d+)\s*(h|d|w|mo)$", re.I)
_UNITS = {"h": "hours", "d": "days", "w": "weeks", "mo": lambda n: n * 30}


def since_ts(spec: str) -> str | None:
    """'7d' -> naive-UTC ISO timestamp 7 days ago; None for invalid input.
    Naive because log_query stores naive-UTC ISO strings (direct string
    comparison in SQLite)."""
    m = _SINCE_RE.match(spec.strip())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit == "mo":
        delta = timedelta(days=n * 30)
    else:
        delta = timedelta(**{_UNITS[unit]: n})
    return (datetime.now(timezone.utc) - delta).replace(tzinfo=None).isoformat(timespec="milliseconds")


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    # Nearest-rank percentile (small samples: ceil(0.95*n) keeps the max honest).
    idx = max(0, min(len(s) - 1, -(-int(0.95 * 100 * len(s)) // 100) - 1))
    return s[idx]


def _results_of(row) -> list[dict]:
    return _load_json(row["results"])


def build_report(rows: list) -> str:
    lines: list[str] = ["# Query log report", ""]
    if not rows:
        return "\n".join(lines + ["No queries logged yet."])

    n = len(rows)
    gated = sum(1 for r in rows if r["gated_out"])
    no_answer = sum(1 for r in rows if r["error"]) + gated
    clarified = sum(1 for r in rows if r["clarified"])
    retried = sum(1 for r in rows if (r["retries"] or 0) > 0)
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]

    lines += [
        f"- Total queries: {n}",
        f"- No-answer / gated-out rate: {no_answer / n:.0%} ({no_answer} gated, "
        f"{sum(1 for r in rows if r['error'])} errored)",
        f"- Clarify rate: {clarified / n:.0%} ({clarified})",
        f"- Retry rate: {retried / n:.0%} ({retried})",
        f"- Average latency: {sum(latencies) / len(latencies):.0f} ms"
        if latencies else "- Average latency: n/a",
        f"- p95 latency: {p95(latencies):.0f} ms" if latencies else "- p95 latency: n/a",
        "",
        "## 10 lowest top_dense_score queries (weakest retrieval)",
        "",
        "| ts | query | top dense | gated | mode |",
        "|---|---|---|---|---|",
    ]
    weakest = sorted((r for r in rows if r["top_dense_score"] is not None),
                     key=lambda r: r["top_dense_score"])[:10]
    for r in weakest:
        q = (r["original_query"] or "").replace("|", "\\|")
        if len(q) > 60:
            q = q[:57] + "..."
        lines.append(f"| {r['ts'][:16]} | {q} | {r['top_dense_score']:.3f} "
                     f"| {'yes' if r['gated_out'] else 'no'} | {r['retrieval_mode']} |")

    file_counts: Counter = Counter()
    for r in rows:
        for res in _results_of(r):
            if res.get("file"):
                file_counts[res["file"]] += 1
    lines += ["", "## Most-used files", ""]
    if file_counts:
        lines += ["| file | times in results |", "|---|---|"]
        for f, c in file_counts.most_common():
            lines.append(f"| {f} | {c} |")
    else:
        lines.append("No results recorded yet.")
    return "\n".join(lines)


def write_csv(rows: list, out: str) -> None:
    cols = ["id", "ts", "session_id", "original_query", "rewritten_query",
            "retrieval_mode", "top_k", "decompose_used", "agent_mode",
            "sub_queries", "results", "top_dense_score", "gated_out",
            "unsupported_topics", "retries", "clarified", "llm_provider",
            "latency_ms", "error", "page_filter"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def main() -> None:
    ap = argparse.ArgumentParser(description="Report on the query log")
    ap.add_argument("--since", help="time window, e.g. 24h, 7d, 4w, 2mo")
    ap.add_argument("--csv", dest="csv_out", help="also dump raw rows to this CSV file")
    args = ap.parse_args()

    since = None
    if args.since:
        since = since_ts(args.since)
        if since is None:
            raise SystemExit("--since must look like '24h', '7d', '4w' or '2mo'")

    rows = rows_since(since)
    report = build_report(rows)
    print(report)
    if args.csv_out:
        write_csv(rows, args.csv_out)
        print(f"\nRaw rows written to {args.csv_out}")


if __name__ == "__main__":
    main()
