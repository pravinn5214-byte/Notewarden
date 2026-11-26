"""Tests for evaluate.py --paired: paired per-question hit differences with
bootstrap CIs, and the section-replacing save. No store/retrieval needed —
run_paired is exercised through monkeypatched evaluate_mode.

Run: python -m pytest tests/test_paired_eval.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate


def _fake_evaluate_mode(questions, mode, k, db_dir):
    """Deterministic per-question reciprocal ranks: 'win' store hits questions
    0-39 only; 'head' store additionally hits 40-43 (4-question gain)."""
    treated = db_dir == "db_head"
    hit = {0: 1, 1: 2, 2: 3}          # same ranks in both stores
    out = {"overall": [], "keyword": [], "semantic": []}
    for i, _q in enumerate(questions):
        if i < 40:
            rr = 1 / hit[i % 3]
        elif treated and i < 44:
            rr = 1.0
        else:
            rr = 0.0
        out["overall"].append(rr)
        out["keyword" if i % 2 == 0 else "semantic"].append(rr)
    return out


def test_paired_bootstrap_ci_centres_on_mean_diff():
    # 10 questions: +1 four times, 0 six times -> mean diff 0.4.
    diffs = [1] * 4 + [0] * 6
    lo, hi = evaluate.paired_bootstrap_ci(diffs, n=1000, seed=42)
    assert lo <= 0.4 <= hi
    # Deterministic given the same seed.
    assert (lo, hi) == evaluate.paired_bootstrap_ci(diffs, n=1000, seed=42)


def test_paired_bootstrap_ci_all_zero_is_zero():
    assert evaluate.paired_bootstrap_ci([0] * 45) == (0.0, 0.0)
    assert evaluate.paired_bootstrap_ci([]) == (0.0, 0.0)


def test_run_paired_rows_exclude_zero_only_when_ci_signs_differ(monkeypatch, tmp_path):
    questions = [{"question": f"q{i}", "expected_source": "s", "expected_page": 1,
                  "tag": "keyword"} for i in range(45)]
    monkeypatch.setattr(evaluate, "load_questions", lambda: questions)
    monkeypatch.setattr(evaluate, "evaluate_mode", _fake_evaluate_mode)
    monkeypatch.setattr(evaluate, "MODES", ["dense"])
    monkeypatch.setattr(evaluate, "KS", [3])
    lines = evaluate.run_paired("db_win", "db_head", "heading")
    # 4-question gain out of 45: diff 0.089; the 1000-resample bootstrap
    # (seed 42) gives a CI that excludes zero for this synthetic pattern.
    assert lines[-1] == ("| dense | 3 | 40/45 (0.889) | 44/45 (0.978) "
                         "| +0.089 | [+0.022, +0.178] | yes |")


def test_save_paired_replaces_section_and_appends_when_missing(monkeypatch, tmp_path):
    path = tmp_path / "res.md"
    path.write_text("# Results\n\n## Paired bootstrap: heading − window\n\n"
                    "| old | row |\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "PAIRED_RESULTS_FILE", path)
    evaluate._save_paired(["## Paired bootstrap: heading − window", "", "| new | row |"], path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Results\n\n")
    assert "| new | row |" in text and "| old | row |" not in text

    path2 = tmp_path / "res2.md"
    path2.write_text("# Results\n", encoding="utf-8")
    evaluate._save_paired(["## Paired bootstrap: heading − window", "", "| a | b |"], path2)
    text2 = path2.read_text(encoding="utf-8")
    assert text2.startswith("# Results\n\n## Paired bootstrap")
