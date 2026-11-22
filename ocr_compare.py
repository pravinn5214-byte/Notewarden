"""Compare OCR engines on one page against ground truth.

Usage:
    python ocr_compare.py <image-or-pdf> <ground-truth.txt> [--engines claude,ollama] [--page N]

Runs every available engine (default: tesseract plus the claude/ollama vision
engines, whichever are usable) on the page, and prints CER (character error
rate), WER (word error rate) and seconds per page. Lower is better.

No network calls happen unless a vision engine is selected and configured;
each vision call sends one page image.
"""
import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pymupdf

from ocr import ENGINE_NAMES, VISION_ENGINE_NAMES, build_engine, is_tesseract_available


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character error rate: edit distance / reference length."""
    ref = ref.strip()
    if not ref:
        return float("nan")
    return levenshtein("".join(hyp.split()), "".join(ref.split())) / len("".join(ref.split()))


def wer(hyp: str, ref: str) -> float:
    """Word error rate: Levenshtein over word sequences / reference word count."""
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    return levenshtein(h, r) / len(r)


def page_image(path: Path, page_num: int):
    """Render page N of a PDF, or open an image file directly."""
    if path.suffix.lower() == ".pdf":
        doc = pymupdf.open(path)
        page = doc[page_num - 1]
        import io
        from PIL import Image
        return Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png")))
    from PIL import Image
    return Image.open(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR engines against ground truth")
    parser.add_argument("source", type=Path, help="image file or PDF")
    parser.add_argument("truth", type=Path, help="ground-truth .txt file")
    parser.add_argument("--engines", default="tesseract,claude,ollama",
                        help="comma-separated subset of %s" % (ENGINE_NAMES,))
    parser.add_argument("--page", type=int, default=1, help="1-based PDF page number")
    args = parser.parse_args()

    truth = args.truth.read_text(encoding="utf-8")
    image = page_image(args.source, args.page)

    engines = []
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name == "auto":
            continue  # auto is a router, not a comparable transcription engine
        try:
            eng = build_engine(name)
        except ValueError as exc:
            print(f"skipping: {exc}")
            continue
        ok = eng.available()
        print(f"{name}: {'available' if ok else 'unavailable - skipped'}"
              + (f" (model {eng.model})" if ok and eng.model else ""))
        if ok:
            engines.append(eng)
    if not engines:
        print("No engines available.")
        sys.exit(1)

    rows = []
    for eng in engines:
        t0 = time.perf_counter()
        try:
            hyp = eng.ocr_image(image)
        except Exception as exc:
            print(f"{eng.name}: FAILED ({exc})")
            continue
        dt = time.perf_counter() - t0
        rows.append((eng.name, eng.model, cer(hyp, truth), wer(hyp, truth), dt))

    print(f"\n{'engine':<10} {'model':<24} {'CER':>7} {'WER':>7} {'time':>8}")
    for name, model, c, w, dt in rows:
        print(f"{name:<10} {model or '-':<24} {c:>6.1%} {w:>6.1%} {dt:>7.1f}s")


if __name__ == "__main__":
    main()
