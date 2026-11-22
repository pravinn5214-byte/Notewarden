"""OCR support for scanned pages, image uploads and handwritten notes.

Engines behind the OcrEngine interface, picked via OCR_ENGINE:
    - "tesseract" (default): free, local, fast; good on printed text, poor on
      cursive handwriting.
    - "claude": Anthropic vision model (ANTHROPIC_API_KEY). Page images are
      sent to the API — best transcription quality, not local.
    - "ollama": a local vision model via Ollama (OLLAMA_HOST respected).
      Local and private, quality depends on the model pulled.
    - "auto": Tesseract first; a page escalates to the vision engine named in
      OCR_VISION_ENGINE when Tesseract's mean word confidence is below
      OCR_CONF_THRESHOLD (default 60) or the text looks like garbage.

To swap in another backend (TrOCR, a different vision API), subclass
OcrEngine and register it in get_engine() / ENGINE_NAMES.

Usage:
    Tesseract binary: install (see README "Setting up OCR"), then it's found
    on PATH. If it lives elsewhere, set TESSERACT_CMD in .env.
    OCR_LANG / OCR_MIN_CHARS in .env tune language and the fallback threshold.
"""
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pymupdf

OCR_CACHE_DIR = Path("ocr_cache")
DEFAULT_DPI = 300
DEFAULT_LANG = "eng"
# Below this many non-whitespace chars, a page counts as scanned (no text layer).
OCR_MIN_CHARS_DEFAULT = 30

ENGINE_NAMES = ("tesseract", "claude", "ollama", "auto")
VISION_ENGINE_NAMES = ("claude", "ollama")
DEFAULT_ENGINE = "tesseract"
# auto: escalate to the vision engine below this mean word confidence.
OCR_CONF_THRESHOLD_DEFAULT = 60
# Longest image edge sent to vision models.
VISION_MAX_EDGE = 1600

WINDOWS_DEFAULT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Shared transcription prompt for the vision engines.
TRANSCRIBE_PROMPT = (
    "Transcribe this page of handwritten or printed notes exactly as written. "
    "Preserve line breaks, headings, bullet points and numbering. Write math "
    "as plain text or LaTeX. For diagrams or figures, write [diagram: <short "
    "description>]. Mark unreadable words as [illegible]. Output only the "
    "transcription, with no commentary."
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or default


class OcrEngine:
    """Interface for OCR backends. Subclass and override ocr_page / ocr_image."""

    name = "base"

    def available(self) -> bool:
        return False

    @property
    def model(self) -> str:
        """Model identifier recorded in chunk metadata / cache keys."""
        return ""

    def ocr_page(self, page: "pymupdf.Page", dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG) -> str:
        raise NotImplementedError

    def ocr_image(self, image, lang: str = DEFAULT_LANG) -> str:
        """OCR a PIL Image."""
        raise NotImplementedError


class TesseractEngine(OcrEngine):
    """pytesseract backend. Preprocessing: grayscale always; autocontrast and
    a fixed threshold are optional (OCR_PREPROCESS=contrast|threshold)."""

    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract
        self._pytesseract = pytesseract
        cmd = _env("TESSERACT_CMD")
        if not cmd and sys.platform == "win32" and Path(WINDOWS_DEFAULT_CMD).exists():
            cmd = WINDOWS_DEFAULT_CMD
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def available(self) -> bool:
        try:
            self._pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _preprocess(self, image, mode: str | None):
        from PIL import ImageOps
        image = image.convert("L")  # grayscale
        if mode in ("contrast", "threshold"):
            image = ImageOps.autocontrast(image)
        if mode == "threshold":
            image = image.point(lambda p: 255 if p > 160 else 0)
        return image

    def ocr_image(self, image, lang: str = DEFAULT_LANG) -> str:
        pre = _env("OCR_PREPROCESS") or None
        return self._pytesseract.image_to_string(self._preprocess(image, pre), lang=lang)

    def ocr_page(self, page: "pymupdf.Page", dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG) -> str:
        pix = page.get_pixmap(dpi=dpi)
        from PIL import Image
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        return self.ocr_image(image, lang=lang)

    def mean_word_confidence(self, image, lang: str = DEFAULT_LANG) -> float:
        """Mean confidence of recognized words (used by 'auto' mode). -1 if none."""
        data = self._pytesseract.image_to_data(self._preprocess(image, _env("OCR_PREPROCESS") or None), lang=lang, output_type=self._pytesseract.Output.DICT)
        confs = [float(c) for c, t in zip(data["conf"], data["text"])
                 if c not in ("-1", "-1.0") and t.strip()]
        return sum(confs) / len(confs) if confs else -1.0


class ClaudeVisionEngine(OcrEngine):
    """Anthropic vision model. Page images leave the machine (ANTHROPIC_API_KEY)."""

    name = "claude"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"  # override with OCR_CLAUDE_MODEL

    def __init__(self) -> None:
        self._model = _env("OCR_CLAUDE_MODEL", self.DEFAULT_MODEL)
        self._api_key = _env("ANTHROPIC_API_KEY")

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        return bool(self._api_key)

    def ocr_image(self, image, lang: str = DEFAULT_LANG) -> str:
        image = _downscale(image)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return self._post({
            "model": self._model,
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }, {"type": "text", "text": TRANSCRIBE_PROMPT}],
            }],
        })

    def _post(self, payload: dict) -> str:
        import requests

        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for OCR_ENGINE=claude).")
        last: Exception | None = None
        for attempt in range(4):  # ~0s, 2s, 4s, 8s backoff
            if attempt:
                time.sleep(2 ** (attempt - 1) * 2)
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self._api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = RuntimeError(f"Anthropic API error {resp.status_code} (transient)")
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Anthropic API error {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                return "".join(b.get("text", "") for b in data.get("content", [])
                               if b.get("type") == "text").strip()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
        raise RuntimeError(f"Claude vision OCR failed after retries: {last}")


class OllamaVisionEngine(OcrEngine):
    """Local Ollama vision model (OLLAMA_HOST respected). Nothing leaves the machine."""

    name = "ollama"
    DEFAULT_MODEL = "llama3.2-vision"  # override with OCR_OLLAMA_MODEL

    def __init__(self) -> None:
        self._model = _env("OCR_OLLAMA_MODEL", self.DEFAULT_MODEL)
        self._host = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    @property
    def model(self) -> str:
        return self._model

    def available(self) -> bool:
        import requests
        try:
            return requests.get(f"{self._host}/api/tags", timeout=5).ok
        except Exception:
            return False

    def ocr_image(self, image, lang: str = DEFAULT_LANG) -> str:
        image = _downscale(image)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        import requests

        last: Exception | None = None
        for attempt in range(4):
            if attempt:
                time.sleep(2 ** (attempt - 1) * 2)
            try:
                resp = requests.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": TRANSCRIBE_PROMPT,
                                      "images": [b64]}],
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=300,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = RuntimeError(f"Ollama API error {resp.status_code} (transient)")
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text[:200]}")
                return resp.json().get("message", {}).get("content", "").strip()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
        raise RuntimeError(f"Ollama vision OCR failed after retries: {last}")


def _downscale(image, max_edge: int = VISION_MAX_EDGE):
    """Longest edge <= max_edge, preserving aspect."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))))


class AutoEngine(OcrEngine):
    """Tesseract first; low-confidence/garbage pages escalate to a vision engine
    (OCR_VISION_ENGINE=claude|ollama). Records which engine produced a page."""

    name = "auto"

    def __init__(self, tesseract: TesseractEngine | None = None, vision: OcrEngine | None = None) -> None:
        self._tesseract = tesseract or TesseractEngine()
        vname = _env("OCR_VISION_ENGINE", "claude")
        self._vision = vision or build_engine(vname)

    @property
    def model(self) -> str:
        return ""

    def available(self) -> bool:
        return self._tesseract.available() or self._vision.available()

    def ocr_image(self, image, lang: str = DEFAULT_LANG) -> str:
        ts_text = self._tesseract.ocr_image(image, lang=lang)
        conf = self._tesseract.mean_word_confidence(image, lang=lang)
        if conf >= float(_env("OCR_CONF_THRESHOLD", OCR_CONF_THRESHOLD_DEFAULT)) and not _mostly_garbage(ts_text):
            self.last_engine_used = self._tesseract
            return ts_text
        if self._vision.available():
            try:
                out = self._vision.ocr_image(image, lang=lang)
                self.last_engine_used = self._vision
                return out
            except Exception as exc:
                print(f"WARNING: vision OCR failed ({exc}); falling back to Tesseract", file=sys.stderr)
        self.last_engine_used = self._tesseract
        return ts_text

    def ocr_page(self, page: "pymupdf.Page", dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG) -> str:
        pix = page.get_pixmap(dpi=dpi)
        from PIL import Image
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        return self.ocr_image(image, lang=lang)


def _mostly_garbage(text: str) -> bool:
    """True if the alphabetic tokens look like OCR noise: very few dictionary
    words among tokens of length >= 3."""
    try:
        from wordninja import lexicon
        words = set(w for w in lexicon() if len(w) >= 3)
    except ImportError:
        words = {"the", "and", "of", "to", "in", "is", "that", "it", "for", "as",
                 "with", "was", "on", "are", "this", "be", "or", "by", "from", "not"}
    toks = [t for t in text.lower().split() if len(t) >= 3 and t.isalpha()]
    if not toks:
        return True  # nothing readable at all
    known = sum(1 for t in toks if t in words)
    return known / len(toks) < 0.4


# --- Engine construction and selection ----------------------------------------

_engine: OcrEngine | None = None
_engine_name: str | None = None


def build_engine(name: str) -> OcrEngine:
    if name == "tesseract":
        return TesseractEngine()
    if name == "claude":
        return ClaudeVisionEngine()
    if name == "ollama":
        return OllamaVisionEngine()
    if name == "auto":
        return AutoEngine()
    raise ValueError(f"Unknown OCR_ENGINE '{name}'. Use one of {ENGINE_NAMES}.")


def get_engine(name: str | None = None) -> OcrEngine:
    """Return the configured engine, rebuilt if `name` differs from the cached one."""
    global _engine, _engine_name
    name = name or _env("OCR_ENGINE", DEFAULT_ENGINE)
    if name not in ENGINE_NAMES:
        raise ValueError(f"Unknown OCR_ENGINE '{name}'. Use one of {ENGINE_NAMES}.")
    if _engine is None or _engine_name != name:
        _engine = build_engine(name)
        _engine_name = name
    return _engine


def is_tesseract_available() -> bool:
    try:
        return TesseractEngine().available()
    except Exception:
        return False


def ocr_page(page: "pymupdf.Page", dpi: int = DEFAULT_DPI, lang: str = DEFAULT_LANG,
             engine_name: str | None = None) -> str:
    return get_engine(engine_name).ocr_page(page, dpi=dpi, lang=lang)


def ocr_image_file(path: str | Path, lang: str = DEFAULT_LANG,
                   engine_name: str | None = None) -> str:
    """OCR a standalone image file (.png/.jpg/.jpeg)."""
    from PIL import Image
    return get_engine(engine_name).ocr_image(Image.open(path), lang=lang)


# --- OCR text cache -----------------------------------------------------------
# Keyed by (file content hash, page, engine, model): switching engines or models
# never returns another engine's stale text. A JSON file per source file under
# ocr_cache/; entries written before engines existed ({"<page>": "<text>"})
# are treated as tesseract results and stay valid.

def _cache_path(file_hash: str) -> Path:
    OCR_CACHE_DIR.mkdir(exist_ok=True)
    return OCR_CACHE_DIR / f"{file_hash}.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_key_for(file_hash: str, page_num: int, engine_name: str, model: str) -> str:
    """"<page>|<engine>|<model>" — the cache-dict key for one OCR result."""
    return f"{page_num}|{engine_name}|{model}"


def _legacy_key(page_num: int) -> str:
    return str(page_num)


def cached_ocr_page(pdf_path: Path, page_num: int, page: "pymupdf.Page",
                    lang: str = DEFAULT_LANG, engine: OcrEngine | None = None) -> str:
    """OCR `page`, using/updating the (file hash, page, engine, model) cache."""
    engine = engine or get_engine()
    key = cache_key_for(_file_hash(pdf_path), page_num, engine.name, engine.model)
    legacy = _legacy_key(page_num)
    path = _cache_path(_file_hash(pdf_path))
    cache: dict = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}  # corrupt cache file: re-OCR rather than fail
    # Old-format entries (page -> text) belong to tesseract.
    if legacy in cache and legacy != key and isinstance(cache[legacy], str):
        cache[key] = cache.pop(legacy)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if key in cache:
        return cache[key]
    text = engine.ocr_page(page, lang=lang)
    cache[key] = text
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return text


def clear_cache(file_hash: str | None = None) -> None:
    """Drop the whole OCR cache, or one file's entry."""
    if not OCR_CACHE_DIR.exists():
        return
    if file_hash is None:
        for p in OCR_CACHE_DIR.glob("*.json"):
            p.unlink()
    else:
        _cache_path(file_hash).unlink(missing_ok=True)
