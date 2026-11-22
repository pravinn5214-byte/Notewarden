"""Step 0 of the pipeline: split multi-topic questions into sub-queries.

A question like "Explain how ribosomes make proteins, and what the Tagverse
CRM lead pipeline does" averages two topics into one dense embedding, so one
topic fills the top-k and the other is never retrieved. decompose() splits
such questions into one self-contained search query per topic, using the same
LLM provider the app already uses for answers (claude or ollama).

Design:
- Kill switch: DECOMPOSE=off returns the question unchanged, no LLM call.
- Cheap skip rule (no LLM): short questions with no multi-topic signal
  (" and ", "compare", "difference between", ";", a second "?", ...) are
  returned as a single sub-query without calling the LLM.
- LLM split: strict JSON out; defensively validated (code fences stripped,
  non-strings, near-duplicates, over-length lists). Any failure falls back to
  the original question — decompose() never raises to the caller, and never
  logs the question text or API keys (only the reason and sub-query count).
- Cache: small in-process LRU keyed on (normalized question, provider, model)
  so Streamlit reruns and re-asks don't repeat the LLM call.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_SUBQUERIES = int(os.environ.get("DECOMPOSE_MAX_SUBQUERIES", "3"))

# Signals that a question may span more than one topic. Checked case-insensitively.
_TOPIC_SIGNALS = (
    " and ", " also ", " as well as ", "compare", "versus", " vs",
    "difference between", "both", ";",
)
MIN_WORDS_BEFORE_LLM = 8

SPLIT_PROMPT = """You split a student's question into search queries for a notes search engine. Return JSON only: {"subqueries": [...], "answer_as_parts": true|false}.
Rules: (1) If the question asks about several INDEPENDENT topics, return one self-contained search query per topic and set answer_as_parts true. (2) If it compares or relates two things (e.g. 'difference between X and Y', 'compare A and B'), return one query per item PLUS nothing else, and set answer_as_parts false, because the answer should be one integrated comparison. (3) If it is a single question, return it unchanged as the only subquery and set answer_as_parts false. (4) Each subquery must make sense on its own: resolve pronouns, repeat the subject, keep key technical terms and acronyms exactly as written. (5) Maximum 3 subqueries. (6) Do not answer the question. (7) No commentary."""


@dataclass
class Decomposition:
    """Result of splitting a question into sub-queries."""
    subqueries: list[str]
    answer_as_parts: bool = False
    used_llm: bool = False
    reason: str = "single"  # disabled | skipped_simple | split | single | fallback_error | fallback_invalid
    original: str = field(default="", repr=False)  # never logged

    @property
    def is_multi(self) -> bool:
        return len(self.subqueries) > 1


def _provider_and_model() -> tuple[str, str]:
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    if provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    else:
        model = "claude-sonnet-4-20250514"
    return provider, model


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _has_multi_topic_signal(question: str) -> bool:
    low = f" {question.lower()} "
    if any(sig in low for sig in _TOPIC_SIGNALS):
        return True
    if low.count("?") >= 2:  # more than the trailing one
        return True
    # numbered/bulleted list: "1. ... 2. ..." or "- ..." at a boundary
    if re.search(r"(?:^|\s)[1-9][.)]\s", question) or re.search(r"(?:^|\n)\s*[-*•]\s", question):
        return True
    return False


def _parse_llm_json(raw: str) -> dict | None:
    """Strip code fences and parse; None on any failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _valid_subqueries(items: list) -> list[str] | None:
    """Cleaned list or None if invalid (non-strings, <3 chars, duplicates)."""
    if not items or not all(isinstance(s, str) for s in items):
        return None
    cleaned = [s.strip() for s in items]
    if any(len(s) < 3 for s in cleaned):
        return None
    seen: set[str] = set()
    for s in cleaned:
        norm = _normalize(s)
        if norm in seen:
            return None
        seen.add(norm)
    return cleaned


_CACHE: dict[tuple[str, str, str], Decomposition] = {}
_CACHE_ORDER: list[tuple[str, str, str]] = []
_CACHE_MAX = 128


def _cache_get(key: tuple[str, str, str]) -> Decomposition | None:
    if key in _CACHE:
        _CACHE_ORDER.remove(key)
        _CACHE_ORDER.append(key)  # LRU touch
        return _CACHE[key]
    return None


def _cache_put(key: tuple[str, str, str], value: Decomposition) -> None:
    if key in _CACHE:
        return
    _CACHE[key] = value
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        old = _CACHE_ORDER.pop(0)
        _CACHE.pop(old, None)


def clear_cache() -> None:
    """Drop all cached decompositions (tests, provider changes)."""
    _CACHE.clear()
    _CACHE_ORDER.clear()


def decompose(question: str, llm=None) -> Decomposition:
    """Split `question` into sub-queries. Never raises; never logs the text.

    llm: optional callable (prompt, max_tokens) -> str, for tests; when None
    the real provider completion from generate.complete() is used.
    """
    question = (question or "").strip()
    if not question:
        return Decomposition([question], reason="single")

    provider, model = _provider_and_model()
    key = (_normalize(question), provider, model)

    if os.environ.get("DECOMPOSE", "on").lower() == "off":
        return Decomposition([question], reason="disabled", original=question)

    cached = _cache_get(key)
    if cached is not None:
        return cached

    words = len(question.split())
    if words < MIN_WORDS_BEFORE_LLM or not _has_multi_topic_signal(question):
        result = Decomposition([question], reason="skipped_simple", original=question)
        _cache_put(key, result)
        return result

    try:
        if llm is None:
            from generate import complete
            raw = complete(SPLIT_PROMPT + f"\n\nQuestion: {question}", max_tokens=200)
        else:
            raw = llm(SPLIT_PROMPT + f"\n\nQuestion: {question}", max_tokens=200)
    except Exception:
        logger.warning("Question decomposition failed (fallback_error); "
                       "subqueries=1")
        result = Decomposition([question], answer_as_parts=False,
                               used_llm=False, reason="fallback_error",
                               original=question)
        _cache_put(key, result)
        return result

    data = _parse_llm_json(raw)
    items = _valid_subqueries(data.get("subqueries", [])) if data else None
    if items is None:
        logger.warning("Question decomposition returned invalid output "
                       "(fallback_invalid); subqueries=1")
        result = Decomposition([question], answer_as_parts=False,
                               used_llm=True, reason="fallback_invalid",
                               original=question)
        _cache_put(key, result)
        return result

    items = items[:MAX_SUBQUERIES]
    parts_flag = bool(data.get("answer_as_parts", False))
    # A comparison with a single returned query is just a single query.
    if len(items) == 1:
        parts_flag = False
    result = Decomposition(items, answer_as_parts=parts_flag,
                           used_llm=True, reason="split", original=question)
    logger.info("Question decomposed (split); subqueries=%d", len(items))
    _cache_put(key, result)
    return result
