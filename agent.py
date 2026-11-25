"""Bounded agentic layer: follow-up rewrite, weak-retrieval retry, clarify-by-rule.

The control flow lives in plain Python; the LLM only makes narrow, validated
decisions (one JSON answer each), so small local models (Ollama llama3.2) stay
reliable. Hard caps: one retrieval retry and at most one clarification per
turn — no open-ended tool-calling loop.

Every LLM decision validates its JSON and falls back to "no agentic behaviour"
(plain retrieve + answer). Nothing here can crash the chat or loop.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HISTORY_TURNS = int(os.environ.get("AGENT_HISTORY_TURNS", "4"))
MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "1"))

# Reference words that suggest the question depends on earlier turns.
_FOLLOWUP_MARKERS = (
    " that ", " it ", " this ", " those ", " these ", " them ", " they ",
    "explain more", "more about", "tell me more", "go on", "continue",
    "why", "how about", "what about", "again", "instead",
)
_VAGUE_QUESTIONS = {"what", "why", "how", "explain", "tell me", "more"}

_REWRITE_PROMPT_TEMPLATE = """Rewrite the user's follow-up question as ONE standalone search query for a notes search engine. Return JSON only: {"query": "..."}.
Rules: (1) Resolve pronouns and references ("that", "it") using the recent conversation. (2) Keep key technical terms and acronyms exactly as written. (3) Do not answer the question. (4) No commentary.

Recent conversation:
__HISTORY__

Follow-up question: __QUESTION__

JSON:"""

_RETRY_PROMPT_TEMPLATE = """The search query below found no relevant notes. Rewrite it ONCE with different wording or synonyms (same topic). Return JSON only: {"query": "..."}.
Rules: (1) Same topic, different words (e.g. "make proteins" -> "protein synthesis translation"). (2) Keep acronyms. (3) Do not answer. (4) No commentary.

Query: __QUERY__

JSON:"""

_SUFFICIENCY_PROMPT_TEMPLATE = """Can the context below fully answer the question? Return JSON only: {"sufficient": true|false}.
Rules: (1) Judge only whether the context contains the needed facts. (2) Partial context counts as false. (3) No commentary.

Context:
__CONTEXT__

Question: __QUESTION__

JSON:"""

_CLARIFY_PROMPT_TEMPLATE = """Write ONE short clarifying question to ask the user. Return JSON only: {"question": "..."}.
Rules: (1) Ask what is needed to search their notes successfully. (2) One sentence, no options list. (3) No commentary.

Original question: __QUESTION__
What is missing: __MISSING__

JSON:"""


@dataclass
class AgentResult:
    """Outcome of one agent turn."""
    action: str                       # "answer" | "clarify"
    answer_or_question: str           # grounded answer or clarifying question
    sources: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def _parse_json_value(raw: str) -> dict | None:
    """Strip code fences and parse a JSON object; None on any failure."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _history_block(history: list[dict]) -> str:
    return "\n".join(f"{h['role']}: {h['text']}" for h in history)


def needs_followup_rewrite(question: str) -> bool:
    """Cheap skip rule (no LLM): True when the question references earlier
    turns. Same style as decompose's skip rule — default to skipping the LLM
    call unless a marker or short-vague question says otherwise."""
    low = f" {question.lower().strip()} "
    if len(question.split()) <= 3:
        return True
    if any(m in low for m in _FOLLOWUP_MARKERS):
        return True
    if low.strip().rstrip("?") in _VAGUE_QUESTIONS:
        return True
    return False


def rewrite_followup(question: str, history: list[dict], llm=None) -> tuple[str, dict]:
    """Turn a follow-up into a standalone query using recent turns.

    Returns (query, trace). Falls back to the original question on any
    validation failure (empty, over 60 words, bad JSON, LLM error).
    """
    trace: dict = {"rewrite_used": False, "rewritten_query": None}
    if not history or not needs_followup_rewrite(question):
        trace["rewrite_skipped"] = True
        return question, trace
    prompt = (_REWRITE_PROMPT_TEMPLATE.replace("__HISTORY__", _history_block(history))
                  .replace("__QUESTION__", question))
    try:
        raw = llm(prompt, max_tokens=100) if llm else _complete(prompt, 100)
    except Exception:
        logger.warning("Follow-up rewrite failed; using the original question")
        return question, trace
    data = _parse_json_value(raw)
    query = data.get("query") if data else None
    if not isinstance(query, str) or not query.strip() or len(query.split()) > 60:
        logger.warning("Follow-up rewrite invalid; using the original question")
        return question, trace
    query = query.strip()
    trace.update(rewrite_used=True, rewritten_query=query)
    return query, trace


def is_vague(question: str, history: list[dict]) -> bool:
    """Hard vagueness signal: just a pronoun/reference word or bare question
    word with no history to resolve it."""
    low = question.lower().strip().rstrip("?").strip()
    if history:
        return False
    return low in _VAGUE_QUESTIONS or low in ("that", "it", "this", "them", "those") or len(low.split()) <= 2


def retrieve_with_fallback(
    query: str,
    mode: str,
    top_k: int,
    files: list[str] | None,
    db_dir: str,
    llm=None,
    trace: dict | None = None,
    page_ranges: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[dict], set[int], dict, dict]:
    """Retrieve; if every result/sub-query fails the gate, rewrite ONCE and
    retry. Returns (hits, supported, decomposition, trace)."""
    trace = trace if trace is not None else {}
    from decompose import decompose
    import retrieve as _retrieve_mod
    retrieve, retrieve_multi = _retrieve_mod.retrieve, _retrieve_mod.retrieve_multi

    decomposition = decompose(query)
    subs = decomposition.subqueries
    queries_tried = [subs if len(subs) > 1 else query]
    if len(subs) > 1:
        hits, supported = retrieve_multi(subs, mode=mode, top_k=top_k,
                                         files=files, db_dir=db_dir,
                                         page_ranges=page_ranges)
    else:
        hits = retrieve(query, k=top_k, mode=mode, allowed_sources=files, db_dir=db_dir,
                        page_ranges=page_ranges)
        supported = {0} if hits else set()

    def _all_failed(hits: list[dict], supported: set[int], n_subs: int) -> bool:
        if not hits:
            return True
        # Per-sub-query gate: any supported sub-query with a dense hit >= MIN_SCORE passes.
        from generate import MIN_SCORE
        strong = {h.get("subquery_index", 0) for h in hits
                  if h.get("dense_score", 0.0) >= MIN_SCORE}
        return not strong

    if not _all_failed(hits, supported, len(subs)):
        trace["retrieval"] = {"queries_tried": queries_tried, "retry": False}
        trace["retry"] = False
        return hits, supported, decomposition, trace

    # Weak first pass: one bounded rewrite-and-retry.
    for attempt in range(MAX_RETRIES):
        rewrite_prompt = _RETRY_PROMPT_TEMPLATE.replace("__QUERY__", query)
        try:
            raw = llm(rewrite_prompt, max_tokens=100) if llm else _complete(rewrite_prompt, 100)
        except Exception:
            logger.warning("Retry rewrite failed; keeping the first results")
            break
        data = _parse_json_value(raw)
        new_query = data.get("query") if data else None
        if not isinstance(new_query, str) or not new_query.strip():
            logger.warning("Retry rewrite invalid; keeping the first results")
            break
        new_query = new_query.strip()
        queries_tried.append(new_query)
        sub2 = decompose(new_query).subqueries
        if len(sub2) > 1:
            hits2, supported2 = retrieve_multi(sub2, mode=mode, top_k=top_k,
                                               files=files, db_dir=db_dir,
                                               page_ranges=page_ranges)
        else:
            hits2 = retrieve(new_query, k=top_k, mode=mode, allowed_sources=files, db_dir=db_dir,
                             page_ranges=page_ranges)
            supported2 = {0} if hits2 else set()
        if not _all_failed(hits2, supported2, len(sub2)):
            trace["retrieval"] = {"queries_tried": queries_tried, "retry": True,
                                  "retried_query": new_query}
            trace["retry"] = True
            return hits2, supported2, decomposition, trace
        hits, supported = hits2, supported2  # keep the better of the two (both weak)

    trace["retrieval"] = {"queries_tried": queries_tried, "retry": True,
                          "retried_query": queries_tried[-1] if len(queries_tried) > 1 else None,
                          "still_weak": True}
    trace["retry"] = True
    return hits, supported, decomposition, trace


def sufficiency_check(question: str, hits: list[dict], llm=None) -> tuple[bool, dict]:
    """Optional yes/no check: can the context answer the question?

    AGENT_SUFFICIENCY_CHECK=auto|on|off; auto = on only when provider is claude.
    Returns (sufficient, trace). Falls back to True (sufficient) on any failure
    so a small-model hiccup can't trigger unnecessary retries.
    """
    setting = os.environ.get("AGENT_SUFFICIENCY_CHECK", "auto").lower()
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()
    enabled = setting == "on" or (setting == "auto" and provider == "claude")
    trace = {"sufficiency_check": enabled}
    if not enabled or not hits:
        return True, trace
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits[:8])
    prompt = (_SUFFICIENCY_PROMPT_TEMPLATE.replace("__CONTEXT__", context)
                  .replace("__QUESTION__", question))
    try:
        raw = llm(prompt, max_tokens=50) if llm else _complete(prompt, 50)
    except Exception:
        logger.warning("Sufficiency check failed; assuming sufficient")
        trace["sufficiency_error"] = True
        return True, trace
    data = _parse_json_value(raw)
    if data is None or not isinstance(data.get("sufficient"), bool):
        logger.warning("Sufficiency check invalid; assuming sufficient")
        trace["sufficiency_error"] = True
        return True, trace
    sufficient = data["sufficient"]
    trace["sufficient"] = sufficient
    return sufficient, trace


def phrase_clarifying_question(question: str, missing: str, llm=None) -> tuple[str, dict]:
    """LLM phrases the clarify question; template fallback if it fails."""
    trace = {"clarify_phrased_by": "template"}
    template = f"Could you tell me more about what you mean by \"__QUESTION__\"? I couldn't find relevant notes."
    prompt = (_CLARIFY_PROMPT_TEMPLATE.replace("__QUESTION__", question)
                  .replace("__MISSING__", missing))
    try:
        raw = llm(prompt, max_tokens=80) if llm else _complete(prompt, 80)
    except Exception:
        return template, trace
    data = _parse_json_value(raw)
    q = data.get("question") if data else None
    if not isinstance(q, str) or not q.strip() or len(q.split()) > 40:
        return template, trace
    trace["clarify_phrased_by"] = "llm"
    return q.strip(), trace


def _complete(prompt: str, max_tokens: int) -> str:
    from generate import complete
    return complete(prompt, max_tokens=max_tokens)


def answer_with_agent(
    question: str,
    history: list[dict] | None = None,
    mode: str = "hybrid_rerank",
    top_k: int = 4,
    files: list[str] | None = None,
    db_dir: str | None = None,
    llm=None,
    page_ranges: dict[str, tuple[int, int]] | None = None,
) -> AgentResult:
    """One bounded agent turn: rewrite -> retrieve -> (retry) -> (clarify) -> answer.

    history: recent turns oldest-first ([{"role", "text"}]); the last
    AGENT_HISTORY_TURNS are used. llm: fake (prompt, max_tokens) -> str for tests.
    page_ranges: per-file inclusive page filters, passed through to retrieval
    and into the trace/query log.
    """
    from generate import MIN_SCORE
    import store as store_mod
    import time as _time

    history = history or []
    recent = history[-HISTORY_TURNS:]
    trace: dict = {"steps": [], "queries_tried": [], "retry": False, "clarify": None}
    _turn_start = _time.monotonic()

    def _log_turn(action: str, text: str, hits: list[dict], supported: set,
                  decomposition) -> None:
        """Query log: best-effort, never breaks the turn (log_query itself
        swallows errors; this wrapper guards field extraction too)."""
        try:
            import os as _os
            from query_log import log_query, results_payload, top_dense, gated
            from generate import MIN_SCORE as _gate
            subqueries = decomposition.subqueries if decomposition is not None else []
            multi = len(subqueries) > 1
            log_query(
                session_id=_os.environ.get("QUERY_LOG_SESSION_ID", ""),
                original_query=question,
                rewritten_query=trace.get("rewritten_query"),
                retrieval_mode=mode,
                top_k=top_k,
                decompose_used=multi,
                agent_mode=True,
                sub_queries=subqueries if multi else None,
                results=results_payload(hits),
                top_dense_score=top_dense(hits),
                gated_out=action != "clarify" and gated(hits, _gate),
                unsupported_topics=[subqueries[i] for i in range(len(subqueries))
                                    if i not in supported] if multi else [],
                retries=1 if trace.get("retry") else 0,
                clarified=action == "clarify",
                llm_provider=_os.environ.get("LLM_PROVIDER", "claude"),
                latency_ms=(_time.monotonic() - _turn_start) * 1000.0,
                answer=text if action == "answer" else None,
                page_filter=[{"file": f, "start": s, "end": e}
                             for f, (s, e) in sorted(page_ranges.items())] if page_ranges else None,
            )
        except Exception as exc:
            logger.warning("Query logging failed (ignored): %s", exc)

    def step(name: str, **info) -> None:
        entry = {"step": name, **info}
        trace["steps"].append(entry)

    prev_was_clarify = bool(history) and history[-1].get("role") == "assistant" and \
        history[-1].get("clarify", False)

    db_dir = db_dir or store_mod.db_path()

    # (a) Follow-up rewrite.
    query, rw = rewrite_followup(question, recent, llm=llm)
    step("rewrite", **{k: v for k, v in rw.items() if k != "rewritten_query"})
    if rw.get("rewritten_query"):
        trace["rewritten_query"] = rw["rewritten_query"]

    # (e-first-check) Hard vagueness with nothing to resolve: clarify now.
    if is_vague(question, recent):
        cq, ct = phrase_clarifying_question(question, "the question is too short to search the notes", llm=llm)
        step("clarify", reason="vague_query")
        trace["clarify"] = {"reason": "vague_query", **ct}
        return AgentResult("clarify", cq, [], trace)

    # (b/c) Retrieve, with at most one bounded retry.
    hits, supported, decomposition, rtrace = retrieve_with_fallback(
        query, mode, top_k, files, db_dir, llm=llm, trace=trace,
        page_ranges=page_ranges)
    if page_ranges:
        trace["page_filter"] = {f: f"{s}-{e}" for f, (s, e) in sorted(page_ranges.items())}
    step("retrieve", queries=len(rtrace.get("queries_tried", [])),
         retry=rtrace.get("retry", False), hits=len(hits))
    trace["queries_tried"] = rtrace.get("queries_tried", [])
    trace["retry"] = rtrace.get("retry", False)

    # Weak even after retry -> clarify once (unless the previous turn already
    # was a clarification; then answer or report instead of asking again).
    from generate import MIN_SCORE as _MS
    strong = [h for h in hits if h.get("dense_score", 0.0) >= _MS]
    if not strong:
        if prev_was_clarify:
            step("no_more_clarify", reason="previous turn was a clarification")
            _log_turn("answer",
                      "I couldn't find that in your notes. Try different words or upload notes covering it.",
                      hits, supported, decomposition)
            return AgentResult("answer",
                               "I couldn't find that in your notes. Try different words or upload notes covering it.",
                               [], trace)
        cq, ct = phrase_clarifying_question(question, "no relevant notes were found", llm=llm)
        step("clarify", reason="retrieval_failed_after_retry")
        trace["clarify"] = {"reason": "retrieval_failed_after_retry", **ct}
        _log_turn("clarify", cq, [], supported, decomposition)
        return AgentResult("clarify", cq, [], trace)

    # (d) Optional sufficiency check.
    sufficient, strace = sufficiency_check(question, strong, llm=llm)
    step("sufficiency", **strace)
    if not sufficient and not prev_was_clarify:
        # Treat as weak retrieval: fall into the clarify path (retry budget
        # already spent in retrieve_with_fallback).
        cq, ct = phrase_clarifying_question(question, "the found notes do not seem to answer it fully", llm=llm)
        step("clarify", reason="insufficient_context")
        trace["clarify"] = {"reason": "insufficient_context", **ct}
        _log_turn("clarify", cq, [], supported, decomposition)
        return AgentResult("clarify", cq, [], trace)

    # (f) Grounded answer through the existing path.
    from generate import generate_answer
    answer = generate_answer(question, hits, decomposition=decomposition, history=recent)
    step("answer")
    _log_turn("answer", answer, hits, supported, decomposition)
    return AgentResult("answer", answer, hits, trace)
