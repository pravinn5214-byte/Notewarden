"""Step 3 of the pipeline: retrieve the chunks most relevant to a question.

Three modes:
- "dense"         — the original behaviour: embed the question, cosine distance
                    over child chunks, top-k. Kept as the baseline.
- "hybrid"        — dense + BM25 (keyword) over the same child chunks, fused
                    with Reciprocal Rank Fusion (k=60). Pull top-20 from each.
- "hybrid_rerank" — hybrid, then a cross-encoder
                    (cross-encoder/ms-marco-MiniLM-L-6-v2) re-scores the fused
                    top-20; cut to top-k. This is the default.

Parent-child: children (~120 words) are matched; the parent chunk (~500 words)
is what gets returned as context, so the LLM sees coherent text. After
re-ranking, each parent is returned at most once (best child's score wins).

Scores in results:
- "score"   — the ranking score of the mode (cosine similarity for dense,
              RRF score for hybrid, cross-encoder score for hybrid_rerank).
- "dense_score" — cosine similarity, always present (used by the relevance gate
              and shown in the Sources panel).
- "bm25_rank" / "rerank_score" — present when the mode computed them.

Usage:
    python retrieve.py "what is normalization in databases?" --mode hybrid_rerank
"""
import argparse
import re
import sys
import threading

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ingest import COLLECTION, DB_DIR, PARENT_COLLECTION
from store import IndexCorruptError, db_path, get_collections, reset_client, with_index_recovery

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match ingest.py
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 4
FUSE_CANDIDATES = 20  # per-retriever candidates before fusion
RRF_K = 60            # standard RRF constant
MODES = ("dense", "hybrid", "hybrid_rerank")
DEFAULT_MODE = "hybrid_rerank"

_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_collection: chromadb.Collection | None = None
_parents: chromadb.Collection | None = None
_collection_key: str | None = None


def _drop_cached_collections() -> None:
    """Forget collection handles from a stale client (called by store.reset_client)."""
    global _collection, _parents, _collection_key
    _collection = None
    _parents = None
    _collection_key = None


from store import on_reset as _store_on_reset  # noqa: E402

_store_on_reset(_drop_cached_collections)

# BM25 index cache: (db_dir, frozenset(sources)) -> (bm25, chunk_texts, metas).
# Rebuilt from the Chroma contents whenever the collection or the file
# selection changes; guarded because Streamlit runs callbacks in threads.
_bm25_cache: dict = {}
_bm25_lock = threading.Lock()


def _get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_resources(db_dir: str) -> tuple[chromadb.Collection, chromadb.Collection]:
    """Lazily load the child and parent collections (shared across calls)."""
    global _collection, _parents, _collection_key
    if _collection is None or _collection_key != db_dir:
        try:
            if db_path() == db_dir:
                _collection, _parents = with_index_recovery(get_collections)
            else:
                # Explicit path (tests/eval): open directly there.
                client = chromadb.PersistentClient(path=db_dir)
                _collection = client.get_collection(COLLECTION)
                _parents = client.get_collection(PARENT_COLLECTION)
        except Exception as exc:
            raise RuntimeError(
                f"No vector store found in {db_dir}/. Run ingest.py (or 'Process "
                "documents' in the app) first."
            ) from exc
        if _collection.count() == 0:
            raise RuntimeError(
                "The vector store is empty. Run ingest.py (or 'Process documents' "
                "in the app) first."
            )
        _collection_key = db_dir
    return _collection, _parents


def _all_children(db_dir: str) -> tuple[list[str], list[str], list[dict], list[str]]:
    """Every child chunk as (id, text, meta, parent_text).

    parent_text falls back to the child's own text if the parent is missing
    (shouldn't happen, but retrieval must not crash on a half-built store).
    """
    collection, parents = _get_resources(db_dir)
    data = with_index_recovery(lambda: collection.get(include=["documents", "metadatas"]))
    ids = data["ids"]
    texts = data["documents"]
    metas = data["metadatas"]
    parent_ids = sorted({m["parent_id"] for m in metas if "parent_id" in m})
    parent_map: dict[str, str] = {}
    if parent_ids:
        pdata = with_index_recovery(lambda: parents.get(ids=parent_ids, include=["documents"]))
        parent_map = dict(zip(pdata["ids"], pdata["documents"]))
    parent_texts = [parent_map.get(m.get("parent_id"), t) for t, m in zip(texts, metas)]
    return ids, texts, metas, parent_texts


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "which", "with", "does", "do", "how", "why",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip surrounding punctuation, keep alphanumerics plus
    intra-token hyphens/underscores (so 'INV-YYYY-XXXX' and 'G-quadruplex'
    survive as single tokens), drop stopwords. Used for BOTH the BM25 index
    and queries."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _get_bm25(db_dir: str, allowed: set[str] | None) -> tuple:
    """BM25 index over child chunks, cached per (db_dir, selection)."""
    key = (db_dir, frozenset(allowed) if allowed else None)
    with _bm25_lock:
        cached = _bm25_cache.get(key)
        if cached is not None:
            return cached
    ids, texts, metas, parent_texts = _all_children(db_dir)
    if allowed is not None:
        keep = [i for i, m in enumerate(metas) if m["source"] in allowed]
        ids = [ids[i] for i in keep]
        texts = [texts[i] for i in keep]
        metas = [metas[i] for i in keep]
        parent_texts = [parent_texts[i] for i in keep]
    bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None
    with _bm25_lock:
        _bm25_cache.clear()  # one selection at a time is enough
        _bm25_cache[key] = (ids, texts, metas, parent_texts, bm25)
    return _bm25_cache[key]


def invalidate_bm25_cache() -> None:
    """Drop cached BM25 indexes (call after ingesting/deleting files)."""
    with _bm25_lock:
        _bm25_cache.clear()


def _dense_ranked(question: str, k: int, db_dir: str, allowed: set[str] | None) -> list[dict]:
    """Dense retrieval: cosine similarity over child chunks."""
    collection, _ = _get_resources(db_dir)
    where = {"source": {"$in": sorted(allowed)}} if allowed else None
    n = min(k * 5, with_index_recovery(collection.count))
    query_vec = _get_embedder().encode([question]).tolist()

    def _query() -> dict:
        col, _ = _get_resources(db_dir)  # re-resolve after a client reset
        return col.query(query_embeddings=query_vec, n_results=max(1, n), where=where)

    res = with_index_recovery(lambda: collection.query(
        query_embeddings=query_vec, n_results=max(1, n), where=where), refetch=_query)
    out = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({
            "id": cid,
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "parent_id": meta.get("parent_id", cid),
            "ocr": bool(meta.get("ocr", False)),
            "ocr_engine": meta.get("ocr_engine", ""),
            "dense_score": 1 - dist,
        })
    return out


def _bm25_ranked(question: str, k: int, db_dir: str, allowed: set[str] | None) -> list[dict]:
    """BM25 keyword retrieval over child chunks."""
    ids, texts, metas, parent_texts, bm25 = _get_bm25(db_dir, allowed)
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(question))
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "id": ids[i],
            "text": texts[i],
            "source": metas[i]["source"],
            "page": metas[i]["page"],
            "parent_id": metas[i].get("parent_id", ids[i]),
            "parent_text": parent_texts[i],
            "ocr": bool(metas[i].get("ocr", False)),
            "ocr_engine": metas[i].get("ocr_engine", ""),
            "bm25_rank": rank,
        }
        for rank, i in enumerate(order, start=1)
        if scores[i] > 0
    ]


def _rrf_fuse(dense: list[dict], bm25_hits: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank)) across lists."""
    fused: dict[str, dict] = {}
    for rank, hit in enumerate(dense, start=1):
        h = dict(hit)
        h["rrf"] = 1 / (RRF_K + rank)
        fused[hit["id"]] = h
    for rank, hit in enumerate(bm25_hits, start=1):
        contrib = 1 / (RRF_K + rank)
        if hit["id"] in fused:
            fused[hit["id"]]["rrf"] += contrib
            if "bm25_rank" not in fused[hit["id"]]:
                fused[hit["id"]]["bm25_rank"] = hit["bm25_rank"]
            fused[hit["id"]].setdefault("parent_text", hit["parent_text"])
        else:
            h = dict(hit)
            h["rrf"] = contrib
            h.setdefault("dense_score", 0.0)
            fused[hit["id"]] = h
    return sorted(fused.values(), key=lambda h: h["rrf"], reverse=True)


def _fill_parent_text(hits: list[dict], parents: chromadb.Collection, db_dir: str) -> None:
    """Attach parent_text to hits that don't have it yet."""
    _, parent_col = _get_resources(db_dir)
    missing = [h for h in hits if "parent_text" not in h]
    if not missing:
        return
    wanted = sorted({h["parent_id"] for h in missing})
    pdata = with_index_recovery(lambda: parent_col.get(ids=wanted, include=["documents"]))
    pmap = dict(zip(pdata["ids"], pdata["documents"]))
    for h in missing:
        h["parent_text"] = pmap.get(h["parent_id"], h["text"])


def _dedupe_parents(hits: list[dict]) -> list[dict]:
    """Keep the best hit per parent_id (input must already be sorted by score)."""
    seen: set[str] = set()
    out = []
    for h in hits:
        pid = h.get("parent_id", h["id"])
        if pid in seen:
            continue
        seen.add(pid)
        out.append(h)
    return out


def retrieve(
    question: str,
    k: int = TOP_K,
    db_dir: str = DB_DIR,
    mode: str = DEFAULT_MODE,
    allowed_sources: list[str] | None = None,
    return_parents: bool = True,
) -> list[dict]:
    """Return the top-k chunks as dicts: text, source, page, score (+ extras).

    mode: "dense", "hybrid" or "hybrid_rerank" (see module docstring).
    allowed_sources: file names to restrict retrieval to (per-file selection);
        None means everything.
    return_parents: if True (default), hybrid/hybrid_rerank return the parent
        chunk's text as "text" (that's what the LLM should read); dense always
        returns the child text itself. Either way "page"/"source" come from
        the matched child.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Use one of {MODES}.")
    allowed = set(allowed_sources) if allowed_sources else None

    if mode == "dense":
        hits = _dense_ranked(question, k, db_dir, allowed)
        for h in hits:
            h["score"] = h["dense_score"]
        return hits[:k]

    # hybrid / hybrid_rerank: fuse dense and BM25 candidate lists.
    dense = _dense_ranked(question, FUSE_CANDIDATES, db_dir, allowed)
    bm25_hits = _bm25_ranked(question, FUSE_CANDIDATES, db_dir, allowed)
    fused = _rrf_fuse(dense, bm25_hits)[:FUSE_CANDIDATES]
    if not fused:
        return []

    if mode == "hybrid":
        _fill_parent_text(fused, _parents, db_dir)
        fused = _dedupe_parents(fused)
        for h in fused:
            h["score"] = h["rrf"]
        return fused[:k]

    # hybrid_rerank: cross-encoder over the fused candidates (child text —
    # what was actually matched — not the long parent, which dilutes the score).
    _fill_parent_text(fused, _parents, db_dir)
    pairs = [(question, h["text"]) for h in fused]
    ce_scores = _get_reranker().predict(pairs)
    for h, s in zip(fused, ce_scores):
        h["rerank_score"] = float(s)
    fused.sort(key=lambda h: h["rerank_score"], reverse=True)
    fused = _dedupe_parents(fused)
    for h in fused:
        h["score"] = h["rerank_score"]
    top = fused[:k]
    if return_parents:
        for h in top:
            h["text"] = h["parent_text"]
    return top


def best_dense_score(question: str, db_dir: str = DB_DIR, allowed_sources: list[str] | None = None) -> float:
    """Cosine similarity of the best dense match — the mode-independent
    relevance signal used by the gate in generate.py."""
    hits = _dense_ranked(question, 1, db_dir, set(allowed_sources) if allowed_sources else None)
    return hits[0]["dense_score"] if hits else 0.0


def _merge_rank_key(h: dict) -> float:
    """Best available ranking score for merging (higher is better)."""
    if "rerank_score" in h:
        return h["rerank_score"]
    if "rrf" in h:
        return h["rrf"]
    return h.get("dense_score", h.get("score", 0.0))


def retrieve_multi(
    subqueries: list[str],
    mode: str = DEFAULT_MODE,
    top_k: int = TOP_K,
    db_dir: str = DB_DIR,
    files: list[str] | None = None,
    return_parents: bool = True,
) -> tuple[list[dict], set[int]]:
    """Retrieve evidence for each sub-query separately, then merge.

    Per sub-query: the existing single-query pipeline (same mode/file filter),
    then the mode-independent relevance gate (dense cosine >= MIN_SCORE).
    A sub-query with no hit above the gate is unsupported (its index is left
    out of the returned `supported` set) instead of failing the question.

    Merge with guaranteed slots: each supported sub-query first keeps its best
    min(2, top_k) distinct parents; remaining slots (budget
    min(top_k * supported, DECOMPOSE_MAX_CONTEXT)) go to the best remaining
    hits by rank score. Dedupe by parent across sub-queries; a parent returned
    for several sub-queries is kept once with its best score and every
    matching subquery_index.

    Returns (merged_hits, supported_subquery_indices).
    """
    from generate import MIN_SCORE  # local import: generate imports retrieve's siblings

    import os as _os
    max_context = int(_os.environ.get("DECOMPOSE_MAX_CONTEXT", "8"))

    allowed = set(files) if files else None
    if len(subqueries) == 1:
        # Exactly the old single-query path: same results, same order, no
        # gate (relevance gating for a single query stays in generate.py, as
        # before), and no parent dedupe (it would collapse dense children
        # sharing a parent).
        hits = retrieve(subqueries[0], k=top_k, db_dir=db_dir, mode=mode,
                        allowed_sources=list(allowed) if allowed else None,
                        return_parents=return_parents)
        for h in hits:
            h["subquery_index"] = 0
            h["subquery_indices"] = [0]
        return hits, ({0} if hits else set())

    per_sub: list[list[dict]] = []
    for i, q in enumerate(subqueries):
        hits = retrieve(q, k=top_k, db_dir=db_dir, mode=mode,
                        allowed_sources=list(allowed) if allowed else None,
                        return_parents=return_parents)
        # Gate per sub-query: mode-independent dense cosine.
        kept = [h for h in hits if h.get("dense_score", 0.0) >= MIN_SCORE]
        for h in kept:
            h["subquery_index"] = i
        per_sub.append(kept)

    supported = {i for i, hits in enumerate(per_sub) if hits}
    unsupported_count = len(subqueries) - len(supported)
    if unsupported_count:
        import logging
        logging.getLogger(__name__).info(
            "retrieve_multi: %d of %d sub-queries unsupported", unsupported_count, len(subqueries))
    if not supported:
        return [], supported

    # Dedupe by parent across sub-queries, keeping the best score and all
    # matching subquery_index values.
    by_parent: dict[str, dict] = {}
    for i, hits in enumerate(per_sub):
        for h in hits:
            pid = h.get("parent_id", h["id"])
            existing = by_parent.get(pid)
            if existing is None:
                merged = dict(h)
                merged["subquery_index"] = i
                merged["subquery_indices"] = {i}
                by_parent[pid] = merged
            else:
                existing["subquery_indices"].add(i)
                if _merge_rank_key(h) > _merge_rank_key(existing):
                    better = dict(h)
                    better["subquery_index"] = existing["subquery_index"]
                    better["subquery_indices"] = existing["subquery_indices"]
                    by_parent[pid] = better
    for m in by_parent.values():
        m["subquery_indices"] = sorted(m["subquery_indices"])

    all_hits = sorted(by_parent.values(), key=_merge_rank_key, reverse=True)

    budget = min(top_k * len(supported), max_context)
    slot_count = min(2, top_k)

    # Guaranteed slots: best distinct parents of each supported sub-query
    # first, up to slot_count each, preferring hits only that sub-query found.
    guaranteed: list[dict] = []
    used_parents: set[str] = set()
    remaining_slots = budget
    for i in sorted(supported):
        if remaining_slots <= 0:
            break
        sub_hits = [h for h in all_hits if i in h["subquery_indices"]]
        own_first = [h for h in sub_hits if len(h["subquery_indices"]) == 1] + \
                    [h for h in sub_hits if len(h["subquery_indices"]) > 1]
        taken = 0
        for h in own_first:
            if taken >= slot_count or remaining_slots <= 0:
                break
            pid = h.get("parent_id", h["id"])
            if pid in used_parents:
                continue
            guaranteed.append(h)
            used_parents.add(pid)
            remaining_slots -= 1
            taken += 1
        if len(guaranteed) >= budget:
            break

    # Remaining slots: best remaining hits across all sub-queries.
    merged = list(guaranteed)
    if remaining_slots > 0:
        for h in all_hits:
            if remaining_slots <= 0:
                break
            pid = h.get("parent_id", h["id"])
            if pid in used_parents:
                continue
            merged.append(h)
            used_parents.add(pid)
            remaining_slots -= 1

    # Order: best score first (stable, matches the single-query shape).
    merged.sort(key=lambda h: (-_merge_rank_key(h), min(h["subquery_indices"])))
    return merged[:budget], supported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve chunks for a question")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()
    for hit in retrieve(" ".join(args.question), k=args.k, mode=args.mode):
        print(f"\n[{hit['source']} p.{hit['page']}  score={hit['score']:.2f}]")
        print(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))