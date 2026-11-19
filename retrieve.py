"""Step 3 of the pipeline: retrieve the chunks most relevant to a question.

Usage:
    python retrieve.py "what is normalization in databases?"
"""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "notes"
EMBED_MODEL = "all-MiniLM-L6-v2"  # must match ingest.py
TOP_K = 4

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_collection_key: str | None = None


def _get_resources(db_dir: str = DB_DIR) -> tuple[SentenceTransformer, chromadb.Collection]:
    """Lazily load the embedding model and collection (shared across calls)."""
    global _model, _collection, _collection_key
    if _model is None or _collection is None or _collection_key != db_dir:
        try:
            _collection = chromadb.PersistentClient(path=db_dir).get_collection(COLLECTION)
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
        _model = SentenceTransformer(EMBED_MODEL)
        _collection_key = db_dir
    return _model, _collection


def retrieve(question: str, k: int = TOP_K, db_dir: str = DB_DIR) -> list[dict]:
    """Return the top-k chunks as dicts: text, source, page, score (1 = identical)."""
    model, collection = _get_resources(db_dir)
    query_vec = model.encode([question]).tolist()
    res = collection.query(query_embeddings=query_vec, n_results=min(k, collection.count()))
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "score": 1 - dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python retrieve.py "your question"')
    for hit in retrieve(" ".join(sys.argv[1:])):
        print(f"\n[{hit['source']} p.{hit['page']}  score={hit['score']:.2f}]")
        print(hit["text"][:400] + ("..." if len(hit["text"]) > 400 else ""))
