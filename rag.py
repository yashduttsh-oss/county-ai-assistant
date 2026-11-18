"""RAG retrieval + answer core for the County AI Assistant.

Loads the index built by ``ingest.py`` and answers a question with Claude,
grounded ONLY in the retrieved chunks.

Run:  python rag.py "your question"
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY is available to the Anthropic client.
load_dotenv()

# --- Configuration ----------------------------------------------------------
MODEL = "claude-sonnet-5-5"             # Claude model used to generate answers
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
INDEX_DIR = Path("index")
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
TOP_K = 4                               # default number of chunks to retrieve
MAX_TOKENS = 1024

# Exact refusal the model must use when the context lacks the answer.
REFUSAL = "I don't have that in the provided documents."

# System prompt: keep the model grounded, require bracketed filename citations,
# and force the exact refusal string above when the context is insufficient.
SYSTEM_PROMPT = (
    "You are the County AI Assistant. Answer the question using ONLY the "
    "context passages provided by the user. Cite the sources you use by their "
    "filename in square brackets, e.g. [owen_county_economy.md]. "
    "If the context does not contain the answer, reply with exactly: "
    f'"{REFUSAL}" and nothing else. '
    "Do not use any outside knowledge or guess."
)

# --- Lazy-loaded singletons -------------------------------------------------
# Loaded once on first use so that importing this module (e.g. from app.py)
# stays cheap and only pays the model-load cost when a query is actually run.
_embed_model = None
_embeddings = None   # np.ndarray, shape (n_chunks, dim)
_chunks = None       # list of {text, source, chunk_id}
_client = None


def _load_index():
    """Load embeddings.npy + chunks.json into module-level caches (once)."""
    global _embeddings, _chunks
    if _embeddings is not None:
        return
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Index not found in {INDEX_DIR.resolve()}. "
            "Run `python ingest.py` first to build it."
        )
    _embeddings = np.load(EMBEDDINGS_PATH)
    _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if len(_chunks) != _embeddings.shape[0]:
        raise ValueError(
            "Index is inconsistent (embeddings vs chunks row counts differ). "
            "Re-run `python ingest.py`."
        )


def _get_embed_model():
    """Load the sentence-transformers embedding model once."""
    global _embed_model
    if _embed_model is None:
        # Imported here so the heavy dependency only loads when needed.
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_client():
    """Create the Anthropic client once (reads ANTHROPIC_API_KEY from env)."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def retrieve(question, k=TOP_K):
    """Return the ``k`` chunks most similar to ``question``.

    The question is embedded with ``normalize_embeddings=True``; because the
    stored chunk vectors are also unit-normalized (see ingest.py), cosine
    similarity reduces to a plain dot product. Each returned item is a dict with
    ``text``, ``source``, and ``score`` (higher = more similar).
    """
    _load_index()
    model = _get_embed_model()

    # Unit-normalized query vector -> dot product == cosine similarity.
    query_vec = model.encode(question, normalize_embeddings=True).astype(np.float32)
    scores = _embeddings @ query_vec            # (n_chunks,) similarity scores
    top_idx = np.argsort(-scores)[:k]           # indices of the k highest scores

    results = []
    for i in top_idx:
        chunk = _chunks[i]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(scores[i]),
        })
    return results


def answer(question, k=TOP_K):
    """Answer ``question`` grounded in the retrieved context.

    Returns ``(answer_text, sources)`` where ``sources`` is the list of unique
    source filenames that were supplied as context to the model.
    """
    retrieved = retrieve(question, k=k)

    # Build numbered context passages, each prefixed by its source filename.
    context = "\n\n".join(
        f"[{i + 1}] {r['source']}:\n{r['text']}"
        for i, r in enumerate(retrieved)
    )
    user_message = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}"
    )

    # De-duplicate source filenames while preserving retrieval order.
    sources = list(dict.fromkeys(r["source"] for r in retrieved))

    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    return answer_text, sources


if __name__ == "__main__":
    # Take the question from the command line: python rag.py "your question"
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer_text, sources = answer(question)

    print(answer_text)
    print("\nSources:")
    for source in sources:
        print(f"  - {source}")
