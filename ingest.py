"""Ingest pipeline for the County AI Assistant.

Builds a local search index from vetted documents in ``docs/``:

  1. Read every ``.md`` / ``.txt`` file.
  2. Scrub PII (emails + phone numbers) before anything else touches the text.
  3. Split each document into overlapping ~800-character chunks.
  4. Embed each chunk with sentence-transformers (all-MiniLM-L6-v2).
  5. Save two row-aligned files to ``index/``: ``embeddings.npy`` and
     ``chunks.json``.

Run with:  python ingest.py
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ----------------------------------------------------------
DOCS_DIR = Path("docs")          # where vetted source documents live
INDEX_DIR = Path("index")        # where the regenerable index is written
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast sentence-embedding model
CHUNK_SIZE = 800                 # target characters per chunk
CHUNK_OVERLAP = 150              # characters shared between adjacent chunks
SOURCE_EXTENSIONS = {".md", ".txt"}

# --- PII scrubbing patterns -------------------------------------------------
# Email addresses, e.g. "jane.doe@example.org".
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Phone numbers: US-style with optional country code, separators, and
# parentheses, e.g. "+1 (812) 555-0137", "812-555-0137", "812.555.0137".
PHONE_RE = re.compile(
    r"(?<!\w)"                       # not preceded by a word char
    r"(?:\+?\d{1,2}[\s.-]?)?"        # optional country code
    r"(?:\(\d{3}\)|\d{3})"           # area code, with or without parens
    r"[\s.-]?\d{3}[\s.-]?\d{4}"      # exchange + line number
    r"(?!\w)"                        # not followed by a word char
)
REDACTION = "[redacted]"


def scrub_pii(text):
    """Replace emails and phone numbers with ``[redacted]``.

    Returns the cleaned text and the number of redactions made, so the caller
    can report per-file counts.
    """
    text, n_emails = EMAIL_RE.subn(REDACTION, text)
    text, n_phones = PHONE_RE.subn(REDACTION, text)
    return text, n_emails + n_phones


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split ``text`` into overlapping character windows.

    Each window is ``chunk_size`` characters; consecutive windows share
    ``overlap`` characters so context is not lost at chunk boundaries. Empty or
    whitespace-only windows are skipped.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks = []
    step = chunk_size - overlap  # how far the window advances each iteration
    start = 0
    while start < len(text):
        window = text[start:start + chunk_size]
        if window.strip():       # ignore all-whitespace windows
            chunks.append(window)
        start += step
    return chunks


def main():
    # --- Step 1: locate source documents ------------------------------------
    source_files = sorted(
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS
    )
    if not source_files:
        raise SystemExit(
            f"No .md or .txt files found in {DOCS_DIR.resolve()}. "
            "Add vetted documents and re-run."
        )

    # Accumulators kept row-aligned: records[i] describes embeddings row i.
    records = []  # list of {text, source, chunk_id}

    # --- Steps 2 & 3: scrub PII, then chunk each document -------------------
    for path in source_files:
        raw = path.read_text(encoding="utf-8")

        cleaned, n_redactions = scrub_pii(raw)
        print(f"{path.name}: {n_redactions} redaction(s)")

        for chunk_id, chunk in enumerate(chunk_text(cleaned)):
            records.append({
                "text": chunk,
                "source": path.name,
                "chunk_id": chunk_id,
            })

    if not records:
        raise SystemExit("Documents contained no chunkable text. Nothing to index.")

    # --- Step 4: embed every chunk -----------------------------------------
    # normalize_embeddings=True yields unit vectors, so cosine similarity at
    # query time reduces to a plain dot product.
    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunk(s)...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)  # compact, standard dtype for the vector store

    # --- Step 5: save the aligned index ------------------------------------
    INDEX_DIR.mkdir(exist_ok=True)
    embeddings_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.json"

    np.save(embeddings_path, embeddings)
    chunks_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Step 6: print a summary -------------------------------------------
    print("\n--- Index summary ---")
    print(f"Documents indexed : {len(source_files)}")
    print(f"Chunks created    : {len(records)}")
    print(f"Embedding dim     : {embeddings.shape[1]}")
    print(f"Saved to          : {INDEX_DIR.resolve()}")
    print(f"  - {embeddings_path.name} (shape {embeddings.shape})")
    print(f"  - {chunks_path.name}")


if __name__ == "__main__":
    main()
