# app/rag/embedder.py
# Purpose: unified embedding interface
# Swap between Ollama (local) and sentence-transformers (cloud)
# via a single environment variable — no agent code changes needed.
#
# Interview point: this is the adapter pattern.
# All callers use embed() without knowing the provider.
# Switching from local to cloud = one environment variable change.

import os
import numpy as np

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")


def embed(text: str) -> list[float]:
    """
    Unified embedding function.
    Local:  Ollama nomic-embed-text (768-dim, free, offline)
    Cloud:  sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU, no API key)
    """
    if PROVIDER == "ollama":
        return _embed_ollama(text)
    elif PROVIDER == "sentence-transformers":
        return _embed_st(text)
    else:
        raise ValueError(f"Unknown embedding provider: {PROVIDER}. Use 'ollama' or 'sentence-transformers'.")


def _embed_ollama(text: str) -> list[float]:
    """Local embedding via Ollama. Requires Ollama running."""
    import ollama
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


# sentence-transformers model loaded once at module level
# Interview point: lazy loading — only loads when first called,
# not at import time. Avoids slow startup if Ollama is being used.
_st_model = None


def _embed_st(text: str) -> list[float]:
    """Cloud-friendly embedding via sentence-transformers. CPU-only, no API key."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Embedder] Loading sentence-transformers model...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedder] Model ready.")
    embedding = _st_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch embedding for indexing efficiency.
    Interview point: batching avoids per-call overhead.
    For 309 documents, batching is 3-5x faster than calling embed() in a loop.
    """
    if PROVIDER == "sentence-transformers":
        global _st_model
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = _st_model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [e.tolist() for e in embeddings]
    else:
        # Ollama does not support batch — fall back to loop
        return [embed(t) for t in texts]


if __name__ == "__main__":
    # quick test
    result = embed("AC bus from Hyderabad to Bangalore")
    print(f"Provider: {PROVIDER}")
    print(f"Embedding dimensions: {len(result)}")
    print(f"First 5 values: {result[:5]}")
