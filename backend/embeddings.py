"""
Embedding module using sentence-transformers.
Runs entirely locally on your machine — no API calls needed.
Uses the lightweight all-MiniLM-L6-v2 model (~80MB).
"""

from sentence_transformers import SentenceTransformer

# Singleton model instance
_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model (downloads on first run, cached after)."""
    global _model
    if _model is None:
        print("📐 Loading embedding model (first time may download ~80MB)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded!")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.
    
    Args:
        texts: List of strings to embed
    
    Returns:
        List of embedding vectors (each is a list of 384 floats)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 50)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Convert a single query string into an embedding vector.
    
    Args:
        query: The search query
    
    Returns:
        Embedding vector (list of 384 floats)
    """
    model = get_model()
    embedding = model.encode([query])
    return embedding[0].tolist()
