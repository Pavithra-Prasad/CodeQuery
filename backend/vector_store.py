"""
Vector store module using ChromaDB.
Stores code chunk embeddings locally on disk for persistent storage.
"""

import os
import chromadb
from chromadb.config import Settings


# Persistent storage directory
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_data")

# Singleton client
_client = None


def get_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client with persistent storage."""
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        print(f"💾 ChromaDB initialized at: {CHROMA_DIR}")
    return _client


def get_or_create_collection(repo_name: str) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for a repository.
    Each repo gets its own collection for isolation.
    
    Args:
        repo_name: Name of the repository (used as collection name)
    
    Returns:
        ChromaDB Collection object
    """
    client = get_client()
    # Sanitize collection name (ChromaDB has naming rules)
    safe_name = sanitize_collection_name(repo_name)
    return client.get_or_create_collection(
        name=safe_name,
        metadata={"repo_name": repo_name},
    )


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize a repository name to be a valid ChromaDB collection name.
    Rules: 3-63 chars, alphanumeric + underscores/hyphens, starts/ends with alphanumeric.
    """
    import re
    # Replace non-alphanumeric chars with underscores
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Ensure it starts and ends with alphanumeric
    safe = safe.strip("_-")
    if not safe:
        safe = "default_repo"
    # Ensure min length 3
    while len(safe) < 3:
        safe += "_x"
    # Truncate to max 63
    safe = safe[:63].rstrip("_-")
    return safe


def add_documents(
    repo_name: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """
    Add document chunks with embeddings to the vector store.
    
    Args:
        repo_name: Name of the repository
        ids: Unique IDs for each chunk
        documents: Text content of each chunk
        embeddings: Embedding vectors for each chunk
        metadatas: Metadata dicts for each chunk (file_path, language, etc.)
    """
    collection = get_or_create_collection(repo_name)
    
    # ChromaDB has a batch size limit, so we add in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )


def query_documents(
    repo_name: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> dict:
    """
    Query the vector store for the most similar chunks.
    
    Args:
        repo_name: Name of the repository to search
        query_embedding: Embedding vector of the query
        top_k: Number of results to return
    
    Returns:
        Dict with keys: 'ids', 'documents', 'metadatas', 'distances'
    """
    collection = get_or_create_collection(repo_name)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
    }


def delete_collection(repo_name: str) -> None:
    """Delete a repository's collection from the vector store."""
    client = get_client()
    safe_name = sanitize_collection_name(repo_name)
    try:
        client.delete_collection(name=safe_name)
    except Exception:
        pass  # Collection might not exist


def list_collections() -> list[str]:
    """List all indexed repository names."""
    client = get_client()
    collections = client.list_collections()
    return [c.metadata.get("repo_name", c.name) for c in collections]


def get_collection_count(repo_name: str) -> int:
    """Get the number of chunks indexed for a repository."""
    try:
        collection = get_or_create_collection(repo_name)
        return collection.count()
    except Exception:
        return 0
