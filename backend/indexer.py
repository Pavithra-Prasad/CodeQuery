"""
Indexer module — handles repository ingestion.
Accepts local paths or GitHub URLs, walks the codebase,
chunks files, generates embeddings, and stores in ChromaDB.
"""

import os
import shutil
import tempfile
from urllib.parse import urlparse

from chunker import walk_codebase, chunk_code_file, CodeChunk
from embeddings import embed_texts
from vector_store import add_documents, delete_collection, get_collection_count


def extract_repo_name(source: str) -> str:
    """
    Extract a clean repository name from a path or URL.
    
    Examples:
        '/Users/me/projects/my-app' → 'my-app'
        'https://github.com/user/repo.git' → 'repo'
        'https://github.com/user/repo' → 'repo'
    """
    # GitHub URL
    if "github.com" in source or source.startswith("http"):
        parsed = urlparse(source)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            name = path_parts[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name
    
    # Local path
    return os.path.basename(os.path.abspath(source))


def clone_github_repo(github_url: str) -> str:
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        github_url: GitHub repository URL
    
    Returns:
        Path to the cloned repository
    """
    try:
        import git
    except ImportError:
        raise ImportError("GitPython is required. Install with: pip install gitpython")
    
    temp_dir = tempfile.mkdtemp(prefix="codebase_qa_")
    
    print(f"📥 Cloning {github_url}...")
    git.Repo.clone_from(
        github_url,
        temp_dir,
        depth=1,  # Shallow clone for speed
    )
    print(f"✅ Cloned to: {temp_dir}")
    
    return temp_dir


def index_codebase(
    source: str,
    repo_name: str = None,
    progress_callback=None,
) -> dict:
    """
    Index an entire codebase into the vector store.
    
    Args:
        source: Local directory path or GitHub URL
        repo_name: Optional custom name for the repo
        progress_callback: Optional callback(stage, current, total) for progress updates
    
    Returns:
        Dict with indexing stats: repo_name, num_files, num_chunks, status
    """
    temp_dir = None
    
    try:
        # On a public deployment, don't let visitors index files on the server
        allow_local = os.environ.get("ALLOW_LOCAL_PATHS", "true").lower() == "true"
        if not allow_local and not source.startswith(("https://", "http://")):
            return {
                "status": "error",
                "message": "Local paths are disabled on this server. Please enter a GitHub URL (https://github.com/...).",
            }

        # Determine if it's a GitHub URL or local path
        is_github = "github.com" in source or (
            source.startswith("http") and "git" in source.lower()
        )
        
        if is_github:
            temp_dir = clone_github_repo(source)
            codebase_path = temp_dir
        else:
            codebase_path = os.path.abspath(source)
            if not os.path.isdir(codebase_path):
                return {
                    "status": "error",
                    "message": f"Directory not found: {codebase_path}",
                }
        
        # Determine repo name
        if not repo_name:
            repo_name = extract_repo_name(source)
        
        if progress_callback:
            progress_callback("scanning", 0, 0)
        
        # Step 1: Walk the codebase and find all files
        print(f"📂 Scanning codebase: {codebase_path}")
        file_paths = walk_codebase(codebase_path)
        num_files = len(file_paths)
        print(f"   Found {num_files} indexable files")
        
        if num_files == 0:
            return {
                "status": "error",
                "message": "No indexable source files found in the given path.",
            }
        
        # Step 2: Chunk all files
        if progress_callback:
            progress_callback("chunking", 0, num_files)
        
        print("🧩 Chunking files...")
        all_chunks: list[CodeChunk] = []
        for i, file_path in enumerate(file_paths):
            chunks = chunk_code_file(file_path, base_path=codebase_path)
            all_chunks.extend(chunks)
            if progress_callback and i % 10 == 0:
                progress_callback("chunking", i, num_files)
        
        num_chunks = len(all_chunks)
        print(f"   Created {num_chunks} chunks from {num_files} files")
        
        if num_chunks == 0:
            return {
                "status": "error",
                "message": "Could not create any chunks from the source files.",
            }
        
        # Step 3: Delete any existing collection for this repo (re-index)
        delete_collection(repo_name)
        
        # Step 4: Generate embeddings in batches
        if progress_callback:
            progress_callback("embedding", 0, num_chunks)
        
        print("📐 Generating embeddings...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        
        # Process in batches of 64 for memory efficiency
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            batch_embeddings = embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            if progress_callback:
                progress_callback("embedding", min(i + batch_size, num_chunks), num_chunks)
        
        # Step 5: Store in ChromaDB
        if progress_callback:
            progress_callback("storing", 0, num_chunks)
        
        print("💾 Storing in vector database...")
        ids = [chunk.id for chunk in all_chunks]
        documents = chunk_texts
        metadatas = [
            {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
            }
            for chunk in all_chunks
        ]
        
        add_documents(repo_name, ids, documents, all_embeddings, metadatas)
        
        # Verify
        stored_count = get_collection_count(repo_name)
        print(f"✅ Indexed '{repo_name}': {num_files} files, {stored_count} chunks stored")
        
        if progress_callback:
            progress_callback("done", num_chunks, num_chunks)
        
        return {
            "status": "success",
            "repo_name": repo_name,
            "num_files": num_files,
            "num_chunks": stored_count,
            "message": f"Successfully indexed {num_files} files into {stored_count} chunks.",
        }
    
    except Exception as e:
        print(f"❌ Indexing error: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
    
    finally:
        # Clean up temp directory if we cloned a repo
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
