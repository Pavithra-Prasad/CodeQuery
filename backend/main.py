"""
FastAPI server — the main entry point for the Codebase Q&A Assistant.
Serves the API endpoints and the static frontend.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncio
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import llm
from agent import run_agent
from config import get_settings
from indexer import index_codebase, extract_repo_name
from rag_engine import ask_question
from vector_store import list_collections, get_collection_count, delete_collection


# ── Request/Response Models ────────────────────────────────────

class IndexRequest(BaseModel):
    source: str  # Local path or GitHub URL
    repo_name: Optional[str] = None  # Optional custom name


class QueryRequest(BaseModel):
    question: str
    repo_name: str
    top_k: int = Field(5, ge=1, le=50)  # Number of code chunks to retrieve
    mode: Literal["rag", "agent"] = "rag"  # rag: one retrieval + one LLM call; agent: tool-calling loop


class IndexResponse(BaseModel):
    status: str
    repo_name: str = ""
    num_files: int = 0
    num_chunks: int = 0
    message: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    mode: str = "rag"
    trace: list[dict] = []  # agent mode: every tool call with its arguments, result size and latency
    iterations: int = 0
    tool_calls: int = 0


class RepoInfo(BaseModel):
    name: str
    num_chunks: int


# ── Indexing state (in-memory, for progress tracking) ──────────

indexing_status = {}


# ── App Setup ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    print("🚀 Codebase Q&A Assistant starting up...")
    port = os.environ.get('PORT', '8080')
    print(f"   Open http://localhost:{port} in your browser")
    settings = get_settings()
    try:
        llm.check_ollama()
        print(f"✅ Ollama is ready ({settings.llm_model})")
    except llm.OllamaNotReady as e:
        # Keep serving (the UI and cloud mode still work); questions will return this message
        needed = "Agent mode needs" if settings.llm_provider == "cloud" else "Answers need"
        print(f"⚠️  {needed} Ollama: {e}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Codebase Q&A Assistant",
    description="RAG-powered natural language Q&A over any codebase",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Routes ─────────────────────────────────────────────────

@app.post("/api/index", response_model=IndexResponse)
async def api_index_repo(request: IndexRequest):
    """Index a codebase from a local path or GitHub URL."""
    source = request.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Source path or URL is required")
    
    repo_name = request.repo_name or extract_repo_name(source)
    
    # Check if already indexing
    if indexing_status.get(repo_name, {}).get("status") == "in_progress":
        return IndexResponse(
            status="in_progress",
            repo_name=repo_name,
            message="This repository is already being indexed.",
        )
    
    # Track progress
    indexing_status[repo_name] = {"status": "in_progress", "stage": "starting", "current": 0, "total": 0}
    
    def progress_callback(stage, current, total):
        indexing_status[repo_name] = {
            "status": "in_progress",
            "stage": stage,
            "current": current,
            "total": total,
        }
    
    # Run indexing in a thread to not block the event loop
    result = await asyncio.to_thread(
        index_codebase, source, repo_name, progress_callback
    )
    
    # Update status
    indexing_status[repo_name] = {
        "status": result["status"],
        "stage": "done" if result["status"] == "success" else "error",
        "current": result.get("num_chunks", 0),
        "total": result.get("num_chunks", 0),
    }
    
    return IndexResponse(**result)


@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    """Ask a question about an indexed codebase."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    
    if not request.repo_name.strip():
        raise HTTPException(status_code=400, detail="Repository name is required")
    
    try:
        if request.mode == "agent":
            # The agent makes several blocking LLM calls; run it in a thread to keep the server responsive
            result = await asyncio.to_thread(run_agent, request.question, request.repo_name)
            return QueryResponse(mode="agent", **{k: v for k, v in result.to_dict().items()
                                                  if k in ("answer", "sources", "trace", "iterations", "tool_calls")})
        result = await ask_question(
            question=request.question,
            repo_name=request.repo_name,
            top_k=request.top_k,
        )
        return QueryResponse(**result)
    except llm.OllamaNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/api/repos")
async def api_list_repos():
    """List all indexed repositories."""
    repos = list_collections()
    return [
        {"name": name, "num_chunks": get_collection_count(name)}
        for name in repos
    ]


@app.get("/api/index-status/{repo_name}")
async def api_index_status(repo_name: str):
    """Get the indexing status for a repository."""
    status = indexing_status.get(repo_name, {"status": "not_started"})
    return status


@app.delete("/api/repos/{repo_name}")
async def api_delete_repo(repo_name: str):
    """Delete an indexed repository."""
    delete_collection(repo_name)
    indexing_status.pop(repo_name, None)
    return {"status": "deleted", "repo_name": repo_name}


# ── Static Frontend Serving ───────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the main frontend page."""
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
