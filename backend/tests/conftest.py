"""
Shared test fixtures.

Every test runs offline and leaves no trace:
- the ChromaDB client points at a fresh temporary folder (never backend/chroma_data),
- embeddings come from a small deterministic fake (no model download),
- the Groq and Gemini API keys are removed and the Ollama startup check is skipped,
  so nothing calls a real LLM (tests that need one pass in a fake).
"""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import textwrap

import pytest
from fastapi.testclient import TestClient

import indexer
import llm
import main
import rag_engine
import vector_store
from fakes import fake_embed
from tools import repo_tools


@pytest.fixture(autouse=True)
def temp_chroma(tmp_path, monkeypatch):
    """A fresh, empty vector store for every test."""
    monkeypatch.setattr(vector_store, "CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "_client", None)
    yield


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No real embedding model and no real LLM keys."""
    monkeypatch.setattr(indexer, "embed_texts", lambda texts: [fake_embed(t) for t in texts])
    monkeypatch.setattr(rag_engine, "embed_query", fake_embed)
    monkeypatch.setattr(repo_tools, "embed_query", fake_embed)
    monkeypatch.setattr(llm, "check_ollama", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "JUDGE_MODEL", "AGENT_MAX_ITERATIONS", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALLOW_LOCAL_PATHS", "true")
    main.indexing_status.clear()
    yield
    main.indexing_status.clear()


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"))


@pytest.fixture
def sample_repo(tmp_path):
    """A small codebase: 3 indexable files plus folders and files the indexer must skip."""
    root = tmp_path / "sample-repo"
    write(root / "auth.py", """
        import hashlib


        def hash_password(password: str) -> str:
            \"\"\"Authenticate users: hash the password before storing it.\"\"\"
            return hashlib.sha256(password.encode()).hexdigest()


        def check_password(password: str, stored: str) -> bool:
            \"\"\"Authentication check for the login endpoint.\"\"\"
            return hash_password(password) == stored
        """)
    write(root / "orders.js", """
        function totalPrice(items) {
            return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
        }

        module.exports = { totalPrice };
        """)
    write(root / "README.md", "# Sample repo\n\nA shop with user login and orders.\n")
    write(root / "node_modules" / "left-pad" / "index.js", "module.exports = () => {};\n")
    write(root / ".git" / "config", "[core]\n")
    write(root / "package-lock.json", "{}\n")
    (root / "logo.png").write_bytes(b"\x89PNG fake image")
    return root
