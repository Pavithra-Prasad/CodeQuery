"""API tests for every FastAPI endpoint, using FastAPI's TestClient."""

import pytest

import main


# ── POST /api/index ─────────────────────────────────────────────

@pytest.fixture
def fake_indexer(monkeypatch):
    calls = []

    def fake(source, repo_name, progress_callback):
        calls.append((source, repo_name))
        progress_callback("chunking", 1, 2)
        return {"status": "success", "repo_name": repo_name, "num_files": 2, "num_chunks": 7,
                "message": "Successfully indexed 2 files into 7 chunks."}

    monkeypatch.setattr(main, "index_codebase", fake)
    return calls


def test_index_success(client, fake_indexer):
    res = client.post("/api/index", json={"source": "https://github.com/someone/shop.git"})
    assert res.status_code == 200
    assert res.json() == {"status": "success", "repo_name": "shop", "num_files": 2, "num_chunks": 7,
                          "message": "Successfully indexed 2 files into 7 chunks."}
    assert fake_indexer == [("https://github.com/someone/shop.git", "shop")]


def test_index_uses_a_custom_repo_name(client, fake_indexer):
    client.post("/api/index", json={"source": "/code/shop", "repo_name": "my-shop"})
    assert fake_indexer == [("/code/shop", "my-shop")]


def test_index_status_after_success(client, fake_indexer):
    client.post("/api/index", json={"source": "/code/shop"})
    assert client.get("/api/index-status/shop").json() == {"status": "success", "stage": "done", "current": 7, "total": 7}


@pytest.mark.parametrize("source", ["", "   "])
def test_index_rejects_an_empty_source(client, fake_indexer, source):
    res = client.post("/api/index", json={"source": source})
    assert res.status_code == 400
    assert res.json()["detail"] == "Source path or URL is required"
    assert fake_indexer == []


def test_index_requires_a_source_field(client):
    res = client.post("/api/index", json={})
    assert res.status_code == 422
    assert res.json()["detail"][0]["loc"] == ["body", "source"]


def test_index_does_not_start_twice(client, fake_indexer):
    main.indexing_status["shop"] = {"status": "in_progress"}
    res = client.post("/api/index", json={"source": "/code/shop"})
    assert res.json()["status"] == "in_progress"
    assert "already being indexed" in res.json()["message"]
    assert fake_indexer == []


def test_index_errors_are_returned_and_tracked(client, monkeypatch):
    monkeypatch.setattr(main, "index_codebase",
                        lambda s, r, cb: {"status": "error", "message": "Directory not found: /nope"})
    res = client.post("/api/index", json={"source": "/nope"})
    assert res.json()["status"] == "error"
    assert res.json()["message"] == "Directory not found: /nope"
    assert client.get("/api/index-status/nope").json()["stage"] == "error"


def test_index_status_of_an_unknown_repo(client):
    assert client.get("/api/index-status/never-seen").json() == {"status": "not_started"}


# ── POST /api/query ─────────────────────────────────────────────

@pytest.fixture
def fake_ask(monkeypatch):
    calls = []

    async def fake(question, repo_name, top_k):
        calls.append((question, repo_name, top_k))
        return {"answer": "It hashes the password.",
                "sources": [{"file_path": "auth.py", "start_line": 1, "end_line": 9}]}

    monkeypatch.setattr(main, "ask_question", fake)
    return calls


def test_query_success(client, fake_ask):
    res = client.post("/api/query", json={"question": "How does login work?", "repo_name": "shop"})
    assert res.status_code == 200
    assert res.json()["answer"] == "It hashes the password."
    assert res.json()["sources"][0]["file_path"] == "auth.py"
    assert fake_ask == [("How does login work?", "shop", 5)]       # top_k defaults to 5


@pytest.mark.parametrize("body, detail", [
    ({"question": "  ", "repo_name": "shop"}, "Question is required"),
    ({"question": "Why?", "repo_name": ""}, "Repository name is required"),
])
def test_query_rejects_blank_fields(client, fake_ask, body, detail):
    res = client.post("/api/query", json=body)
    assert res.status_code == 400
    assert res.json()["detail"] == detail
    assert fake_ask == []


@pytest.mark.parametrize("body", [
    {"repo_name": "shop"},
    {"question": "Why?"},
    {"question": "Why?", "repo_name": "shop", "top_k": "lots"},
])
def test_query_validates_the_request_body(client, fake_ask, body):
    assert client.post("/api/query", json=body).status_code == 422


def test_query_value_errors_become_400(client, monkeypatch):
    async def fail(**kwargs):
        raise ValueError("No API key set!")
    monkeypatch.setattr(main, "ask_question", fail)
    res = client.post("/api/query", json={"question": "Why?", "repo_name": "shop"})
    assert res.status_code == 400
    assert res.json()["detail"] == "No API key set!"


def test_query_unexpected_errors_become_500(client, monkeypatch):
    async def fail(**kwargs):
        raise RuntimeError("LLM timeout")
    monkeypatch.setattr(main, "ask_question", fail)
    res = client.post("/api/query", json={"question": "Why?", "repo_name": "shop"})
    assert res.status_code == 500
    assert res.json()["detail"] == "Error generating answer: LLM timeout"


@pytest.mark.parametrize("top_k", [0, -3, 51])
def test_query_rejects_invalid_top_k(client, top_k):
    res = client.post("/api/query", json={"question": "Why?", "repo_name": "shop", "top_k": top_k})
    assert res.status_code == 422


def test_query_on_an_unindexed_repo_gives_a_friendly_answer(client):
    res = client.post("/api/query", json={"question": "Why?", "repo_name": "never-indexed"})
    assert res.status_code == 200
    assert "couldn't find any relevant code" in res.json()["answer"]
    assert client.get("/api/repos").json() == []          # asking didn't create an empty repo


# ── GET /api/repos and DELETE /api/repos/{name} ─────────────────

def test_list_repos(client, monkeypatch):
    monkeypatch.setattr(main, "list_collections", lambda: ["shop", "blog"])
    monkeypatch.setattr(main, "get_collection_count", {"shop": 12, "blog": 3}.get)
    assert client.get("/api/repos").json() == [{"name": "shop", "num_chunks": 12}, {"name": "blog", "num_chunks": 3}]


def test_list_repos_when_empty(client):
    assert client.get("/api/repos").json() == []


def test_delete_repo(client, monkeypatch):
    deleted = []
    monkeypatch.setattr(main, "delete_collection", deleted.append)
    main.indexing_status["shop"] = {"status": "success"}

    res = client.delete("/api/repos/shop")

    assert res.json() == {"status": "deleted", "repo_name": "shop"}
    assert deleted == ["shop"]
    assert "shop" not in main.indexing_status


# ── Frontend ────────────────────────────────────────────────────

def test_home_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "<title>CodeQuery AI" in res.text


def test_static_files_are_served(client):
    res = client.get("/static/script.js")
    assert res.status_code == 200
    assert "handleAskQuestion" in res.text
