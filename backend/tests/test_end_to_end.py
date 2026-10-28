"""Regression test for the whole flow through the API: index -> list -> ask -> delete.

Only the embedding model and the LLM are faked; chunking, ChromaDB and the API are real.
"""

import rag_engine


def test_index_ask_and_delete_a_repo(client, sample_repo, monkeypatch):
    prompts = []
    monkeypatch.setattr(rag_engine, "_call_llm",
                        lambda prompt: prompts.append(prompt) or "`check_password` in auth.py hashes and compares.")

    # 1. Index the sample repo
    res = client.post("/api/index", json={"source": str(sample_repo), "repo_name": "shop"})
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["num_files"] == 3

    # 2. It shows up in the repo list
    repos = client.get("/api/repos").json()
    assert [r["name"] for r in repos] == ["shop"]
    assert repos[0]["num_chunks"] == res.json()["num_chunks"]

    # 3. A question about authentication cites auth.py first
    res = client.post("/api/query", json={"question": "How does password authentication work?",
                                          "repo_name": "shop", "top_k": 2})
    assert res.status_code == 200
    body = res.json()
    assert "check_password" in body["answer"]
    assert body["sources"][0]["file_path"] == "auth.py"
    assert len(body["sources"]) == 2
    assert "def check_password" in prompts[0]

    # 4. Deleting it removes it from the list
    client.delete("/api/repos/shop")
    assert client.get("/api/repos").json() == []
