"""Tests for indexing a codebase into the vector store (embeddings faked, ChromaDB real)."""

import os
import shutil

import pytest

import indexer
import vector_store as vs


def test_indexes_a_local_folder(sample_repo):
    stages = []
    result = indexer.index_codebase(str(sample_repo), "shop", lambda stage, cur, total: stages.append(stage))

    assert result["status"] == "success"
    assert result["repo_name"] == "shop"
    assert result["num_files"] == 3                      # auth.py, orders.js, README.md
    assert result["num_chunks"] == vs.get_collection_count("shop") > 0
    assert stages[0] == "scanning" and stages[-1] == "done"
    assert {"chunking", "embedding", "storing"} <= set(stages)


def test_stored_chunks_keep_their_file_metadata(sample_repo):
    indexer.index_codebase(str(sample_repo), "shop")
    stored = vs.get_or_create_collection("shop").get(include=["metadatas"])["metadatas"]
    assert {m["file_path"] for m in stored} == {"auth.py", "orders.js", "README.md"}
    assert all(m["start_line"] >= 1 and m["end_line"] >= m["start_line"] for m in stored)


def test_repo_name_defaults_to_the_folder_name(sample_repo):
    assert indexer.index_codebase(str(sample_repo))["repo_name"] == "sample-repo"


def test_reindexing_replaces_the_old_chunks(sample_repo):
    first = indexer.index_codebase(str(sample_repo), "shop")["num_chunks"]
    second = indexer.index_codebase(str(sample_repo), "shop")["num_chunks"]
    assert second == first == vs.get_collection_count("shop")


def test_missing_folder_is_an_error(tmp_path):
    result = indexer.index_codebase(str(tmp_path / "nope"), "x")
    assert result["status"] == "error"
    assert "Directory not found" in result["message"]


def test_folder_without_source_files_is_an_error(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    result = indexer.index_codebase(str(tmp_path), "x")
    assert result == {"status": "error", "message": "No indexable source files found in the given path."}


def test_files_that_produce_no_chunks_are_an_error(tmp_path):
    (tmp_path / "blank.py").write_text("   \n")
    result = indexer.index_codebase(str(tmp_path), "x")
    assert result["message"] == "Could not create any chunks from the source files."


def test_local_paths_can_be_disabled_on_public_servers(sample_repo, monkeypatch):
    monkeypatch.setenv("ALLOW_LOCAL_PATHS", "false")
    result = indexer.index_codebase(str(sample_repo), "x")
    assert result["status"] == "error"
    assert "Local paths are disabled" in result["message"]


def test_embedding_failure_is_reported_not_raised(sample_repo, monkeypatch):
    def broken(texts):
        raise RuntimeError("model not available")
    monkeypatch.setattr(indexer, "embed_texts", broken)
    assert indexer.index_codebase(str(sample_repo), "x") == {"status": "error", "message": "model not available"}


def test_github_urls_are_cloned_indexed_and_cleaned_up(sample_repo, tmp_path, monkeypatch):
    clone_dir = tmp_path / "clone"

    def fake_clone(url):
        shutil.copytree(sample_repo, clone_dir)
        return str(clone_dir)

    monkeypatch.setattr(indexer, "clone_github_repo", fake_clone)
    result = indexer.index_codebase("https://github.com/someone/shop.git")

    assert result["status"] == "success"
    assert result["repo_name"] == "shop"
    assert not clone_dir.exists()                         # temp clone deleted afterwards


def test_github_urls_work_even_when_local_paths_are_disabled(sample_repo, tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_LOCAL_PATHS", "false")
    monkeypatch.setattr(indexer, "clone_github_repo",
                        lambda url: shutil.copytree(sample_repo, tmp_path / "clone") and str(tmp_path / "clone"))
    assert indexer.index_codebase("https://github.com/someone/shop")["status"] == "success"


def test_clone_github_repo_makes_a_shallow_clone(monkeypatch):
    import git
    calls = {}
    monkeypatch.setattr(git.Repo, "clone_from", staticmethod(lambda url, path, depth: calls.update(url=url, path=path, depth=depth)))

    path = indexer.clone_github_repo("https://github.com/someone/shop")
    try:
        assert calls["url"] == "https://github.com/someone/shop"
        assert calls["path"] == path and calls["depth"] == 1
        assert os.path.basename(path).startswith("codebase_qa_")
    finally:
        shutil.rmtree(path, ignore_errors=True)
