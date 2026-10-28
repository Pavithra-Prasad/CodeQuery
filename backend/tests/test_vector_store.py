"""Tests for the ChromaDB wrapper, run against a real ChromaDB in a temporary folder."""

import pytest

import vector_store as vs
from fakes import fake_embed


def add(repo: str, texts: list[str]) -> None:
    vs.add_documents(
        repo,
        ids=[f"doc{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=[fake_embed(t) for t in texts],
        metadatas=[{"file_path": f"file{i}.py"} for i in range(len(texts))],
    )


def test_add_and_count_documents():
    add("demo", ["login and password check", "shopping cart total"])
    assert vs.get_collection_count("demo") == 2


def test_adding_more_than_one_batch_stores_everything():
    add("big", [f"chunk number {i}" for i in range(250)])
    assert vs.get_collection_count("big") == 250


def test_query_returns_the_closest_document_first():
    add("demo", ["login and password check", "shopping cart total", "send email notification"])
    results = vs.query_documents("demo", fake_embed("password login"), top_k=2)
    assert results["documents"][0] == "login and password check"
    assert len(results["documents"]) == 2
    assert results["metadatas"][0]["file_path"] == "file0.py"
    assert results["distances"][0] <= results["distances"][1]


def test_top_k_larger_than_the_collection_returns_everything():
    add("demo", ["one", "two"])
    assert len(vs.query_documents("demo", fake_embed("one"), top_k=10)["documents"]) == 2


def test_list_collections_shows_the_original_repo_names():
    add("my repo!", ["x"])
    add("other-repo", ["y"])
    assert sorted(vs.list_collections()) == ["my repo!", "other-repo"]


def test_delete_collection_removes_the_repo():
    add("demo", ["x"])
    vs.delete_collection("demo")
    assert vs.list_collections() == []


def test_deleting_an_unknown_repo_does_not_raise():
    vs.delete_collection("never-indexed")


def test_client_is_created_once():
    assert vs.get_client() is vs.get_client()


def test_looking_up_an_unknown_repo_does_not_create_it():
    vs.get_collection_count("typo-repo")
    assert "typo-repo" not in vs.list_collections()
