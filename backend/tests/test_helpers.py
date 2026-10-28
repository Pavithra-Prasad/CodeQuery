"""Unit tests for the small input helpers: repo names and ChromaDB collection names."""

import pytest

from indexer import extract_repo_name
from vector_store import sanitize_collection_name


@pytest.mark.parametrize("source, expected", [
    ("https://github.com/user/repo", "repo"),
    ("https://github.com/user/repo.git", "repo"),
    ("https://github.com/user/repo/", "repo"),
    ("http://gitlab.example.com/team/service.git", "service"),
    ("/home/me/projects/my-app", "my-app"),
    ("/home/me/projects/my-app/", "my-app"),
])
def test_extract_repo_name(source, expected):
    assert extract_repo_name(source) == expected


@pytest.mark.parametrize("name", ["my-repo", "repo_2", "CodeQuery", "a" * 63])
def test_valid_names_are_kept_as_they_are(name):
    assert sanitize_collection_name(name) == name


@pytest.mark.parametrize("name, prefix", [
    ("my repo!", "my_repo_"),
    ("__repo__", "repo_"),
    ("-repo-", "repo_"),
    ("ab", "ab_"),
    ("", "repo_"),
    ("!!!", "repo_"),
])
def test_other_names_are_cleaned_and_get_a_short_hash(name, prefix):
    safe = sanitize_collection_name(name)
    assert safe.startswith(prefix)
    assert len(safe) == len(prefix) + 8
    assert sanitize_collection_name(name) == safe        # same name, same collection


def test_sanitize_collection_name_caps_length_at_63():
    safe = sanitize_collection_name("a" * 100)
    assert len(safe) == 63


@pytest.mark.parametrize("name", ["x", "a b c", "repo.with.dots", "z" * 80, "-_-ok-_-"])
def test_sanitized_names_follow_chromadb_rules(name):
    safe = sanitize_collection_name(name)
    assert 3 <= len(safe) <= 63
    assert safe[0].isalnum() and safe[-1].isalnum()


def test_different_repo_names_get_different_collections():
    assert sanitize_collection_name("my.repo") != sanitize_collection_name("my_repo")
