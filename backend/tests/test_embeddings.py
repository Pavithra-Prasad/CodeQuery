"""Tests for the embeddings wrapper, with the sentence-transformers model replaced by a fake."""

import numpy as np

import embeddings


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, show_progress_bar=False):
        self.calls.append((list(texts), show_progress_bar))
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_embed_texts_returns_plain_lists(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "_model", fake)
    assert embeddings.embed_texts(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert fake.calls == [(["ab", "abcd"], False)]


def test_progress_bar_only_for_large_batches(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "_model", fake)
    embeddings.embed_texts(["x"] * 51)
    assert fake.calls[0][1] is True


def test_embed_query_returns_one_vector(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", FakeModel())
    assert embeddings.embed_query("abc") == [3.0, 1.0]


def test_model_is_loaded_once(monkeypatch):
    loads = []
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "SentenceTransformer", lambda name: loads.append(name) or FakeModel())
    assert embeddings.get_model() is embeddings.get_model()
    assert loads == ["all-MiniLM-L6-v2"]
