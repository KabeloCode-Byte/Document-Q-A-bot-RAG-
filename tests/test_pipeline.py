"""Integration tests using a tiny offline embedder, so no model download is needed."""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from docqa import config, rag, store
from docqa.loader import Page

DIM = 128


class HashEmbedder(chromadb.EmbeddingFunction):
    """Bag-of-words hashing embedder: deterministic and good enough for tests."""

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "hash-embedder"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedder()

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = [0.0] * DIM
            for word in re.findall(r"[a-z]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(store, "embedding_function", HashEmbedder())


PAGES = [
    Page("lending.txt", 1, "Interest rates on credit agreements are capped by regulation."),
    Page("lending.txt", 2, "A written notice is required before legal action on arrears."),
    Page("campus.txt", 1, "The library is open from eight until ten during exam season."),
]


def test_index_and_search_returns_relevant_page_first():
    assert store.index_pages(PAGES) == 3
    hits = store.search("when does the library open", top_k=2)
    assert hits[0]["source"] == "campus.txt"
    assert hits[0]["page"] == 1


def test_reindexing_replaces_instead_of_duplicating():
    store.index_pages(PAGES)
    store.index_pages(PAGES)
    assert store.count() == 3


def test_answer_without_documents_explains_how_to_ingest():
    result = rag.answer("anything at all")
    assert result["sources"] == []
    assert "ingest" in result["answer"].lower()


def test_answer_without_api_key_falls_back_to_passages():
    store.index_pages(PAGES)
    result = rag.answer("what is required before legal action", top_k=2)
    assert "ANTHROPIC_API_KEY" in result["answer"]
    assert result["sources"][0]["source"] == "lending.txt"


def test_api_health_and_ask():
    from docqa.api import app

    store.index_pages(PAGES)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 3}

    ok = client.post("/ask", json={"question": "are interest rates capped?"})
    assert ok.status_code == 200
    assert ok.json()["sources"]

    bad = client.post("/ask", json={"question": "hi"})  # too short
    assert bad.status_code == 422

    assert client.get("/").status_code == 200
