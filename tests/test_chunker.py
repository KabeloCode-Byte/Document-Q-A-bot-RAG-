import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from docqa.chunker import chunk_text
from docqa.rag import build_prompt


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_long_text_is_split_within_size():
    text = ("This is a sentence about lending. " * 100).strip()
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_overlap_and_cover_text():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(words, size=300, overlap=60)
    assert chunks[0].startswith("w0")
    assert chunks[-1].endswith("w399")
    # consecutive chunks share some text
    assert chunks[1][:20] in chunks[0] or chunks[0][-30:].split()[-1] in chunks[1]


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_prompt_contains_numbered_sources_and_question():
    passages = [{"text": "Interest is capped.", "source": "act.pdf", "page": 4}]
    prompt = build_prompt("What is capped?", passages)
    assert "[1] (act.pdf, page 4)" in prompt
    assert "Interest is capped." in prompt
    assert prompt.endswith("Question: What is capped?")
