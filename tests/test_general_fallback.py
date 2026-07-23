"""
tests/test_general_fallback.py
──────────────────────────────
The general-knowledge fallback: when the documents can't answer (empty index,
nothing retrieved, or an abstained grounded answer), and config.GENERAL_FALLBACK
is on, answer from the model's general knowledge — clearly flagged as ungrounded.
Grounded answers with sufficient context stay grounded and cited.

The LLM and retrieval are stubbed, so these are fast and need no Ollama / no
embedding model.
"""

import pytest

import config
from src.rag_pipeline import RAGPipeline

GENERAL_ANSWER = "⚠️ Not in your documents — from general knowledge: yes."
ABSTENTION = "I couldn't find a clear answer in the uploaded documents."


@pytest.fixture
def pipe(tmp_path):
    u = tmp_path / "u"; v = tmp_path / "v"; u.mkdir(); v.mkdir()
    p = RAGPipeline(uploads_dir=str(u), vectorstore_dir=str(v))
    p.llm.generate_general = lambda q, h=None: GENERAL_ANSWER
    return p


@pytest.fixture(autouse=True)
def restore_flag():
    saved = config.GENERAL_FALLBACK
    yield
    config.GENERAL_FALLBACK = saved


def _chunk(text="the plant grows in shade", src="doc.txt"):
    return {"text": text, "source_file": src, "chunk_index": 0,
            "total_chunks": 1, "page_number": None, "chunk_id": "doc_x"}


def test_is_abstention():
    assert RAGPipeline._is_abstention(ABSTENTION)
    assert RAGPipeline._is_abstention("Sorry, I COULDN'T FIND A CLEAR ANSWER here.")
    assert not RAGPipeline._is_abstention("The plant grows 23 cm tall.")


def test_empty_index_fallback_on(pipe):
    config.GENERAL_FALLBACK = True
    r = pipe.query("is the plant beneficial to humans?")
    assert r.general_fallback is True
    assert r.grounded is False
    assert r.no_retrieval is True
    assert r.sources == []
    assert r.answer == GENERAL_ANSWER


def test_empty_index_fallback_off_refuses(pipe):
    config.GENERAL_FALLBACK = False
    r = pipe.query("is the plant beneficial to humans?")
    assert r.general_fallback is False
    assert r.no_retrieval is True
    assert r.grounded is False
    assert r.answer != GENERAL_ANSWER   # the refusal message, not a general answer


def test_grounded_answer_stays_grounded(pipe, monkeypatch):
    config.GENERAL_FALLBACK = True
    monkeypatch.setattr(pipe.vector_store, "chunk_count", lambda: 1)
    monkeypatch.setattr(pipe, "retrieve", lambda q, **k: [(_chunk(), 0.9)])
    pipe.llm.generate = lambda prompt, **kw: "The plant grows 23 cm tall."
    r = pipe.query("How tall does the plant grow?")
    assert r.grounded is True
    assert r.general_fallback is False
    assert len(r.sources) == 1
    assert "23 cm" in r.answer


def test_abstention_triggers_fallback(pipe, monkeypatch):
    config.GENERAL_FALLBACK = True
    monkeypatch.setattr(pipe.vector_store, "chunk_count", lambda: 1)
    monkeypatch.setattr(pipe, "retrieve", lambda q, **k: [(_chunk(), 0.9)])
    pipe.llm.generate = lambda prompt, **kw: ABSTENTION
    r = pipe.query("is the plant beneficial to humans?")
    assert r.general_fallback is True
    assert r.grounded is False
    assert r.sources == []            # not cited — the answer isn't from the docs
    assert r.no_retrieval is False    # retrieval DID happen; it just didn't answer
    assert r.answer == GENERAL_ANSWER


def test_abstention_without_fallback_returns_refusal(pipe, monkeypatch):
    config.GENERAL_FALLBACK = False
    monkeypatch.setattr(pipe.vector_store, "chunk_count", lambda: 1)
    monkeypatch.setattr(pipe, "retrieve", lambda q, **k: [(_chunk(), 0.9)])
    pipe.llm.generate = lambda prompt, **kw: ABSTENTION
    r = pipe.query("is the plant beneficial to humans?")
    assert r.general_fallback is False
    assert r.grounded is True          # strict mode returns the grounded (abstaining) answer as-is
    assert r.answer == ABSTENTION
