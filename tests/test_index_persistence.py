"""
tests/test_index_persistence.py
───────────────────────────────
Deletion must be DURABLE.

The bug these guard against: VectorStore.save() early-returned when the index was
None, and remove_file() sets it to None once the last chunk goes. So deleting the
last document was a no-op on disk — the stale faiss.index/metadata.pkl survived
and load() read them straight back on the next launch. The document came back.

Durable deletion is the core privacy promise of a local-only document tool, so
these tests exercise the real thing: a genuine restart (a brand-new RAGPipeline
against the same directories), not just an in-memory check.

No Ollama needed: deletion and retrieval are pure FAISS + embeddings, and a query
against an empty index short-circuits to a refusal without calling the LLM.
"""

import os

import pytest

from src.rag_pipeline import RAGPipeline

DOC_A = "notes_a.txt"
DOC_B = "notes_b.txt"

TEXT_A = (
    "The Lunaria Nova plant grows only under blue LED light combined with "
    "Beethoven's Symphony No. 7. After 14 days it reached a height of 23 cm."
)
TEXT_B = (
    "FundForge is a crowdfunding platform. Prisma error P2002 maps to a "
    "409 Conflict response, and P2025 maps to 404 Not Found."
)


@pytest.fixture
def workspace(tmp_path):
    """Isolated uploads/ + vectorstore/ — never touches the real ones."""
    uploads = tmp_path / "uploads"
    store = tmp_path / "vectorstore"
    uploads.mkdir()
    store.mkdir()
    (uploads / DOC_A).write_text(TEXT_A)
    (uploads / DOC_B).write_text(TEXT_B)
    return {"uploads": str(uploads), "store": str(store), "root": tmp_path}


def open_pipeline(ws) -> RAGPipeline:
    """
    Simulate an application start: a fresh pipeline that loads whatever is on
    disk. Calling this a second time IS the restart.
    """
    p = RAGPipeline(uploads_dir=ws["uploads"], vectorstore_dir=ws["store"])
    p.load()
    return p


def index_files_on_disk(ws):
    return sorted(os.listdir(ws["store"]))


def test_deleting_last_document_is_durable(workspace):
    """Index one doc, delete it, restart — it must be gone, not resurrected."""
    ws = workspace

    # 1. Index one document.
    p = open_pipeline(ws)
    added, errors = p.ingest_files([os.path.join(ws["uploads"], DOC_A)])
    assert not errors
    assert added > 0
    assert p.get_indexed_files() == [DOC_A]
    assert "faiss.index" in index_files_on_disk(ws), "index should be persisted"

    # 2. Delete it.
    assert p.delete_file(DOC_A) is True
    assert p.chunk_count() == 0

    # The on-disk store must be gone too — this is the actual bug.
    assert "faiss.index" not in index_files_on_disk(ws), (
        "stale faiss.index survived deletion — it will resurrect on next launch"
    )
    assert "metadata.pkl" not in index_files_on_disk(ws)

    # 3. RESTART: a brand-new pipeline reading the same directories.
    p2 = open_pipeline(ws)

    # 4. The document must be gone.
    assert p2.chunk_count() == 0, "deleted document came back after restart"
    assert p2.get_indexed_files() == []
    assert p2.retrieve("How tall did the plant grow?") == []

    # And a query must never answer FROM a resurrected index. Pin strict mode
    # (GENERAL_FALLBACK off) so this stays deterministic and needs no live LLM —
    # the general-knowledge fallback is a separate mode tested elsewhere.
    import config
    saved = config.GENERAL_FALLBACK
    config.GENERAL_FALLBACK = False
    try:
        result = p2.query("How tall did the Lunaria Nova plant grow?", stream=False)
    finally:
        config.GENERAL_FALLBACK = saved
    assert result.no_retrieval is True
    assert result.grounded is False
    assert result.sources == []
    assert "23 cm" not in result.answer


def test_deleting_one_of_several_keeps_the_others(workspace):
    """The purge must not over-reach: deleting one doc must not wipe the rest."""
    ws = workspace

    p = open_pipeline(ws)
    p.ingest_files([
        os.path.join(ws["uploads"], DOC_A),
        os.path.join(ws["uploads"], DOC_B),
    ])
    assert sorted(p.get_indexed_files()) == [DOC_A, DOC_B]
    both = p.chunk_count()

    assert p.delete_file(DOC_A) is True

    # Survives a restart with exactly the other document intact.
    p2 = open_pipeline(ws)
    assert p2.get_indexed_files() == [DOC_B]
    assert 0 < p2.chunk_count() < both
    assert "faiss.index" in index_files_on_disk(ws)

    # ...and it is still actually retrievable.
    hits = p2.retrieve("Which Prisma error maps to 409 Conflict?")
    assert hits, "surviving document should still be searchable"
    assert all(c["source_file"] == DOC_B for c, _ in hits)


def test_reindex_with_empty_uploads_purges_disk(workspace):
    """
    Second leak path: reindex_all() cleared memory but returned before save(),
    so an empty uploads dir left the old index on disk.
    """
    ws = workspace

    p = open_pipeline(ws)
    p.ingest_files([os.path.join(ws["uploads"], DOC_A)])
    assert p.chunk_count() > 0

    # Remove the source file behind the pipeline's back, then reindex.
    os.remove(os.path.join(ws["uploads"], DOC_A))
    os.remove(os.path.join(ws["uploads"], DOC_B))
    added, errors = p.reindex_all()
    assert added == 0
    assert not errors

    assert index_files_on_disk(ws) == [], "reindex with no files left a stale index on disk"

    p2 = open_pipeline(ws)
    assert p2.chunk_count() == 0, "documents resurrected after an empty reindex"
    assert p2.get_indexed_files() == []


def test_corrupt_store_is_refused_not_served(workspace):
    """
    A desynced index (N vectors, M != N metadata chunks) must be refused. Serving
    it would index into a mismatched list and cite the WRONG document.
    """
    import pickle

    ws = workspace
    p = open_pipeline(ws)
    p.ingest_files([os.path.join(ws["uploads"], DOC_A)])
    assert p.chunk_count() > 0

    # Corrupt it: drop a metadata chunk without touching the FAISS index.
    meta_path = os.path.join(ws["store"], "metadata.pkl")
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks[:-1], f)

    p2 = RAGPipeline(uploads_dir=ws["uploads"], vectorstore_dir=ws["store"])
    assert p2.load() is False, "corrupt store should be refused"
    assert p2.chunk_count() == 0
