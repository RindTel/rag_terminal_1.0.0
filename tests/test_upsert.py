"""
tests/test_upsert.py
────────────────────
Re-indexing a file must REPLACE its chunks, not append a second copy.

The bug: RAGPipeline.ingest_files() called VectorStore.add_chunks(), which
appends unconditionally. Re-uploading a file with the same name overwrote it in
uploads/ but stacked another full set of its chunks into the index. The live
vectorstore/ had notes.txt in it four times over.

Two of these tests exist specifically to pin down design decisions that are easy
to get subtly wrong:

  * test_reindexing_edited_file_replaces_old_content
        The upsert MUST key on source_file. Keying on chunk_id (a hash of the
        chunk's own text) looks reasonable but matches nothing once a document is
        edited — every chunk_id changes — so the old chunks survive as orphans
        and the index serves stale and fresh content simultaneously.

  * test_upsert_does_not_reembed_untouched_files
        Removal must reuse the vectors already in FAISS. Rebuilding by
        re-embedding every survivor turns a one-file upload into an
        entire-corpus re-encode.

No Ollama required.
"""

import os

import pytest

from src.rag_pipeline import RAGPipeline

DOC_A = "alpha.txt"
DOC_B = "beta.txt"

TEXT_A_V1 = (
    "FundForge stores passwords using bcrypt with twelve salt rounds. "
    "The access token expires after fifteen minutes. " * 3
)
TEXT_A_V2 = (
    "Zephyr Analytics stores passwords using argon2 with four iterations. "
    "The session cookie expires after nine hours. " * 3
)
TEXT_B = (
    "Prisma error P2002 maps to a 409 Conflict response, and P2025 maps to "
    "404 Not Found. Rate limiting allows 100 requests per 15 minutes. " * 3
)


@pytest.fixture
def ws(tmp_path):
    uploads = tmp_path / "uploads"
    store = tmp_path / "vectorstore"
    uploads.mkdir()
    store.mkdir()
    (uploads / DOC_A).write_text(TEXT_A_V1)
    (uploads / DOC_B).write_text(TEXT_B)
    return {"uploads": str(uploads), "store": str(store)}


def open_pipeline(ws) -> RAGPipeline:
    p = RAGPipeline(uploads_dir=ws["uploads"], vectorstore_dir=ws["store"])
    p.load()
    return p


def path(ws, name):
    return os.path.join(ws["uploads"], name)


def all_text(p):
    return " ".join(c["text"] for c in p.vector_store._chunks)


def test_reindexing_same_file_does_not_duplicate(ws):
    """Ingest a file twice — the chunk count must not double."""
    p = open_pipeline(ws)

    p.ingest_files([path(ws, DOC_A)])
    first = p.chunk_count()
    assert first > 0

    # Re-index the very same file.
    p.ingest_files([path(ws, DOC_A)])

    assert p.chunk_count() == first, (
        f"re-indexing duplicated chunks: {first} -> {p.chunk_count()}"
    )
    assert p.get_indexed_files() == [DOC_A]

    # And the index/metadata stay in step across a restart.
    p2 = open_pipeline(ws)
    assert p2.chunk_count() == first
    assert p2.vector_store._index.ntotal == first


def test_reindexing_many_times_stays_flat(ws):
    """Five re-indexes, still one copy. Guards against slow unbounded growth."""
    p = open_pipeline(ws)
    p.ingest_files([path(ws, DOC_A)])
    baseline = p.chunk_count()

    for _ in range(5):
        p.ingest_files([path(ws, DOC_A)])

    assert p.chunk_count() == baseline
    assert p.get_indexed_files() == [DOC_A]


def test_reindexing_edited_file_replaces_old_content(ws):
    """
    Editing a document then re-indexing must PURGE the old content.

    This is the test a chunk_id-keyed upsert fails: the edit changes every
    chunk's text, so every chunk_id changes, so nothing matches and the stale
    chunks are never removed.
    """
    p = open_pipeline(ws)
    p.ingest_files([path(ws, DOC_A)])
    assert "bcrypt" in all_text(p)

    # Edit the document in place, then re-index it.
    open(path(ws, DOC_A), "w").write(TEXT_A_V2)
    p.ingest_files([path(ws, DOC_A)])

    text = all_text(p)
    assert "argon2" in text, "new content was not indexed"
    assert "bcrypt" not in text, (
        "STALE content survived the re-index — the index now serves both the old "
        "and new versions of the document"
    )
    assert p.get_indexed_files() == [DOC_A]

    # The old facts must be genuinely unretrievable, not merely outranked.
    for chunk, _ in p.retrieve("How are passwords hashed?"):
        assert "bcrypt" not in chunk["text"]


def test_upsert_leaves_other_files_intact(ws):
    """Replacing one document must not disturb the others."""
    p = open_pipeline(ws)
    p.ingest_files([path(ws, DOC_A), path(ws, DOC_B)])
    total = p.chunk_count()
    b_chunks = sum(1 for c in p.vector_store._chunks if c["source_file"] == DOC_B)

    # Re-index only A.
    p.ingest_files([path(ws, DOC_A)])

    assert p.chunk_count() == total, "upsert of A changed the total chunk count"
    assert sorted(p.get_indexed_files()) == [DOC_A, DOC_B]
    assert sum(1 for c in p.vector_store._chunks if c["source_file"] == DOC_B) == b_chunks

    # B is still searchable — its vectors survived the rebuild intact.
    hits = p.retrieve("Which Prisma error maps to 409 Conflict?")
    assert hits
    assert any(c["source_file"] == DOC_B for c, _ in hits)


def test_upsert_does_not_reembed_untouched_files(ws):
    """
    Re-indexing one file must not re-encode the rest of the corpus.

    Rebuilding the index by re-embedding every survivor would make a single small
    upload cost an entire-corpus forward pass. The vectors are already in FAISS;
    the rebuild must reconstruct them instead.
    """
    p = open_pipeline(ws)
    p.ingest_files([path(ws, DOC_A), path(ws, DOC_B)])

    # Spy on the encoder.
    embedded = []
    original = p.vector_store.embed_texts

    def spy(texts):
        embedded.extend(texts)
        return original(texts)

    p.vector_store.embed_texts = spy

    p.ingest_files([path(ws, DOC_A)])   # re-index A only

    assert embedded, "nothing was embedded — the new chunks must still be encoded"

    # Only A's text may be re-encoded. B's vectors come from reconstruct().
    b_text = TEXT_B[:60]
    assert not any(b_text in t for t in embedded), (
        f"re-embedded untouched file {DOC_B} — removal is recomputing vectors "
        f"instead of reconstructing them ({len(embedded)} texts encoded)"
    )


def test_upsert_into_empty_index_is_a_plain_add(ws):
    """Upserting when nothing is indexed must behave like a normal add."""
    p = open_pipeline(ws)
    assert p.chunk_count() == 0

    added = p.vector_store.upsert_chunks(p.processor.process_file(path(ws, DOC_A)))
    assert added > 0
    assert p.chunk_count() == added
    assert p.get_indexed_files() == [DOC_A]
