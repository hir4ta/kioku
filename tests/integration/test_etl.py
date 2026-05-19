"""End-to-end integration test: vault → SQLite full rebuild.

Voyage is stubbed (no API key, no network). The point of this test is
to prove the wiring across :mod:`kioku.vault`, :mod:`kioku.chunk`,
:mod:`kioku.embed`, and :mod:`kioku.store_sqlite` actually composes: a
freshly scaffolded vault with one Markdown file produces a populated
SQLite store, and both BM25 and vector search return results.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import voyageai
from kioku.chunk import chunk_body
from kioku.embed import Embedder, content_hash
from kioku.store_sqlite import (
    ChunkInsert,
    cache_get,
    cache_put,
    connect,
    init_schema,
    insert_chunks,
    mark_accessed,
    search_bm25,
    search_vec,
    stats,
    upsert_memory,
)
from kioku.vault import scaffold, walk


@pytest.fixture
def stub_voyage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``voyageai.Client`` with a stub that returns deterministic vectors."""

    class _Resp:
        def __init__(self, embeddings: list[list[float]]) -> None:
            self.embeddings = embeddings

    client = MagicMock()

    def _embed(texts: list[str], **_kwargs: object) -> _Resp:
        return _Resp([[float(abs(hash(t)) % 1000) / 1000.0] * 1024 for t in texts])

    client.embed.side_effect = _embed
    monkeypatch.setattr(voyageai, "Client", lambda **_kwargs: client)
    return client


def test_full_rebuild_round_trip(
    tmp_path: Path, sample_memory_text: str, stub_voyage: MagicMock
) -> None:
    # ----- Arrange: a vault with one decision file. -----
    vault = tmp_path / "vault"
    vault.mkdir()
    scaffold(vault, write_claude_md=False)
    file_path = vault / "semantic" / "decisions" / "active" / "voyage.md"
    file_path.write_text(sample_memory_text, encoding="utf-8")

    db_path = tmp_path / "kioku.sqlite"

    # ----- Act: run the ETL pipeline by hand (same code paths as `kioku rebuild`). -----
    with connect(db_path) as conn:
        init_schema(conn)
        embedder = Embedder(
            api_key="x",
            cache_get=lambda h, m: cache_get(conn, h, m),
            cache_put=lambda h, m, v: cache_put(conn, h, m, v),
        )

        ingested = 0
        for record in walk(vault):
            pieces = chunk_body(record.body)
            results = embedder.embed(
                [p.body for p in pieces], kind="general", input_type="document"
            )
            upsert_memory(conn, record, content_hash=content_hash(record.body))
            insert_chunks(
                conn,
                [
                    ChunkInsert(
                        memory_id=record.id,
                        parent_in_batch=p.parent_index,
                        section=p.section,
                        body=p.body,
                        token_count=p.token_count,
                        heading_path=p.heading_path,
                        embedding_general=r.vector,
                    )
                    for p, r in zip(pieces, results, strict=True)
                ],
            )
            ingested += 1

        # ----- Assert: stats reflect the ingest. -----
        s = stats(conn)
        assert ingested == 1
        assert s.memory_count == 1
        assert s.chunk_count >= 2  # at least toc + body
        assert s.embedding_cache_count >= 2
        assert s.by_type.get("decision") == 1

        # ----- BM25 finds the body. -----
        hits = search_bm25(conn, "voyage", limit=5)
        assert hits, "expected BM25 to find chunks mentioning 'voyage'"
        assert hits[0].memory_id == "DEC-2026-05-19-test-decision"

        # ----- Vector search returns something. -----
        any_vec = [0.5] * 1024
        vhits = search_vec(conn, any_vec, limit=3)
        assert vhits, "expected vector search to return at least one hit"

        # ----- Access logging bumps the counter. -----
        mark_accessed(
            conn,
            "DEC-2026-05-19-test-decision",
            session_id="test-session",
            injection_type="auto",
        )
        row = conn.execute(
            "SELECT access_count FROM memories WHERE id = ?",
            ("DEC-2026-05-19-test-decision",),
        ).fetchone()
        assert int(row["access_count"]) == 1


def test_rebuild_idempotent_with_cache(
    tmp_path: Path, sample_memory_text: str, stub_voyage: MagicMock
) -> None:
    """Re-running rebuild on an unchanged vault should hit the cache."""
    vault = tmp_path / "vault"
    vault.mkdir()
    scaffold(vault, write_claude_md=False)
    (vault / "semantic" / "decisions" / "active" / "voyage.md").write_text(
        sample_memory_text, encoding="utf-8"
    )
    db_path = tmp_path / "kioku.sqlite"

    def _run_rebuild() -> int:
        """Returns the number of Voyage embed calls made during the run."""
        before = stub_voyage.embed.call_count
        with connect(db_path) as conn:
            init_schema(conn)
            embedder = Embedder(
                api_key="x",
                cache_get=lambda h, m: cache_get(conn, h, m),
                cache_put=lambda h, m, v: cache_put(conn, h, m, v),
            )
            for record in walk(vault):
                pieces = chunk_body(record.body)
                results = embedder.embed(
                    [p.body for p in pieces], kind="general", input_type="document"
                )
                upsert_memory(conn, record, content_hash=content_hash(record.body))
                insert_chunks(
                    conn,
                    [
                        ChunkInsert(
                            memory_id=record.id,
                            parent_in_batch=p.parent_index,
                            section=p.section,
                            body=p.body,
                            token_count=p.token_count,
                            heading_path=p.heading_path,
                            embedding_general=r.vector,
                        )
                        for p, r in zip(pieces, results, strict=True)
                    ],
                )
        return stub_voyage.embed.call_count - before

    first_calls = _run_rebuild()
    second_calls = _run_rebuild()
    assert first_calls > 0
    # All embeddings were cached from the first run.
    assert second_calls == 0
