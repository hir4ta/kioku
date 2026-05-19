"""Integration: ingest a small vault → ``hybrid_search`` returns sane hits.

Voyage is stubbed (no API key, no network). The point of this test is
the full ``BM25 + dense → RRF → composite score`` plumbing, including
the metadata lookup that powers recency / importance / pinned / etc.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import voyageai
from kioku.chunk import chunk_body
from kioku.config import load_settings
from kioku.embed import Embedder, content_hash
from kioku.retrieve import hybrid_search
from kioku.store_sqlite import (
    ChunkInsert,
    cache_get,
    cache_put,
    connect,
    init_schema,
    insert_chunks,
    upsert_memory,
)
from kioku.vault import scaffold, walk


def _write_decision(vault: Path, slug: str, body: str) -> None:
    """Write a frontmatter-valid decision file.

    Built by string concatenation rather than ``textwrap.dedent`` because
    the body parameter is interpolated without leading whitespace,
    which breaks ``dedent``'s common-prefix detection.
    """
    path = vault / "semantic" / "decisions" / "active" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "---\n"
        f"id: DEC-2026-05-19-{slug}\n"
        "type: decision\n"
        "status: active\n"
        "created_at: '2026-05-19T10:00:00+00:00'\n"
        "event_at: '2026-05-19T10:00:00+00:00'\n"
        "source: user-notes\n"
        "trust: high\n"
        "importance: 0.5\n"
        "---\n"
        "\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def stub_voyage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    class _Resp:
        def __init__(self, embeddings: list[list[float]]) -> None:
            self.embeddings = embeddings

    client = MagicMock()

    def _embed(texts: list[str], **_kwargs: object) -> _Resp:
        return _Resp([[float(abs(hash(t)) % 1000) / 1000.0] * 1024 for t in texts])

    client.embed.side_effect = _embed
    monkeypatch.setattr(voyageai, "Client", lambda **_kwargs: client)
    return client


def _ingest_vault(tmp_path: Path) -> Path:
    """Ingest a fixed 3-decision vault and return the SQLite path.

    The Voyage stub is installed by the ``stub_voyage`` fixture in
    callers; this helper just exercises the ingest pipeline.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    scaffold(vault, write_claude_md=False)
    _write_decision(
        vault, "voyage", "# Use voyage-4-large\n\nDecided to use Voyage for embeddings."
    )
    _write_decision(vault, "sqlite", "# Use sqlite-vec\n\nLocal hybrid index lives in sqlite.")
    _write_decision(vault, "obsidian", "# Use Obsidian\n\nVault is the source of truth.")

    db_path = tmp_path / "kioku.sqlite"
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
    return db_path


def test_hybrid_search_returns_ranked_hits(
    tmp_path: Path, stub_voyage: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(tmp_path / "vault"))
    db_path = _ingest_vault(tmp_path)
    settings = load_settings()

    with connect(db_path) as conn:
        embedder = Embedder(
            api_key="x",
            cache_get=lambda h, m: cache_get(conn, h, m),
            cache_put=lambda h, m, v: cache_put(conn, h, m, v),
        )
        # Stub: rerank returns descending scores by input order so the call
        # path exercises but doesn't alter relative ordering.
        rerank_response = MagicMock()
        rerank_response.results = [
            MagicMock(index=i, relevance_score=1.0 - i * 0.1) for i in range(5)
        ]
        stub_voyage.rerank.return_value = rerank_response

        hits = hybrid_search(conn, "voyage", settings=settings, embedder=embedder, top_k=5)

    assert hits, "hybrid_search returned no hits"
    # The "voyage" decision should be the top hit.
    assert hits[0].memory_id == "DEC-2026-05-19-voyage"
    # Composite + RRF scores are populated.
    for hit in hits:
        assert hit.composite_score >= 0.0
        assert hit.rrf_score >= 0.0
        assert hit.vault_path  # non-empty path on every hit


def test_hybrid_search_bm25_only_when_no_embedder(
    tmp_path: Path, stub_voyage: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(tmp_path / "vault"))
    db_path = _ingest_vault(tmp_path)
    settings = load_settings()

    with connect(db_path) as conn:
        hits = hybrid_search(conn, "obsidian", settings=settings, embedder=None, top_k=3)

    assert hits, "BM25-only retrieval should still return hits"
    # No rerank ran, so rerank_score is None on every hit.
    for hit in hits:
        assert hit.rerank_score is None


def test_hybrid_search_empty_query_returns_empty(
    tmp_path: Path, stub_voyage: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(tmp_path / "vault"))
    db_path = _ingest_vault(tmp_path)
    settings = load_settings()

    with connect(db_path) as conn:
        assert hybrid_search(conn, "", settings=settings, embedder=None) == []
        assert hybrid_search(conn, "   ", settings=settings, embedder=None) == []
