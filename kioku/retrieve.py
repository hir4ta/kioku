"""Hybrid retrieval — BM25 + dense + Voyage rerank + composite score.

Pipeline:

1. **BM25** via FTS5 (:func:`kioku.store_sqlite.search_bm25`) returns up
   to ``fetch_pool_size`` candidates ordered by BM25 score.
2. **Dense** via ``sqlite-vec`` (:func:`kioku.store_sqlite.search_vec`)
   after embedding the query with Voyage (``input_type="query"``).
3. **Reciprocal Rank Fusion** (RRF, ``k=60`` per Cormack et al. 2009)
   merges the two ranked lists into a single relevance score per chunk.
4. **Voyage rerank-2.5** re-orders the fused top-``fetch_pool_size``
   against the query, producing a fine-grained relevance signal.
5. **Composite score** (:func:`kioku.rank.score`) applies recency,
   importance, access frequency, project match, and the multiplicative
   modifiers (pinned / deprecated / trust=low).
6. Top-``default_top_k`` is returned.

The pipeline degrades gracefully:

* No ``embedder`` → BM25-only, no rerank, RRF skipped (relevance
  comes from normalized BM25 scores).
* ``enable_rerank=False`` or rerank API failure → composite score uses
  RRF only; the candidate body is still returned, just less precisely
  ordered within the top hits.
* Empty query → empty result (no Voyage round-trip).
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass

from kioku.config import KiokuSettings
from kioku.embed import Embedder
from kioku.errors import EmbedError
from kioku.rank import ScoreInputs, fallback_event_at, score
from kioku.store_sqlite import (
    SearchHit,
    search_bm25,
    search_vec,
)

log = logging.getLogger("kioku.retrieve")

RRF_K = 60  # standard tuning from Cormack et al. 2009


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RankedHit:
    """A retrieval result with all the signals downstream consumers need."""

    chunk_id: int
    memory_id: str
    body: str
    section: str
    rrf_score: float
    rerank_score: float | None
    composite_score: float
    vault_path: str


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def _rrf_fuse(
    rankings: list[list[SearchHit]],
    *,
    k: int = RRF_K,
) -> dict[int, float]:
    """Reciprocal Rank Fusion. Returns ``chunk_id → fused score``."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Memory metadata loader
# ---------------------------------------------------------------------------


def _load_memory_metadata(
    conn: sqlite3.Connection,
    memory_ids: list[str],
) -> dict[str, sqlite3.Row]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        SELECT id, project, importance, access_count, event_at,
               pinned, deprecated_by, trust, vault_path
        FROM memories
        WHERE id IN ({placeholders})
        """,
        memory_ids,
    ).fetchall()
    return {str(row["id"]): row for row in rows}


def _parse_iso(s: str) -> dt.datetime:
    if not s:
        return fallback_event_at()
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return fallback_event_at()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hybrid_search(  # noqa: PLR0912 — pipeline orchestration, splitting would obscure flow
    conn: sqlite3.Connection,
    query: str,
    *,
    settings: KiokuSettings,
    project: str | None = None,
    embedder: Embedder | None = None,
    enable_rerank: bool = True,
    pool_size: int | None = None,
    top_k: int | None = None,
) -> list[RankedHit]:
    """Run the full hybrid pipeline.

    ``embedder=None`` triggers a BM25-only path (no Voyage call), useful
    for tests and for users who haven't supplied an API key yet.
    """
    if not query.strip():
        return []

    pool = pool_size if pool_size is not None else settings.inject.fetch_pool_size
    k = top_k if top_k is not None else settings.inject.default_top_k

    # ----- 1. BM25 -----
    bm25_hits = search_bm25(conn, query, limit=pool, project=project)

    # ----- 2. Dense -----
    dense_hits: list[SearchHit] = []
    if embedder is not None:
        try:
            results = embedder.embed([query], kind="general", input_type="query")
            if results:
                dense_hits = search_vec(conn, results[0].vector, limit=pool, project=project)
        except EmbedError as exc:
            log.warning("dense retrieval skipped: %s", exc)

    # ----- 3. RRF -----
    if dense_hits:
        rrf_scores = _rrf_fuse([bm25_hits, dense_hits])
    elif bm25_hits:
        # Normalize BM25 score (already negated to be larger-is-better) to [0, 1].
        max_score = max(h.score for h in bm25_hits) or 1.0
        rrf_scores = {h.chunk_id: max(0.0, h.score / max_score) for h in bm25_hits}
    else:
        return []

    # ----- 4. Collect candidates -----
    candidates_by_id: dict[int, SearchHit] = {h.chunk_id: h for h in bm25_hits}
    for h in dense_hits:
        candidates_by_id.setdefault(h.chunk_id, h)

    top_fused_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )[:pool]
    candidates = [candidates_by_id[cid] for cid in top_fused_ids if cid in candidates_by_id]

    if not candidates:
        return []

    # ----- 5. Voyage rerank (optional) -----
    rerank_scores: dict[int, float] = {}
    if enable_rerank and embedder is not None:
        try:
            rr = embedder.rerank(
                query,
                [c.body for c in candidates],
                model=settings.voyage.model_rerank,
                top_k=min(k * 2, len(candidates)),
            )
            for rr_result in rr:
                if 0 <= rr_result.index < len(candidates):
                    rerank_scores[candidates[rr_result.index].chunk_id] = rr_result.score
        except EmbedError as exc:
            log.warning("rerank skipped: %s", exc)

    # ----- 6. Memory metadata for composite score -----
    memory_meta = _load_memory_metadata(conn, [c.memory_id for c in candidates])

    # ----- 7. Composite score -----
    now = dt.datetime.now(dt.UTC)
    ranked: list[RankedHit] = []
    for c in candidates:
        meta = memory_meta.get(c.memory_id)
        if meta is None:
            # Memory row missing — chunk references a stale memory_id.
            # Skip rather than crash; the rebuild will repair this.
            log.debug("skipping chunk %d: memory row missing", c.chunk_id)
            continue

        # Prefer rerank score over RRF for the relevance axis when available;
        # rerank-2.5 is the dedicated quality signal.
        relevance = rerank_scores.get(c.chunk_id, rrf_scores.get(c.chunk_id, 0.0))

        inputs = ScoreInputs(
            relevance=relevance,
            event_at=_parse_iso(str(meta["event_at"] or "")),
            importance=float(meta["importance"] or 0.5),
            access_count=int(meta["access_count"] or 0),
            project_match=(project is not None and meta["project"] == project),
            pinned=bool(meta["pinned"]),
            deprecated=bool(meta["deprecated_by"]),
            trust_low=(str(meta["trust"]) == "low"),
        )
        composite = score(inputs, scoring=settings.scoring, now=now)

        ranked.append(
            RankedHit(
                chunk_id=c.chunk_id,
                memory_id=c.memory_id,
                body=c.body,
                section=c.section,
                rrf_score=rrf_scores.get(c.chunk_id, 0.0),
                rerank_score=rerank_scores.get(c.chunk_id),
                composite_score=composite,
                vault_path=str(meta["vault_path"]),
            )
        )

    ranked.sort(key=lambda h: h.composite_score, reverse=True)
    return ranked[:k]
