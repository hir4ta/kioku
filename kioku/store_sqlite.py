"""SQLite store — derived L3 index over the vault.

Schema:

* ``memories``: one row per Markdown file. Frontmatter columns are
  promoted to first-class columns for fast filtering.
* ``chunks``: one row per chunk inside a memory. ``parent_id`` preserves
  the document hierarchy from :mod:`lib.chunk`.
* ``chunks_fts``: FTS5 virtual table over the chunk body, BM25 ranking.
* ``chunks_vec_general``: ``sqlite-vec`` ``vec0`` virtual table holding
  the general embedding (voyage-4-large or whatever is configured).
  ``rowid`` matches ``chunks.id``.
* ``chunks_vec_code``: same, for ``voyage-code-3`` embeddings (optional).
* ``access_log``: every injection / retrieval hit. Drives the access
  frequency axis in :mod:`lib.rank`.
* ``conflicts``: pairwise conflict records detected by the cron.
* ``embedding_cache``: ``(content_hash, model) → vector`` cache, so
  ``kioku rebuild`` does not re-pay Voyage costs on every run.

The store is intentionally a *derived* index: any layer can be dropped
and rebuilt from the Markdown vault. Schema migrations therefore favor
``DROP TABLE`` + ``kioku rebuild`` over delicate ``ALTER`` choreography.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import sqlite3
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sqlite_vec

from kioku.errors import StoreError
from kioku.vault import MemoryRecord

log = logging.getLogger("kioku.store")

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection with ``sqlite-vec`` loaded and FK enforcement on.

    Commits on clean exit, rolls back on exception, always closes.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def init_schema(
    conn: sqlite3.Connection,
    *,
    dim_general: int = 1024,
    dim_code: int = 1024,
) -> None:
    """Create every table and index, idempotent across runs."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memories (
            id                TEXT PRIMARY KEY,
            type              TEXT NOT NULL,
            vault_path        TEXT NOT NULL UNIQUE,
            status            TEXT NOT NULL,
            source            TEXT NOT NULL,
            trust             TEXT NOT NULL,
            project           TEXT,
            importance        REAL NOT NULL DEFAULT 0.5,
            event_at          TEXT NOT NULL,
            ingestion_at      TEXT NOT NULL,
            last_accessed     TEXT,
            access_count      INTEGER NOT NULL DEFAULT 0,
            supersedes        TEXT REFERENCES memories(id) ON DELETE SET NULL,
            deprecated_by     TEXT REFERENCES memories(id) ON DELETE SET NULL,
            pinned            INTEGER NOT NULL DEFAULT 0,
            content_hash      TEXT NOT NULL,
            tags_json         TEXT NOT NULL DEFAULT '[]',
            frontmatter_json  TEXT NOT NULL,
            body              TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_type    ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
        CREATE INDEX IF NOT EXISTS idx_memories_status  ON memories(status);

        CREATE TABLE IF NOT EXISTS chunks (
            id            INTEGER PRIMARY KEY,
            memory_id     TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            parent_id     INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
            section       TEXT NOT NULL,
            body          TEXT NOT NULL,
            token_count   INTEGER NOT NULL,
            heading_path  TEXT NOT NULL DEFAULT '[]'
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_memory ON chunks(memory_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            body,
            section,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS access_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id         TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            session_id        TEXT,
            injected_at       TEXT NOT NULL,
            injection_type    TEXT NOT NULL,
            used_in_response  INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_access_memory ON access_log(memory_id);

        CREATE TABLE IF NOT EXISTS conflicts (
            id                 TEXT PRIMARY KEY,
            memory_a           TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            memory_b           TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            similarity         REAL NOT NULL,
            predicate_opposed  INTEGER,
            detected_at        TEXT NOT NULL,
            resolution         TEXT NOT NULL,
            resolved_at        TEXT,
            resolved_by        TEXT,
            explanation        TEXT
        );

        CREATE TABLE IF NOT EXISTS embedding_cache (
            content_hash  TEXT NOT NULL,
            model         TEXT NOT NULL,
            vector_blob   BLOB NOT NULL,
            dim           INTEGER NOT NULL,
            created_at    TEXT NOT NULL,
            PRIMARY KEY (content_hash, model)
        );
        """
    )

    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_general USING vec0(
            embedding FLOAT[{dim_general}]
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec_code USING vec0(
            embedding FLOAT[{dim_code}]
        )
        """
    )

    # Keep the FTS5 mirror and vec tables in lock-step with ``chunks``.
    # Without this trigger, cascade-deletes from ``memories → chunks`` leave
    # orphan rows in the vec tables, which then collide on the next INSERT
    # (rowid UNIQUE constraint). FTS5 contentless tables also need explicit
    # cleanup; we ride the same trigger to avoid drift between mirrors.
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_after_delete
        AFTER DELETE ON chunks
        BEGIN
            DELETE FROM chunks_fts          WHERE rowid = OLD.id;
            DELETE FROM chunks_vec_general  WHERE rowid = OLD.id;
            DELETE FROM chunks_vec_code     WHERE rowid = OLD.id;
        END;
        """
    )

    conn.executemany(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        [
            ("schema_version", str(SCHEMA_VERSION)),
            ("dim_general", str(dim_general)),
            ("dim_code", str(dim_code)),
        ],
    )


# ---------------------------------------------------------------------------
# Vector pack / unpack
# ---------------------------------------------------------------------------


def pack_vector(vec: list[float]) -> bytes:
    """Pack a vector into the binary format ``sqlite-vec`` expects."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`."""
    if len(blob) % 4 != 0:
        raise StoreError(f"vector blob length {len(blob)} is not a multiple of 4")
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Memory upsert / delete
# ---------------------------------------------------------------------------


def upsert_memory(
    conn: sqlite3.Connection,
    record: MemoryRecord,
    *,
    content_hash: str,
    ingestion_at: str | None = None,
) -> None:
    """Upsert a :class:`MemoryRecord` and prune its existing chunks.

    Chunks are deleted on upsert (cascade), so the caller is expected to
    re-insert the new set inside the same transaction.
    """
    fm = record.frontmatter
    conn.execute(
        """
        INSERT INTO memories (
            id, type, vault_path, status, source, trust, project,
            importance, event_at, ingestion_at, last_accessed, access_count,
            supersedes, deprecated_by, pinned, content_hash, tags_json,
            frontmatter_json, body
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, COALESCE(?, 0),
            ?, ?, ?, ?, ?,
            ?, ?
        )
        ON CONFLICT(id) DO UPDATE SET
            type             = excluded.type,
            vault_path       = excluded.vault_path,
            status           = excluded.status,
            source           = excluded.source,
            trust            = excluded.trust,
            project          = excluded.project,
            importance       = excluded.importance,
            event_at         = excluded.event_at,
            ingestion_at     = excluded.ingestion_at,
            supersedes       = excluded.supersedes,
            deprecated_by    = excluded.deprecated_by,
            pinned           = excluded.pinned,
            content_hash     = excluded.content_hash,
            tags_json        = excluded.tags_json,
            frontmatter_json = excluded.frontmatter_json,
            body             = excluded.body
        """,
        (
            record.id,
            record.type,
            str(record.path),
            record.status,
            record.source,
            record.trust,
            record.project,
            float(fm.get("importance", 0.5)),
            str(fm["event_at"]),
            ingestion_at or dt.datetime.now(dt.UTC).isoformat(),
            fm.get("last_accessed"),
            int(fm.get("access_count", 0)),
            fm.get("supersedes"),
            fm.get("deprecated_by"),
            1 if record.pinned else 0,
            content_hash,
            json.dumps(fm.get("tags", [])),
            json.dumps(fm),
            record.body,
        ),
    )
    conn.execute("DELETE FROM chunks WHERE memory_id = ?", (record.id,))


def delete_memory(conn: sqlite3.Connection, memory_id: str) -> None:
    """Delete a memory and all dependent rows (chunks, access_log) via cascade."""
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


# ---------------------------------------------------------------------------
# Chunk insert
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChunkInsert:
    """One chunk plus its embeddings, ready to insert."""

    memory_id: str
    parent_in_batch: int | None
    section: str
    body: str
    token_count: int
    heading_path: tuple[str, ...]
    embedding_general: list[float] | None = None
    embedding_code: list[float] | None = None


def insert_chunks(conn: sqlite3.Connection, batch: list[ChunkInsert]) -> list[int]:
    """Insert chunks in order, returning the assigned row IDs.

    ``parent_in_batch`` is an index into ``batch`` (or ``None`` for the
    root chunk). It is translated to a real ``chunks.id`` reference at
    insert time, which is why we insert sequentially rather than with a
    single ``executemany``.
    """
    db_ids: list[int] = []
    for c in batch:
        parent_id: int | None = None
        if c.parent_in_batch is not None:
            if c.parent_in_batch >= len(db_ids):
                raise StoreError(f"forward parent reference in batch: {c.parent_in_batch}")
            parent_id = db_ids[c.parent_in_batch]

        cur = conn.execute(
            """
            INSERT INTO chunks (memory_id, parent_id, section, body, token_count, heading_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                c.memory_id,
                parent_id,
                c.section,
                c.body,
                c.token_count,
                json.dumps(list(c.heading_path)),
            ),
        )
        chunk_id = int(cur.lastrowid or 0)
        if chunk_id == 0:
            raise StoreError("failed to obtain rowid after INSERT INTO chunks")
        db_ids.append(chunk_id)

        # FTS5 external-content mirror.
        conn.execute(
            "INSERT INTO chunks_fts(rowid, body, section) VALUES (?, ?, ?)",
            (chunk_id, c.body, c.section),
        )

        if c.embedding_general is not None:
            conn.execute(
                "INSERT INTO chunks_vec_general(rowid, embedding) VALUES (?, ?)",
                (chunk_id, pack_vector(c.embedding_general)),
            )
        if c.embedding_code is not None:
            conn.execute(
                "INSERT INTO chunks_vec_code(rowid, embedding) VALUES (?, ?)",
                (chunk_id, pack_vector(c.embedding_code)),
            )

    return db_ids


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SearchHit:
    chunk_id: int
    memory_id: str
    score: float
    body: str
    section: str
    heading_path: tuple[str, ...]


def _row_to_hit(row: sqlite3.Row) -> SearchHit:
    return SearchHit(
        chunk_id=int(row["chunk_id"]),
        memory_id=str(row["memory_id"]),
        score=float(row["score"]),
        body=str(row["body"]),
        section=str(row["section"]),
        heading_path=tuple(json.loads(row["heading_path"] or "[]")),
    )


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    project: str | None = None,
) -> list[SearchHit]:
    """Full-text search via FTS5 BM25 ranking."""
    project_clause = "AND m.project = ?" if project else ""
    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.memory_id,
            -bm25(chunks_fts) AS score,
            c.body,
            c.section,
            c.heading_path
        FROM chunks_fts
        JOIN chunks   c ON c.id = chunks_fts.rowid
        JOIN memories m ON m.id = c.memory_id
        WHERE chunks_fts MATCH ?
          AND m.status = 'active'
          {project_clause}
        ORDER BY bm25(chunks_fts)
        LIMIT ?
    """
    params: list[Any] = [query]
    if project:
        params.append(project)
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_hit(r) for r in rows]


def search_vec(
    conn: sqlite3.Connection,
    vector: list[float],
    *,
    limit: int = 20,
    kind: Literal["general", "code"] = "general",
    project: str | None = None,
) -> list[SearchHit]:
    """Dense vector search via sqlite-vec.

    Distance is negated to become a score so RRF / weighted blends can
    use the same convention as :func:`search_bm25`.

    sqlite-vec's ``vec0`` KNN query requires a ``k = ?`` constraint
    inside its own WHERE clause; we wrap the vector lookup in a
    subquery so the outer joins can still filter by project / status.
    """
    table = "chunks_vec_general" if kind == "general" else "chunks_vec_code"
    project_clause = "AND m.project = ?" if project else ""
    sql = f"""
        WITH vec_hits AS (
            SELECT rowid, distance
            FROM {table}
            WHERE embedding MATCH ?
              AND k = ?
        )
        SELECT
            c.id AS chunk_id,
            c.memory_id,
            -v.distance AS score,
            c.body,
            c.section,
            c.heading_path
        FROM vec_hits v
        JOIN chunks   c ON c.id = v.rowid
        JOIN memories m ON m.id = c.memory_id
        WHERE m.status = 'active'
          {project_clause}
        ORDER BY v.distance
    """
    params: list[Any] = [pack_vector(vector), limit]
    if project:
        params.append(project)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_hit(r) for r in rows]


# ---------------------------------------------------------------------------
# Access logging
# ---------------------------------------------------------------------------


def mark_accessed(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    session_id: str | None,
    injection_type: Literal["auto", "tool_read", "rerank_hit"],
) -> None:
    """Record an access and bump ``memories.access_count`` / ``last_accessed``."""
    now = dt.datetime.now(dt.UTC).isoformat()
    conn.execute(
        """
        INSERT INTO access_log (memory_id, session_id, injected_at, injection_type)
        VALUES (?, ?, ?, ?)
        """,
        (memory_id, session_id, now, injection_type),
    )
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
        (now, memory_id),
    )


# ---------------------------------------------------------------------------
# Embedding cache (callable interface used by lib.embed.Embedder)
# ---------------------------------------------------------------------------


def cache_get(conn: sqlite3.Connection, content_hash: str, model: str) -> list[float] | None:
    row = conn.execute(
        "SELECT vector_blob FROM embedding_cache WHERE content_hash = ? AND model = ?",
        (content_hash, model),
    ).fetchone()
    if row is None:
        return None
    return unpack_vector(bytes(row["vector_blob"]))


def cache_put(
    conn: sqlite3.Connection,
    content_hash: str,
    model: str,
    vector: list[float],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO embedding_cache
            (content_hash, model, vector_blob, dim, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            content_hash,
            model,
            pack_vector(vector),
            len(vector),
            dt.datetime.now(dt.UTC).isoformat(),
        ),
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class StoreStats:
    memory_count: int
    chunk_count: int
    embedding_cache_count: int
    by_type: dict[str, int]
    by_project: dict[str, int]


def stats(conn: sqlite3.Connection) -> StoreStats:
    memory_count = int(conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"])
    chunk_count = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
    cache_count = int(conn.execute("SELECT COUNT(*) AS n FROM embedding_cache").fetchone()["n"])
    by_type = {
        row["type"]: int(row["n"])
        for row in conn.execute("SELECT type, COUNT(*) AS n FROM memories GROUP BY type").fetchall()
    }
    by_project = {
        (row["project"] or "<global>"): int(row["n"])
        for row in conn.execute(
            "SELECT project, COUNT(*) AS n FROM memories GROUP BY project"
        ).fetchall()
    }
    return StoreStats(
        memory_count=memory_count,
        chunk_count=chunk_count,
        embedding_cache_count=cache_count,
        by_type=by_type,
        by_project=by_project,
    )
