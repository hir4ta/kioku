"""kioku CLI entry point.

Implements the ``kioku`` console script declared in ``pyproject.toml``.

Subcommands:

* ``rebuild``   — Full re-ETL: walk vault → chunk → embed → upsert SQLite.
* ``scaffold``  — Create the canonical vault directory layout.
* ``search``    — Hybrid retrieval. Phase 1 ships BM25 only; Phase 3 adds
                  dense + rerank.
* ``status``    — Print vault + store statistics.
* ``version``   — Print the kioku version.
* ``hook``      — Claude Code lifecycle hook entry points (Phase 2+).
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from kioku import __version__
from kioku.chunk import chunk_body
from kioku.cli.hook import hook as hook_group
from kioku.config import get_voyage_api_key, load_settings
from kioku.embed import Embedder, content_hash
from kioku.errors import ConfigError, KiokuError
from kioku.retrieve import hybrid_search
from kioku.store_sqlite import (
    ChunkInsert,
    cache_get,
    cache_put,
    connect,
    init_schema,
    insert_chunks,
    stats,
    upsert_memory,
)
from kioku.vault import scaffold as vault_scaffold
from kioku.vault import walk as vault_walk

log = logging.getLogger("kioku")
console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Route logging through ``rich`` for readable terminal output."""
    handler = RichHandler(console=console, show_time=False, show_path=False)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[handler],
    )


def _db_path(_vault_path: Path) -> Path:
    """Resolve the SQLite store path.

    Currently fixed at ``~/.local/share/kioku/store/kioku.sqlite``. This
    is single-vault friendly. Phase 8 may make it per-vault if running
    multiple parallel vaults turns out to be a real use case (yagni for now).
    """
    base = Path.home() / ".local" / "share" / "kioku" / "store"
    base.mkdir(parents=True, exist_ok=True)
    return base / "kioku.sqlite"


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging (DEBUG level).")
def cli(verbose: bool) -> None:
    """kioku — persistent long-term memory for Claude Code."""
    _setup_logging(verbose)


cli.add_command(hook_group)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@cli.command()
def version() -> None:
    """Print the kioku version."""
    click.echo(__version__)


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------


@cli.command()
def scaffold() -> None:
    """Create the canonical vault directory layout under the configured vault path."""
    settings = load_settings()
    vault_scaffold(settings.vault_path)
    console.print(f"[green]vault scaffolded at[/green] {settings.vault_path}")


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------


@cli.command()
def rebuild() -> None:
    """Full re-ETL: vault → SQLite (memories + chunks + FTS5 + vec)."""
    settings = load_settings()
    api_key = get_voyage_api_key(settings)
    db_path = _db_path(settings.vault_path)
    log.info("kioku rebuild — vault=%s db=%s", settings.vault_path, db_path)

    with connect(db_path) as conn:
        init_schema(
            conn,
            dim_general=settings.voyage.dim,
            dim_code=settings.voyage.dim,
        )

        embedder = Embedder(
            api_key=api_key,
            model_general=settings.voyage.model_general,
            model_code=settings.voyage.model_code,
            dim=settings.voyage.dim,
            cache_get=lambda h, m: cache_get(conn, h, m),
            cache_put=lambda h, m, v: cache_put(conn, h, m, v),
        )

        memory_count = 0
        chunk_count = 0
        ingestion_at = dt.datetime.now(dt.UTC).isoformat()

        for record in vault_walk(
            settings.vault_path,
            ignore_patterns=tuple(settings.vault.ignore_patterns),
        ):
            log.debug("ingesting %s", record.id)
            pieces = chunk_body(record.body)
            embed_results = embedder.embed(
                [p.body for p in pieces],
                kind="general",
                input_type="document",
            )

            upsert_memory(
                conn,
                record,
                content_hash=content_hash(record.body),
                ingestion_at=ingestion_at,
            )
            inserts = [
                ChunkInsert(
                    memory_id=record.id,
                    parent_in_batch=p.parent_index,
                    section=p.section,
                    body=p.body,
                    token_count=p.token_count,
                    heading_path=p.heading_path,
                    embedding_general=e.vector,
                )
                for p, e in zip(pieces, embed_results, strict=True)
            ]
            insert_chunks(conn, inserts)
            memory_count += 1
            chunk_count += len(pieces)

        console.print(
            f"[green]rebuild complete[/green] — {memory_count} memories, {chunk_count} chunks"
        )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query", required=True)
@click.option("--top-k", default=5, show_default=True, help="Number of hits to return.")
@click.option("--project", default=None, help="Limit to a single project slug.")
@click.option(
    "--no-rerank",
    is_flag=True,
    help="Skip Voyage rerank-2.5 (faster, slightly less precise ordering).",
)
@click.option(
    "--bm25-only",
    is_flag=True,
    help="Skip dense embedding and rerank entirely — no Voyage API call.",
)
def search(
    query: str,
    top_k: int,
    project: str | None,
    no_rerank: bool,
    bm25_only: bool,
) -> None:
    """Hybrid search: BM25 + dense + Voyage rerank-2.5 + composite scoring.

    Gracefully degrades to BM25-only when no Voyage API key is set.
    """
    settings = load_settings()
    db_path = _db_path(settings.vault_path)
    if not db_path.is_file():
        console.print("[yellow]store not initialized — run `kioku rebuild` first[/yellow]")
        sys.exit(1)

    api_key: str | None = None
    if not bm25_only:
        try:
            api_key = get_voyage_api_key(settings)
        except ConfigError as exc:
            console.print(
                f"[yellow]Voyage API key not available; falling back to BM25-only "
                f"({exc.__class__.__name__})[/yellow]"
            )

    with connect(db_path) as conn:
        embedder: Embedder | None = None
        if api_key is not None:
            embedder = Embedder(
                api_key=api_key,
                model_general=settings.voyage.model_general,
                model_code=settings.voyage.model_code,
                model_rerank=settings.voyage.model_rerank,
                dim=settings.voyage.dim,
                cache_get=lambda h, m: cache_get(conn, h, m),
                cache_put=lambda h, m, v: cache_put(conn, h, m, v),
            )

        hits = hybrid_search(
            conn,
            query,
            settings=settings,
            project=project,
            embedder=embedder,
            enable_rerank=(not no_rerank) and embedder is not None,
            top_k=top_k,
        )

    if not hits:
        console.print("[yellow]no hits[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("memory_id", style="cyan", no_wrap=True)
    table.add_column("section")
    table.add_column("composite", justify="right")
    table.add_column("rrf", justify="right")
    table.add_column("rerank", justify="right")
    table.add_column("body")
    for h in hits:
        body_preview = h.body.replace("\n", " ")[:60]
        rerank_str = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "—"
        table.add_row(
            h.memory_id,
            h.section,
            f"{h.composite_score:.3f}",
            f"{h.rrf_score:.3f}",
            rerank_str,
            body_preview,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command()
def status() -> None:
    """Print vault and store statistics."""
    settings = load_settings()
    db_path = _db_path(settings.vault_path)

    console.print(f"[bold]vault path:[/bold]  {settings.vault_path}")
    console.print(f"[bold]db path:[/bold]     {db_path}")
    console.print(
        f"[bold]voyage:[/bold]      {settings.voyage.model_general}, dim={settings.voyage.dim}"
    )

    if not db_path.is_file():
        console.print("[yellow]store not initialized — run `kioku rebuild`[/yellow]")
        return

    with connect(db_path) as conn:
        s = stats(conn)

    summary = Table(show_header=False, box=None)
    summary.add_row("memories", str(s.memory_count))
    summary.add_row("chunks", str(s.chunk_count))
    summary.add_row("cached embeddings", str(s.embedding_cache_count))
    console.print(summary)

    if s.by_type:
        t = Table(title="by type", show_header=True, header_style="bold")
        t.add_column("type")
        t.add_column("count", justify="right")
        for type_name, count in sorted(s.by_type.items(), key=lambda x: -x[1]):
            t.add_row(type_name, str(count))
        console.print(t)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """``kioku`` console script entry point."""
    try:
        cli(standalone_mode=False)
    except KiokuError as exc:
        console.print(f"[red]error:[/red] {exc.__class__.__name__}: {exc}")
        sys.exit(1)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.exceptions.Abort:
        console.print("[yellow]aborted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
