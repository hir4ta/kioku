"""kioku — persistent long-term memory for Claude Code.

This package houses the Python core. Hook entry points (`hooks/memory/*.sh`)
are thin bash wrappers that shell out to the ``kioku`` CLI (see ``cli``).

The architecture is layered:

* L4 (truth): Obsidian vault on disk, plain Markdown with YAML frontmatter.
* L3 (index): SQLite + FTS5 + sqlite-vec, derived from L4 via the ETL in
  this package (``lib.vault`` → ``lib.chunk`` → ``lib.embed`` → ``lib.store_sqlite``).
* L2 (view): DuckDB analytical views, planned for Phase 8.
* L1 (logic): this package, plus ``cli`` and ``hooks`` and ``cron``.

See ``docs/architecture.md`` for the full picture.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
