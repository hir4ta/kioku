"""Vault I/O — read and write Markdown memory files in the Obsidian vault.

This module is the only place that touches the vault filesystem. All
other layers consume :class:`MemoryRecord` objects produced here.

The vault is the source of truth (L4 in the kioku architecture). When
this module writes a file, downstream layers (L3 SQLite, L2 DuckDB) must
be told to update via ``kioku rebuild`` or via the SessionEnd hook.

Frontmatter is validated against ``schemas/memory.schema.json`` on every
read and write, so drift between Pydantic models and the JSON Schema is
caught immediately rather than at search time.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import jsonschema

from kioku.errors import SchemaError, VaultError

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

VAULT_SUBDIRS: tuple[str, ...] = (
    "_meta",
    "working",
    "episodic/sessions",
    "episodic/transcripts",
    "semantic/decisions/active",
    "semantic/decisions/deprecated",
    "semantic/decisions/archive",
    "semantic/patterns",
    "semantic/mistakes",
    "semantic/preferences",
    "semantic/glossary",
    "semantic/references",
    "procedural/skills",
    "procedural/recipes",
    "projects",
    "people",
    "compact-handover",
)

TYPE_TO_SUBDIR: dict[str, str] = {
    "session": "episodic/sessions",
    "decision": "semantic/decisions/active",
    "pattern": "semantic/patterns",
    "mistake": "semantic/mistakes",
    "preference": "semantic/preferences",
    "reference": "semantic/references",
    "skill": "procedural/skills",
    "recipe": "procedural/recipes",
    "person": "people",
    "glossary": "semantic/glossary",
}

TYPE_PREFIX: dict[str, str] = {
    "session": "SESSION",
    "decision": "DEC",
    "pattern": "PAT",
    "mistake": "MIS",
    "preference": "PREF",
    "reference": "REF",
    "skill": "SKILL",
    "recipe": "RECIPE",
    "person": "PERSON",
    "glossary": "GLOSS",
}

ID_PATTERN = re.compile(r"^[A-Z]+-\d{4}-\d{2}-\d{2}-[a-z0-9-]+$")
DEFAULT_IGNORE: tuple[str, ...] = (".obsidian/**", "_meta/**", "**/*.tmp.md")

# ---------------------------------------------------------------------------
# Schema cache
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "memory.schema.json"
_TEMPLATE_DIR = _REPO_ROOT / "templates"

_MEMORY_SCHEMA: dict[str, Any] | None = None


def _load_memory_schema() -> dict[str, Any]:
    """Lazily load and cache ``schemas/memory.schema.json``."""
    global _MEMORY_SCHEMA
    if _MEMORY_SCHEMA is None:
        if not _SCHEMA_PATH.is_file():
            raise VaultError(f"memory.schema.json not found at {_SCHEMA_PATH}")
        _MEMORY_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _MEMORY_SCHEMA


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MemoryRecord:
    """A vault Markdown file, frontmatter + body, post-validation."""

    path: Path
    frontmatter: dict[str, Any]
    body: str

    @property
    def id(self) -> str:
        return str(self.frontmatter["id"])

    @property
    def type(self) -> str:
        return str(self.frontmatter["type"])

    @property
    def trust(self) -> str:
        return str(self.frontmatter["trust"])

    @property
    def source(self) -> str:
        return str(self.frontmatter["source"])

    @property
    def status(self) -> str:
        return str(self.frontmatter["status"])

    @property
    def pinned(self) -> bool:
        return bool(self.frontmatter.get("pinned", False))

    @property
    def project(self) -> str | None:
        value = self.frontmatter.get("project")
        return str(value) if value else None


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------


def scaffold(vault_root: Path, *, write_claude_md: bool = True) -> None:
    """Create the canonical kioku vault layout under ``vault_root``.

    Idempotent: missing directories are created, existing files are
    untouched. The vault root itself must already exist (the user's
    Obsidian setup decides where it lives).

    Raises
    ------
    VaultError
        ``vault_root`` does not exist or is not a directory.
    """
    if not vault_root.exists():
        raise VaultError(f"vault root does not exist: {vault_root}")
    if not vault_root.is_dir():
        raise VaultError(f"vault root is not a directory: {vault_root}")

    for sub in VAULT_SUBDIRS:
        (vault_root / sub).mkdir(parents=True, exist_ok=True)

    if write_claude_md:
        target = vault_root / "CLAUDE.md"
        source = _TEMPLATE_DIR / "CLAUDE.md.tmpl"
        if not target.exists() and source.is_file():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------


def _validate_frontmatter(fm: dict[str, Any], *, source_path: Path) -> None:
    schema = _load_memory_schema()
    try:
        jsonschema.validate(fm, schema)
    except jsonschema.ValidationError as exc:
        raise SchemaError(
            f"frontmatter invalid in {source_path}: {exc.message} at {list(exc.absolute_path)}"
        ) from exc
    if not ID_PATTERN.match(str(fm.get("id", ""))):
        raise SchemaError(f"id pattern mismatch in {source_path}: id={fm.get('id')!r}")


def read_memory(path: Path) -> MemoryRecord:
    """Read and validate a single memory file."""
    if not path.is_file():
        raise VaultError(f"not a file: {path}")
    try:
        post = frontmatter.load(str(path))
    except OSError as exc:
        raise VaultError(f"failed to read {path}: {exc}") from exc

    fm: dict[str, Any] = dict(post.metadata)
    _validate_frontmatter(fm, source_path=path)
    return MemoryRecord(path=path, frontmatter=fm, body=post.content)


def write_memory(record: MemoryRecord) -> None:
    """Write the record back, validating before touching disk.

    The write is atomic on POSIX: contents are written to ``<path>.tmp``
    and renamed over the destination. This avoids leaving a half-written
    file if the process is killed mid-write.
    """
    _validate_frontmatter(record.frontmatter, source_path=record.path)
    record.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.path.with_suffix(record.path.suffix + ".tmp")
    payload = frontmatter.Post(record.body, **record.frontmatter)
    try:
        tmp.write_text(frontmatter.dumps(payload), encoding="utf-8")
        tmp.replace(record.path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise VaultError(f"failed to write {record.path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------


def walk(
    vault_root: Path,
    *,
    ignore_patterns: Sequence[str] = DEFAULT_IGNORE,
) -> Iterator[MemoryRecord]:
    """Yield every :class:`MemoryRecord` under ``vault_root``.

    Files that fail frontmatter validation raise :class:`SchemaError`
    immediately — callers handle the error explicitly rather than
    silently skipping. A corrupt frontmatter is a bug, not a no-op.
    """
    if not vault_root.is_dir():
        raise VaultError(f"vault root missing: {vault_root}")

    for md in sorted(vault_root.rglob("*.md")):
        rel = md.relative_to(vault_root)
        if _matches_ignore(rel, ignore_patterns):
            continue
        if md.name == "CLAUDE.md":
            continue
        yield read_memory(md)


def _matches_ignore(rel: Path, patterns: Sequence[str]) -> bool:
    """Return True if ``rel`` matches any of the ignore globs.

    Uses :meth:`pathlib.PurePath.full_match` on 3.13+ (which honours
    ``**`` recursively, the behavior we want). Falls back to a manual
    decomposition on 3.12 so the codebase stays portable across the
    declared ``requires-python = ">=3.12"`` range.
    """
    if hasattr(rel, "full_match"):
        return any(rel.full_match(pat) for pat in patterns)

    posix = rel.as_posix()
    name = rel.name
    for pat in patterns:
        if pat.startswith("**/"):
            sub = pat[3:]
            if fnmatch.fnmatch(name, sub) or fnmatch.fnmatch(posix, sub):
                return True
        elif pat.endswith("/**"):
            prefix = pat[:-3]
            if posix == prefix or posix.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatch(posix, pat):
            return True
    return False


# ---------------------------------------------------------------------------
# ID / path helpers
# ---------------------------------------------------------------------------


def make_id(type_: str, slug: str, *, on: dt.date | None = None) -> str:
    """Compose a canonical memory ID: ``<PREFIX>-YYYY-MM-DD-<slug>``."""
    prefix = TYPE_PREFIX.get(type_)
    if prefix is None:
        raise VaultError(f"unknown memory type: {type_!r}")
    date = (on or dt.date.today()).isoformat()
    return f"{prefix}-{date}-{slug}"


def path_for(
    vault_root: Path,
    type_: str,
    slug: str,
    *,
    on: dt.date | None = None,
    project: str | None = None,
) -> Path:
    """Build the canonical on-disk path for a new memory of ``type_``.

    When ``project`` is given, the path is rooted under ``projects/<project>/``
    and the type-specific subdirectory is collapsed to the file basename
    (so project-scoped decisions live in ``projects/foo/decisions/``).
    """
    date = (on or dt.date.today()).isoformat()
    if project:
        return vault_root / "projects" / project / f"{type_}s" / f"{date}-{slug}.md"
    sub = TYPE_TO_SUBDIR.get(type_)
    if sub is None:
        raise VaultError(f"unknown memory type: {type_!r}")
    return vault_root / sub / f"{date}-{slug}.md"


# ---------------------------------------------------------------------------
# Plain Markdown — for working/ and compact-handover/ files
# ---------------------------------------------------------------------------


def read_plain_markdown(path: Path) -> str:
    """Read a Markdown file without frontmatter validation.

    Used for ``working/*.md`` and ``compact-handover/*.md`` files,
    which are user-editable / auto-generated plain Markdown and
    intentionally do *not* carry the strict frontmatter contract that
    :func:`read_memory` enforces.
    """
    if not path.is_file():
        raise VaultError(f"not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VaultError(f"failed to read {path}: {exc}") from exc


def first_heading(markdown: str) -> str | None:
    """Return the text of the first ``#``-level heading, or ``None``.

    Useful for synthesising a display title for plain-Markdown files
    that have no frontmatter ``title`` field.
    """
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None
