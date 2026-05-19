"""Tests for ``lib.vault``."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from kioku.errors import SchemaError, VaultError
from kioku.vault import (
    VAULT_SUBDIRS,
    make_id,
    path_for,
    read_memory,
    scaffold,
    walk,
    write_memory,
)


def test_scaffold_creates_all_subdirs(tmp_vault: Path) -> None:
    scaffold(tmp_vault, write_claude_md=False)
    for sub in VAULT_SUBDIRS:
        assert (tmp_vault / sub).is_dir(), f"missing {sub}"


def test_scaffold_is_idempotent(tmp_vault: Path) -> None:
    scaffold(tmp_vault, write_claude_md=False)
    scaffold(tmp_vault, write_claude_md=False)
    for sub in VAULT_SUBDIRS:
        assert (tmp_vault / sub).is_dir()


def test_scaffold_refuses_missing_root(tmp_path: Path) -> None:
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(VaultError):
        scaffold(bogus)


def test_read_write_roundtrip(tmp_vault: Path, sample_memory_text: str) -> None:
    path = tmp_vault / "semantic" / "decisions" / "active" / "sample.md"
    path.parent.mkdir(parents=True)
    path.write_text(sample_memory_text, encoding="utf-8")

    record = read_memory(path)
    assert record.id == "DEC-2026-05-19-test-decision"
    assert record.type == "decision"
    assert record.trust == "high"
    assert "voyage-4-large" in record.body

    record.body = "# Updated body\n\nNew content here."
    write_memory(record)

    record2 = read_memory(path)
    assert record2.body.startswith("# Updated body")


def test_read_rejects_bad_id_pattern(tmp_vault: Path) -> None:
    path = tmp_vault / "bad.md"
    path.write_text(
        """---
id: BAD_ID
type: decision
status: active
created_at: '2026-05-19T10:00:00+00:00'
event_at: '2026-05-19T10:00:00+00:00'
source: user-notes
trust: high
---

body
""",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        read_memory(path)


def test_read_rejects_missing_required(tmp_vault: Path) -> None:
    path = tmp_vault / "bad.md"
    path.write_text(
        """---
id: DEC-2026-05-19-bad
type: decision
status: active
---

missing source / trust / created_at / event_at
""",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        read_memory(path)


def test_walk_yields_only_valid_md(tmp_vault: Path, sample_memory_text: str) -> None:
    scaffold(tmp_vault, write_claude_md=True)
    sample = tmp_vault / "semantic" / "decisions" / "active" / "x.md"
    sample.write_text(sample_memory_text, encoding="utf-8")

    found = list(walk(tmp_vault))
    assert len(found) == 1
    assert found[0].id == "DEC-2026-05-19-test-decision"


def test_walk_skips_claude_md_and_ignored(tmp_vault: Path, sample_memory_text: str) -> None:
    scaffold(tmp_vault, write_claude_md=True)
    # _meta is in DEFAULT_IGNORE.
    (tmp_vault / "_meta" / "dashboard.md").write_text("should be ignored", encoding="utf-8")
    (tmp_vault / "x.tmp.md").write_text("temp", encoding="utf-8")  # tmp.md ignored
    valid = tmp_vault / "semantic" / "decisions" / "active" / "ok.md"
    valid.write_text(sample_memory_text, encoding="utf-8")

    found = list(walk(tmp_vault))
    assert {r.id for r in found} == {"DEC-2026-05-19-test-decision"}


def test_make_id_format() -> None:
    d = dt.date(2026, 5, 19)
    assert make_id("decision", "foo-bar", on=d) == "DEC-2026-05-19-foo-bar"


def test_make_id_rejects_unknown_type() -> None:
    with pytest.raises(VaultError):
        make_id("bogus", "x")


def test_path_for_global(tmp_vault: Path) -> None:
    d = dt.date(2026, 5, 19)
    p = path_for(tmp_vault, "pattern", "bash-strict", on=d)
    assert p.parts[-3:] == ("semantic", "patterns", "2026-05-19-bash-strict.md")


def test_path_for_project_scoped(tmp_vault: Path) -> None:
    d = dt.date(2026, 5, 19)
    p = path_for(tmp_vault, "decision", "x", on=d, project="myproj")
    assert "projects/myproj/decisions" in str(p)
