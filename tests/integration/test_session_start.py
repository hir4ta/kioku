"""End-to-end test: ``kioku hook session-start`` returns a valid hook response.

Uses ``click.testing.CliRunner`` to invoke the CLI in-process, which is
faster and easier to debug than spawning a subprocess. The bash wrapper
in ``hooks/memory/session-start.sh`` is a thin pass-through to this
same code path, so testing the CLI here covers it.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
from click.testing import CliRunner
from kioku.cli.main import cli


def _write_decision(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        dedent(
            """\
            ---
            id: DEC-2026-05-19-voyage
            type: decision
            status: active
            created_at: '2026-05-19T10:00:00+00:00'
            event_at: '2026-05-19T10:00:00+00:00'
            source: user-notes
            trust: high
            ---

            # Use voyage-4-large

            Decided.
            """
        ),
        encoding="utf-8",
    )


def test_session_start_startup_inlines_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    (vault / "working").mkdir(parents=True)
    (vault / "working" / "focus.md").write_text("# Focus\nWorking on Phase 2.", encoding="utf-8")
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-start", "startup"], input="")
    assert result.exit_code == 0, result.output

    parsed = json.loads(result.stdout)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "<system_constraint>" in ctx
    assert "Working on Phase 2" in ctx  # focus body inlined


def test_session_start_resume_skips_focus_uses_next_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    (vault / "working").mkdir(parents=True)
    (vault / "working" / "next.md").write_text("# Next\nWrite tests.", encoding="utf-8")
    (vault / "working" / "unresolved.md").write_text("# Unresolved\nReview docs.", encoding="utf-8")
    # focus.md exists but resume should NOT inject it (transcript covers it).
    (vault / "working" / "focus.md").write_text("# Focus\nThe thing.", encoding="utf-8")
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-start", "resume"], input="")
    assert result.exit_code == 0, result.output

    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Write tests" in ctx
    assert "Review docs" in ctx
    assert "The thing" not in ctx  # focus excluded on resume


def test_session_start_fail_open_on_missing_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(tmp_path / "does-not-exist"))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-start", "startup"], input="")
    assert result.exit_code == 0, result.output

    parsed = json.loads(result.stdout)
    # Constraint always renders even when the vault is gone.
    assert "<system_constraint>" in parsed["hookSpecificOutput"]["additionalContext"]


def test_session_start_startup_appends_active_decision_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    _write_decision(vault / "semantic" / "decisions" / "active" / "voyage.md")
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-start", "startup"], input="")
    assert result.exit_code == 0, result.output

    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "DEC-2026-05-19-voyage" in ctx
    assert "<system_memory_layer" in ctx
    # Decision is identifier-only (system layer), no inline content.
    assert "<content>Decided." not in ctx


def test_session_start_compact_pulls_handover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    (vault / "compact-handover").mkdir(parents=True)
    (vault / "compact-handover" / "session-abc.md").write_text(
        "# Handover\nKeep the cache invariant tested.", encoding="utf-8"
    )
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-start", "compact"], input="")
    assert result.exit_code == 0, result.output

    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "Keep the cache invariant tested" in ctx


def test_session_start_invalid_stdin_still_returns_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed stdin payload must not crash the hook (fail-open)."""
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["hook", "session-start", "startup"],
        input="this is not JSON",
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert "hookSpecificOutput" in parsed
