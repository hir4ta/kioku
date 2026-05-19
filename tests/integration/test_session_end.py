"""End-to-end: ``kioku hook session-end`` materialises a transcript.

Driven through ``click.testing.CliRunner`` for parity with the
session-start integration test. The bash wrapper
``hooks/memory/session-end.sh`` is a thin pass-through to this same
code path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from kioku.cli.main import cli


def _write_transcript(path: Path, *turns: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(t) for t in turns) + "\n", encoding="utf-8")


def _payload(transcript_path: Path, session_id: str = "test-session-12345") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": str(transcript_path),
            "hook_event_name": "SessionEnd",
            "cwd": "/tmp",
        }
    )


def test_session_end_writes_vault_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    transcript = tmp_path / "transcripts" / "session.jsonl"
    _write_transcript(
        transcript,
        {"role": "user", "content": "Implement Phase 4 of kioku"},
        {"role": "assistant", "content": "Working on it. Built redact + transcript."},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-end"], input=_payload(transcript))
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {}

    sessions_dir = vault / "episodic" / "sessions"
    md_files = list(sessions_dir.glob("*.md"))
    assert len(md_files) == 1
    body = md_files[0].read_text(encoding="utf-8")
    assert "Implement Phase 4" in body
    assert "Working on it" in body
    assert "type: session" in body
    assert "source: auto-extracted" in body
    assert "trust: medium" in body


def test_session_end_redacts_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    transcript = tmp_path / "transcripts" / "session.jsonl"
    secret = "sk-ant-api03-AbCdEfGhIj0123456789KlMnOpQrSt"
    _write_transcript(
        transcript,
        {"role": "user", "content": f"my key is {secret}"},
        {"role": "assistant", "content": "noted"},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-end"], input=_payload(transcript))
    assert result.exit_code == 0

    md = next((vault / "episodic" / "sessions").glob("*.md"))
    body = md.read_text(encoding="utf-8")
    assert secret not in body
    assert "[REDACTED:anthropic-key]" in body


def test_session_end_no_transcript_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    # Empty payload — no transcript_path at all.
    result = runner.invoke(cli, ["hook", "session-end"], input="{}")
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {}
    # Vault is untouched.
    sessions = vault / "episodic" / "sessions"
    if sessions.is_dir():
        assert list(sessions.glob("*.md")) == []


def test_stop_uses_latest_mtime_over_hinted_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mitigation for claude-code issue #8564 — stop picks the newest .jsonl."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    older = transcript_dir / "older.jsonl"
    newer = transcript_dir / "newer.jsonl"

    _write_transcript(older, {"role": "user", "content": "stale session content"})
    time.sleep(0.01)
    _write_transcript(newer, {"role": "user", "content": "fresh session content"})

    # Hint at the OLDER file deliberately; stop must pick newer via mtime.
    payload = json.dumps(
        {
            "session_id": "x",
            "transcript_path": str(older),
            "hook_event_name": "Stop",
            "cwd": "/tmp",
        }
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "stop"], input=payload)
    assert result.exit_code == 0

    md = next((vault / "episodic" / "sessions").glob("*.md"))
    body = md.read_text(encoding="utf-8")
    assert "fresh session content" in body
    assert "stale session content" not in body


def test_session_end_invalid_stdin_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("KIOKU_VAULT_PATH", str(vault))

    runner = CliRunner()
    result = runner.invoke(cli, ["hook", "session-end"], input="this is not JSON")
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {}
