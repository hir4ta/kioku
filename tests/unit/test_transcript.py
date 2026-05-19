"""Tests for ``kioku.transcript``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from kioku.errors import KiokuError
from kioku.transcript import (
    MAX_TRANSCRIPT_BYTES,
    TranscriptTurn,
    latest_transcript,
    parse_transcript,
    summarize_machine,
)

# ---------------------------------------------------------------------------
# latest_transcript — issue #8564 mitigation
# ---------------------------------------------------------------------------


def test_latest_transcript_none_input_returns_none() -> None:
    assert latest_transcript(None) is None


def test_latest_transcript_missing_parent_returns_hint_or_none(tmp_path: Path) -> None:
    bogus = tmp_path / "no-such-dir" / "x.jsonl"
    assert latest_transcript(bogus) is None


def test_latest_transcript_picks_newest_in_dir(tmp_path: Path) -> None:
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    newer.write_text("{}", encoding="utf-8")

    # Even when hinted at the older one, latest_transcript should pick newer.
    chosen = latest_transcript(older)
    assert chosen == newer


def test_latest_transcript_single_jsonl_returned(tmp_path: Path) -> None:
    only = tmp_path / "only.jsonl"
    only.write_text("{}", encoding="utf-8")
    assert latest_transcript(only) == only


# ---------------------------------------------------------------------------
# parse_transcript
# ---------------------------------------------------------------------------


def test_parse_transcript_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(KiokuError):
        parse_transcript(tmp_path / "nope.jsonl")


def test_parse_transcript_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("", encoding="utf-8")
    assert parse_transcript(p) == []


def test_parse_transcript_simple_turns(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    lines = [
        json.dumps({"role": "user", "content": "Hello"}),
        json.dumps({"role": "assistant", "content": "Hi there"}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    turns = parse_transcript(p)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hi there"


def test_parse_transcript_redacts_secrets(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    secret = "sk-ant-api03-AbCdEfGhIj0123456789KlMnOpQrSt"
    p.write_text(
        json.dumps({"role": "user", "content": f"my key is {secret}"}) + "\n",
        encoding="utf-8",
    )
    turns = parse_transcript(p)
    assert secret not in turns[0].content
    assert "[REDACTED:anthropic-key]" in turns[0].content


def test_parse_transcript_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        "not json at all\n"
        + json.dumps({"role": "user", "content": "ok"})
        + "\n"
        + "{broken json\n",
        encoding="utf-8",
    )
    turns = parse_transcript(p)
    assert len(turns) == 1
    assert turns[0].content == "ok"


def test_parse_transcript_handles_content_blocks(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first block"},
                    {"type": "text", "text": "second block"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    turns = parse_transcript(p)
    assert len(turns) == 1
    assert "first block" in turns[0].content
    assert "second block" in turns[0].content


def test_parse_transcript_oversize_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "big.jsonl"
    p.write_text("x" * (MAX_TRANSCRIPT_BYTES + 1), encoding="utf-8")
    assert parse_transcript(p) == []


# ---------------------------------------------------------------------------
# summarize_machine
# ---------------------------------------------------------------------------


def test_summarize_machine_empty() -> None:
    assert summarize_machine([]) == "(empty transcript)"


def test_summarize_machine_uses_first_user_and_last_assistant() -> None:
    turns = [
        TranscriptTurn(timestamp=None, role="user", content="Build me Phase 4."),
        TranscriptTurn(timestamp=None, role="assistant", content="Working on it."),
        TranscriptTurn(timestamp=None, role="user", content="More?"),
        TranscriptTurn(timestamp=None, role="assistant", content="Phase 4 complete."),
    ]
    out = summarize_machine(turns)
    assert "What was asked" in out
    assert "Build me Phase 4" in out
    assert "What was done" in out
    assert "Phase 4 complete" in out


def test_summarize_machine_truncates_long_bodies() -> None:
    huge = "x" * 5000
    turns = [
        TranscriptTurn(timestamp=None, role="user", content=huge),
        TranscriptTurn(timestamp=None, role="assistant", content=huge),
    ]
    out = summarize_machine(turns)
    # 1000 cap on user, 2000 on assistant → total << 5000+5000
    assert len(out) < 5000
