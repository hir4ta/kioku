"""Claude Code transcript parsing — JSONL + latest-mtime resolve.

Two known Claude Code issues motivate this module:

* **Issue #8564**: the ``transcript_path`` handed to a hook can be a
  stale sibling of the live transcript — Claude Code occasionally
  writes a fresh ``.jsonl`` next to the one it tells the hook about.
  :func:`latest_transcript` looks at the directory and picks the
  freshest ``.jsonl`` instead of trusting the hint blindly.
* **Issue #21022**: transcripts over ~50 MB hang Claude Code's stdin
  delivery. We mirror that floor: never read a file larger than
  :data:`MAX_TRANSCRIPT_BYTES`.

The output is a list of :class:`TranscriptTurn` with bodies that have
already been passed through :func:`kioku.redact.redact`, so downstream
callers (the SessionEnd summary writer, the Phase 5 ``claude -p``
extractor) never see raw secrets.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kioku.errors import KiokuError
from kioku.redact import redact

log = logging.getLogger("kioku.transcript")

MAX_TRANSCRIPT_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass(slots=True, frozen=True)
class TranscriptTurn:
    """One Claude Code conversation turn after redaction."""

    timestamp: dt.datetime | None
    role: str  # 'user' | 'assistant' | 'tool' | 'unknown'
    content: str


# ---------------------------------------------------------------------------
# Path resolution (issue #8564 mitigation)
# ---------------------------------------------------------------------------


def latest_transcript(hinted_path: Path | None) -> Path | None:
    """Pick the freshest ``.jsonl`` in the same dir as ``hinted_path``.

    Returns ``None`` when no candidate exists. If ``hinted_path`` is
    itself a regular file but a newer sibling exists, the newer one
    wins — that's the whole point of this helper.
    """
    if hinted_path is None:
        return None
    parent = hinted_path.parent
    if not parent.is_dir():
        return hinted_path if hinted_path.is_file() else None

    candidates = sorted(
        parent.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return hinted_path if hinted_path.is_file() else None
    return candidates[0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_transcript(path: Path) -> list[TranscriptTurn]:
    """Parse a transcript ``.jsonl`` into redacted turns.

    Raises :class:`KiokuError` if ``path`` does not exist. Files larger
    than :data:`MAX_TRANSCRIPT_BYTES` produce a warning and an empty
    list rather than blocking — we don't want a runaway transcript to
    take the SessionEnd hook down with it.
    """
    if not path.is_file():
        raise KiokuError(f"transcript not found: {path}")

    size = path.stat().st_size
    if size > MAX_TRANSCRIPT_BYTES:
        log.warning(
            "transcript too large (%.1f MB > %d MB cap), skipping: %s",
            size / 1024 / 1024,
            MAX_TRANSCRIPT_BYTES // 1024 // 1024,
            path,
        )
        return []

    turns: list[TranscriptTurn] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                log.debug("skipping malformed transcript line: %s", exc)
                continue
            if not isinstance(obj, dict):
                continue
            content = _extract_content(obj)
            if not content:
                continue
            r = redact(content)
            turns.append(
                TranscriptTurn(
                    timestamp=_parse_timestamp(obj),
                    role=str(obj.get("role") or obj.get("type") or "unknown"),
                    content=r.text,
                )
            )
    return turns


def _extract_content(obj: dict[str, Any]) -> str:
    """Pull a string body out of a transcript JSON object.

    Claude Code's transcript schema has drifted over time; this tries
    the field names that have shown up in 2025-2026 captures.
    """
    for field in ("content", "text", "message", "body"):
        value = obj.get(field)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for block in value:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            if parts:
                return "\n".join(parts)
    return ""


def _parse_timestamp(obj: dict[str, Any]) -> dt.datetime | None:
    for field in ("timestamp", "created_at", "time"):
        value = obj.get(field)
        if isinstance(value, str):
            try:
                return dt.datetime.fromisoformat(value)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Phase 4 machine summary (Phase 5 replaces this with `claude -p` extraction)
# ---------------------------------------------------------------------------


def summarize_machine(turns: list[TranscriptTurn]) -> str:
    """Cheap, deterministic session summary built from the turns alone.

    Phase 4 only — Phase 5 will run a fresh ``claude -p`` against the
    same turns and produce structured JSON (decisions / unresolved /
    next action). For Phase 4 we keep just the first user request and
    the last assistant response, which is enough to make the resulting
    ``episodic/sessions/*.md`` file useful when re-read later.
    """
    if not turns:
        return "(empty transcript)"
    user_first = next((t for t in turns if t.role == "user"), None)
    assistant_last = next((t for t in reversed(turns) if t.role == "assistant"), None)
    parts: list[str] = []
    if user_first:
        parts.append("## What was asked\n\n" + user_first.content[:1000])
    if assistant_last:
        parts.append("## What was done\n\n" + assistant_last.content[:2000])
    return "\n\n".join(parts) or "(no extractable content)"
