"""Hook entry points — ``kioku hook <event>`` subcommands.

This module is the Python side of the bash wrappers in
``hooks/memory/``. The wrappers shell out here with the matched event
and source; this module reads the Claude Code hook payload from stdin,
decides what to inject, and writes the spec-compliant response JSON to
stdout.

Phase 2 ships ``session-start`` for the four SessionStart matchers
(``startup`` / ``resume`` / ``clear`` / ``compact``). PreCompact,
SessionEnd, Stop, and UserPromptSubmit subcommands land in Phases 4–5.

The hook is **fail-open**: any unexpected exception is caught, logged
to stderr, and replaced with an empty ``additionalContext`` so a
broken kioku install never blocks the user from starting a session.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import click

from kioku.config import KiokuSettings, load_settings
from kioku.errors import KiokuError
from kioku.inject import (
    InjectedMemory,
    InjectionPayload,
    format_payload,
)
from kioku.vault import first_heading, read_memory, read_plain_markdown

log = logging.getLogger("kioku.hook")

# Claude Code's SessionStart source enum.
SessionStartSource = Literal["startup", "resume", "clear", "compact"]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def hook() -> None:
    """Claude Code lifecycle hook entry points."""


@hook.command("session-start")
@click.argument(
    "source",
    type=click.Choice(["startup", "resume", "clear", "compact"]),
)
def hook_session_start(source: SessionStartSource) -> None:
    """Handle a ``SessionStart`` hook for the given matcher source.

    Reads the hook payload (JSON) from stdin, builds an injection
    payload appropriate for ``source``, and writes the spec-compliant
    JSON response to stdout.
    """
    response = _safe_session_start(source)
    json.dump(response, sys.stdout)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Top-level "do not crash the session" wrapper
# ---------------------------------------------------------------------------


def _safe_session_start(source: SessionStartSource) -> dict[str, Any]:
    """Return a hook response, swallowing every failure as empty context."""
    try:
        payload_in = _read_hook_stdin()
        settings = load_settings()
        injection = _build_session_start_payload(source, settings, payload_in)
        xml, notes = format_payload(
            injection,
            token_budget=settings.inject.active_recall_token_cap,
        )
        if notes:
            log.info("inject notes (%s): %s", source, "; ".join(notes))
    except Exception as exc:  # fail-open
        log.warning("session-start hook failed (%s): %s", source, exc)
        xml = ""

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": xml,
        }
    }


# ---------------------------------------------------------------------------
# stdin parsing
# ---------------------------------------------------------------------------


def _read_hook_stdin() -> dict[str, Any]:
    """Parse the single JSON object Claude Code passes on stdin.

    Returns ``{}`` (and logs a warning) on empty / unparseable input
    rather than raising; the surrounding ``_safe_session_start`` wrapper
    relies on that to keep the hook fail-open.
    """
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("hook stdin is not valid JSON: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        log.warning("hook stdin is not a JSON object (got %s)", type(parsed).__name__)
        return {}
    return parsed


# ---------------------------------------------------------------------------
# Per-source payload assembly
# ---------------------------------------------------------------------------


def _build_session_start_payload(
    source: SessionStartSource,
    settings: KiokuSettings,
    _hook_payload: dict[str, Any],
) -> InjectionPayload:
    """Decide what to inject for the given SessionStart source.

    * ``startup`` / ``clear``: full ``focus.md``, plus identifier-only
      ``next.md`` / ``unresolved.md`` and an identifier-only index of
      every active decision.
    * ``resume``: full ``next.md`` and ``unresolved.md`` only — Claude
      Code has just restored the transcript itself, so the focus is
      already in the prompt.
    * ``compact``: the most recent ``compact-handover/<id>.md`` (full
      body) plus identifier-only ``working/`` summary. PreCompact in
      Phase 5 actually writes these handover files; until then this
      branch quietly degrades to "no handover available".
    """
    vault_root = settings.vault_path
    payload = InjectionPayload()
    if not vault_root.is_dir():
        return payload

    if source == "resume":
        _append_working(
            payload,
            vault_root,
            identifier_only=False,
            names=("next", "unresolved"),
        )
        return payload

    if source == "compact":
        _append_compact_handover(payload, vault_root)
        _append_working(
            payload,
            vault_root,
            identifier_only=True,
            names=("focus", "next", "unresolved"),
        )
        return payload

    # source in {"startup", "clear"}
    _append_working(
        payload,
        vault_root,
        identifier_only=False,
        names=("focus",),
    )
    _append_working(
        payload,
        vault_root,
        identifier_only=True,
        names=("next", "unresolved"),
    )
    _append_active_decisions_index(payload, vault_root)
    return payload


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _append_working(
    payload: InjectionPayload,
    vault_root: Path,
    *,
    identifier_only: bool,
    names: Sequence[str],
) -> None:
    """Append ``working/<name>.md`` files as session-layer memories."""
    for name in names:
        path = vault_root / "working" / f"{name}.md"
        if not path.is_file():
            continue
        try:
            body = read_plain_markdown(path)
        except KiokuError as exc:
            log.warning("skipping %s: %s", path, exc)
            continue
        if not body.strip():
            continue
        title = first_heading(body) or name
        mtime_iso = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat()
        payload.session_memories.append(
            InjectedMemory(
                id=f"working/{name}",
                source="user-notes",
                trust="high",
                event_at=mtime_iso,
                vault_path=str(path),
                title=title,
                body=None if identifier_only else body,
            )
        )


def _append_compact_handover(
    payload: InjectionPayload,
    vault_root: Path,
) -> None:
    """Append the most recent ``compact-handover/*.md`` as a session memory."""
    handover_dir = vault_root / "compact-handover"
    if not handover_dir.is_dir():
        return
    files = sorted(handover_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if not files:
        return
    most_recent = files[-1]
    try:
        body = read_plain_markdown(most_recent)
    except KiokuError as exc:
        log.warning("skipping compact-handover %s: %s", most_recent, exc)
        return
    title = first_heading(body) or "compact handover"
    mtime_iso = dt.datetime.fromtimestamp(most_recent.stat().st_mtime, tz=dt.UTC).isoformat()
    payload.session_memories.append(
        InjectedMemory(
            id=f"compact-handover/{most_recent.stem}",
            source="auto-extracted",
            trust="medium",
            event_at=mtime_iso,
            vault_path=str(most_recent),
            title=title,
            body=body,
        )
    )


def _append_active_decisions_index(
    payload: InjectionPayload,
    vault_root: Path,
) -> None:
    """Append every active decision as an identifier-only system memory."""
    decisions_dir = vault_root / "semantic" / "decisions" / "active"
    if not decisions_dir.is_dir():
        return
    for path in sorted(decisions_dir.glob("*.md")):
        try:
            record = read_memory(path)
        except KiokuError as exc:
            log.warning("skipping decision %s: %s", path, exc)
            continue
        title = str(record.frontmatter.get("title") or first_heading(record.body) or record.id)
        payload.system_memories.append(
            InjectedMemory(
                id=record.id,
                source=record.source,
                trust=record.trust,
                event_at=str(record.frontmatter.get("event_at", "")),
                vault_path=str(path),
                title=title,
                body=None,
            )
        )
