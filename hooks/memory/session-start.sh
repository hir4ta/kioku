#!/usr/bin/env bash
# kioku SessionStart hook wrapper.
#
# Invoked by hooks.json with one argument: the matcher source
# (startup / resume / clear / compact). Reads the Claude Code hook
# payload (JSON) from stdin and forwards it to the Python entry point
# ``kioku hook session-start <source>``.
#
# Conventions:
#   * fail-open: any error degrades to an empty additionalContext, so a
#     broken kioku install never blocks the user from starting a session.
#   * KIOKU_BYPASS=1 short-circuits everything and emits an empty response.
#   * stdout is the JSON response Claude Code expects; everything
#     diagnostic goes to stderr (and is suppressed unless KIOKU_DEBUG=1).

set -u

EMPTY_RESPONSE='{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":""}}'

# Escape hatch — emit an empty SessionStart response and bail out.
if [ "${KIOKU_BYPASS:-0}" = "1" ]; then
    printf '%s\n' "$EMPTY_RESPONSE"
    exit 0
fi

source="${1:-startup}"

# Locate the kioku executable. Order of preference:
#   1. The plugin's own ``.venv`` (created by ``uv sync`` in the
#      plugin install step).
#   2. Any ``kioku`` already on PATH (developer setup).
kioku_bin=""
if [ -n "${CLAUDE_PLUGIN_ROOT:-}" ] && [ -x "${CLAUDE_PLUGIN_ROOT}/.venv/bin/kioku" ]; then
    kioku_bin="${CLAUDE_PLUGIN_ROOT}/.venv/bin/kioku"
elif command -v kioku >/dev/null 2>&1; then
    kioku_bin="$(command -v kioku)"
fi

if [ -z "$kioku_bin" ]; then
    if [ "${KIOKU_DEBUG:-0}" = "1" ]; then
        echo "kioku: executable not found (set CLAUDE_PLUGIN_ROOT or install kioku)" >&2
    fi
    printf '%s\n' "$EMPTY_RESPONSE"
    exit 0
fi

if [ "${KIOKU_DEBUG:-0}" = "1" ]; then
    "$kioku_bin" hook session-start "$source" || {
        echo "kioku: session-start hook errored; emitting empty context" >&2
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
else
    "$kioku_bin" hook session-start "$source" 2>/dev/null || {
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
fi
