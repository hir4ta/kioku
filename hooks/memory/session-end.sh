#!/usr/bin/env bash
# kioku SessionEnd hook wrapper.
#
# Forwards the Claude Code hook payload to ``kioku hook session-end``.
# SessionEnd hooks are async: true / timeout 600s in hooks.json — the
# user's session terminates immediately while we persist the
# transcript in the background.
#
# Conventions match session-start.sh: fail-open, KIOKU_BYPASS=1
# escape hatch, stdout is the (empty) JSON response, stderr is silent
# unless KIOKU_DEBUG=1.

set -u

EMPTY_RESPONSE='{}'

if [ "${KIOKU_BYPASS:-0}" = "1" ]; then
    printf '%s\n' "$EMPTY_RESPONSE"
    exit 0
fi

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
    "$kioku_bin" hook session-end || {
        echo "kioku: session-end errored; emitting empty response" >&2
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
else
    "$kioku_bin" hook session-end 2>/dev/null || {
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
fi
