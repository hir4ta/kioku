#!/usr/bin/env bash
# kioku Stop hook wrapper.
#
# Companion to session-end.sh. Stop runs after every assistant turn,
# but we treat it as a chance to materialise the just-finished session
# using *latest-mtime* resolution on the transcript path — a workaround
# for Claude Code issue #8564 (transcript_path can be stale).
#
# Same conventions as session-end.sh: async / fail-open / silent on
# success / stdout {}.

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
    "$kioku_bin" hook stop || {
        echo "kioku: stop hook errored; emitting empty response" >&2
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
else
    "$kioku_bin" hook stop 2>/dev/null || {
        printf '%s\n' "$EMPTY_RESPONSE"
        exit 0
    }
fi
