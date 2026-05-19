"""Secrets filter — regex + Shannon entropy.

Applied before any transcript content leaves its in-memory home: before
writing to the vault, before shipping to Voyage for embedding, before
handing off to ``claude -p`` in Phase 5.

Two-pass detection:

1. **Regex pass** matches well-known secret shapes (Anthropic / OpenAI /
   Voyage / AWS / GitHub credentials, JWTs, PEM private keys). Precise
   and cheap.
2. **Entropy pass** flags runs of unusually high Shannon entropy. This
   catches custom token formats and any long random-looking strings
   that didn't trip the regex pass. Tuned with a length floor so prose
   doesn't get false-positive'd.

Matches are replaced with ``[REDACTED:<reason>]`` placeholders so the
surrounding sentence remains readable to a human reviewer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# (compiled regex, redaction reason) pairs.
_REGEX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-ant-(?:api0[34]-)?[A-Za-z0-9_-]{20,}"), "anthropic-key"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"), "openai-key"),
    (re.compile(r"pa-[A-Za-z0-9_-]{20,}"), "voyage-key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws-access-key"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "github-pat"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "github-fine-grained-pat"),
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "jwt"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"), "private-key"),
)

# Entropy pass tuning.
_ENTROPY_THRESHOLD = 4.5  # bits per character
_ENTROPY_MIN_LENGTH = 32  # only inspect tokens at least this long
_TOKEN_RE = re.compile(r"\S+")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character. Empty string → 0."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


@dataclass(slots=True, frozen=True)
class RedactResult:
    """Outcome of a redact pass.

    ``text`` is the redacted body. The two counters expose how many
    redactions each pass made, useful for logging and tests.
    """

    text: str
    n_regex_redactions: int
    n_entropy_redactions: int

    @property
    def total_redactions(self) -> int:
        return self.n_regex_redactions + self.n_entropy_redactions


def redact(text: str) -> RedactResult:
    """Redact secrets from ``text`` using regex then entropy passes."""
    out = text
    n_regex = 0
    for pat, reason in _REGEX_PATTERNS:
        out, n = pat.subn(f"[REDACTED:{reason}]", out)
        n_regex += n

    # Entropy pass — walk tokens, replace high-entropy ones.
    parts: list[str] = []
    last_end = 0
    n_entropy = 0
    for match in _TOKEN_RE.finditer(out):
        token = match.group()
        # Skip tokens that already touch our own placeholders. A regex
        # pass may have produced fragments like ``KEY=[REDACTED:openai-key]``
        # that are still one non-whitespace token; we don't want the
        # entropy pass to clobber them with ``[REDACTED:high-entropy]``.
        if "[REDACTED:" in token:
            parts.append(out[last_end : match.end()])
            last_end = match.end()
            continue
        if len(token) >= _ENTROPY_MIN_LENGTH and _shannon_entropy(token) >= _ENTROPY_THRESHOLD:
            parts.append(out[last_end : match.start()])
            parts.append("[REDACTED:high-entropy]")
            n_entropy += 1
            last_end = match.end()
    parts.append(out[last_end:])
    out = "".join(parts)

    return RedactResult(text=out, n_regex_redactions=n_regex, n_entropy_redactions=n_entropy)
