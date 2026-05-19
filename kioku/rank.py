"""Generative-Agents-style scoring for retrieval candidates.

Reference: Park et al. 2023, *Generative Agents: Interactive Simulacra
of Human Behavior* (arxiv:2304.03442). The paper's ablation shows that
recency + importance + relevance is meaningfully better than any one
axis alone. kioku extends that with two practical signals:

* **access_freq**: how often this memory has actually been recalled.
  Frequently-accessed memories drift slightly upward, which gives
  pinned-by-usage behaviour without explicit annotation.
* **project_match**: bias toward memories that belong to the project
  the current Claude Code session is running in.

After the weighted sum, three multiplicative modifiers apply:

* ``pinned=true`` short-circuits to ``score := 1.0`` (always at top).
* ``deprecated_by != None`` multiplies score by ``0.1`` (superseded
  memories are strongly suppressed but not zeroed, so they remain
  surfaceable in archaeological queries).
* ``trust == "low"`` multiplies score by ``0.5`` (untrusted sources
  are de-prioritized but still visible).

The weights live in :class:`kioku.config.ScoringConfig` and are *not*
normalized — biasing toward one axis is a deliberate user choice, not
a bug. The output is a non-negative ``float``; ``1.0`` and above
indicate forced inclusion (currently only via ``pinned``).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from kioku.config import ScoringConfig

# Access count → log scale → /log(100). 100 cumulative accesses ≈ 1.0.
ACCESS_FREQ_SATURATION = 100

# Fallback "old" date for memories with malformed or missing event_at,
# so they age out gracefully rather than crashing the scorer.
_OLD_MEMORY_FALLBACK_DAYS = 365

DEPRECATED_FACTOR = 0.1
LOW_TRUST_FACTOR = 0.5


@dataclass(slots=True, frozen=True)
class ScoreInputs:
    """Per-candidate inputs to :func:`score`.

    All boolean fields default to False so a caller missing a field
    falls through to neutral behaviour.
    """

    relevance: float  # RRF-fused BM25 + dense, ideally in [0, 1]
    event_at: dt.datetime
    importance: float  # 0..1 (defaults to 0.5 in the schema)
    access_count: int  # cumulative
    project_match: bool = False
    pinned: bool = False
    deprecated: bool = False
    trust_low: bool = False


def score(
    inputs: ScoreInputs,
    *,
    scoring: ScoringConfig,
    now: dt.datetime | None = None,
) -> float:
    """Compute the composite score for one candidate."""
    if inputs.pinned:
        return 1.0

    now = now or dt.datetime.now(dt.UTC)
    days_since = max(0.0, (now - inputs.event_at).total_seconds() / 86_400.0)
    recency_decay = math.exp(-days_since / scoring.recency_decay_days)

    # log(count + 1) / log(SAT + 1), capped at 1.0.
    access_freq = math.log(inputs.access_count + 1) / math.log(ACCESS_FREQ_SATURATION + 1)
    access_freq = min(1.0, access_freq)

    project_match_score = 1.0 if inputs.project_match else 0.0

    raw = (
        scoring.relevance * inputs.relevance
        + scoring.recency * recency_decay
        + scoring.importance * inputs.importance
        + scoring.access_freq * access_freq
        + scoring.project_match * project_match_score
    )

    if inputs.deprecated:
        raw *= DEPRECATED_FACTOR
    if inputs.trust_low:
        raw *= LOW_TRUST_FACTOR

    return raw


def fallback_event_at() -> dt.datetime:
    """Return a date well in the past for memories missing ``event_at``."""
    return dt.datetime.now(dt.UTC) - dt.timedelta(days=_OLD_MEMORY_FALLBACK_DAYS)
