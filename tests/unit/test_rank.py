"""Tests for ``kioku.rank``."""

from __future__ import annotations

import datetime as dt

from kioku.config import ScoringConfig
from kioku.rank import ScoreInputs, score


def _now() -> dt.datetime:
    return dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.UTC)


def _inputs(**overrides: object) -> ScoreInputs:
    defaults: dict[str, object] = {
        "relevance": 0.5,
        "event_at": _now(),
        "importance": 0.5,
        "access_count": 0,
        "project_match": False,
        "pinned": False,
        "deprecated": False,
        "trust_low": False,
    }
    defaults.update(overrides)
    return ScoreInputs(**defaults)  # type: ignore[arg-type]


def test_pinned_short_circuits_to_one() -> None:
    s = score(_inputs(pinned=True, relevance=0.0), scoring=ScoringConfig(), now=_now())
    assert s == 1.0


def test_zero_inputs_produces_zero_score() -> None:
    inputs = _inputs(
        relevance=0.0,
        importance=0.0,
        access_count=0,
        event_at=_now() - dt.timedelta(days=10_000),  # ancient → decay ≈ 0
    )
    s = score(inputs, scoring=ScoringConfig(), now=_now())
    assert s < 0.01


def test_recency_decays_exponentially() -> None:
    cfg = ScoringConfig()
    fresh = score(_inputs(event_at=_now()), scoring=cfg, now=_now())
    week_old = score(_inputs(event_at=_now() - dt.timedelta(days=7)), scoring=cfg, now=_now())
    month_old = score(_inputs(event_at=_now() - dt.timedelta(days=30)), scoring=cfg, now=_now())
    assert fresh > week_old > month_old


def test_importance_axis_dominates_with_weight() -> None:
    cfg = ScoringConfig(
        relevance=0.0, recency=0.0, importance=1.0, access_freq=0.0, project_match=0.0
    )
    low = score(_inputs(importance=0.1), scoring=cfg, now=_now())
    high = score(_inputs(importance=0.9), scoring=cfg, now=_now())
    assert high > low * 5  # importance weight is 1.0; 0.9 / 0.1 = 9


def test_access_freq_log_scaled() -> None:
    cfg = ScoringConfig(
        relevance=0.0, recency=0.0, importance=0.0, access_freq=1.0, project_match=0.0
    )
    none = score(_inputs(access_count=0), scoring=cfg, now=_now())
    some = score(_inputs(access_count=10), scoring=cfg, now=_now())
    many = score(_inputs(access_count=100), scoring=cfg, now=_now())
    assert none < some < many
    # log scaling: 100 / 10 should be < 10x.
    assert (many / max(some, 1e-9)) < 10


def test_project_match_only_when_flag_true() -> None:
    cfg = ScoringConfig(
        relevance=0.0, recency=0.0, importance=0.0, access_freq=0.0, project_match=1.0
    )
    no_match = score(_inputs(project_match=False), scoring=cfg, now=_now())
    match = score(_inputs(project_match=True), scoring=cfg, now=_now())
    assert no_match == 0.0
    assert match == 1.0


def test_deprecated_factor_reduces_score_strongly() -> None:
    cfg = ScoringConfig()
    base = score(_inputs(relevance=1.0, importance=1.0), scoring=cfg, now=_now())
    deprecated = score(
        _inputs(relevance=1.0, importance=1.0, deprecated=True),
        scoring=cfg,
        now=_now(),
    )
    assert abs(deprecated - base * 0.1) < 1e-6


def test_low_trust_factor_halves_score() -> None:
    cfg = ScoringConfig()
    trusted = score(_inputs(relevance=1.0), scoring=cfg, now=_now())
    untrusted = score(_inputs(relevance=1.0, trust_low=True), scoring=cfg, now=_now())
    assert abs(untrusted - trusted * 0.5) < 1e-6


def test_both_modifiers_multiply() -> None:
    cfg = ScoringConfig()
    base = score(_inputs(relevance=1.0), scoring=cfg, now=_now())
    both = score(
        _inputs(relevance=1.0, deprecated=True, trust_low=True),
        scoring=cfg,
        now=_now(),
    )
    # 0.1 * 0.5 = 0.05
    assert abs(both - base * 0.05) < 1e-6


def test_future_event_at_does_not_explode_decay() -> None:
    # Clock skew or wrong system time can produce future event_at.
    cfg = ScoringConfig()
    future = score(
        _inputs(event_at=_now() + dt.timedelta(days=30)),
        scoring=cfg,
        now=_now(),
    )
    fresh = score(_inputs(event_at=_now()), scoring=cfg, now=_now())
    # days_since clamps to 0 → both should produce identical recency contribution.
    assert abs(future - fresh) < 1e-6
