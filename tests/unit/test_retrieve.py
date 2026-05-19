"""Tests for ``kioku.retrieve``."""

from __future__ import annotations

from kioku.retrieve import RRF_K, _parse_iso, _rrf_fuse
from kioku.store_sqlite import SearchHit


def _hit(cid: int) -> SearchHit:
    return SearchHit(
        chunk_id=cid,
        memory_id=f"MEM-{cid}",
        score=1.0,
        body=f"body-{cid}",
        section="body",
        heading_path=(),
    )


def test_rrf_fuse_empty_lists() -> None:
    assert _rrf_fuse([]) == {}
    assert _rrf_fuse([[], []]) == {}


def test_rrf_fuse_single_list_first_hit_has_largest_score() -> None:
    fused = _rrf_fuse([[_hit(1), _hit(2), _hit(3)]])
    assert fused[1] > fused[2] > fused[3]


def test_rrf_fuse_two_lists_overlap_boosts_chunk() -> None:
    # chunk 7 appears at rank 1 in both lists → must outscore solo entries.
    list_a = [_hit(7), _hit(1), _hit(2)]
    list_b = [_hit(7), _hit(3), _hit(4)]
    fused = _rrf_fuse([list_a, list_b])
    assert fused[7] > fused[1]
    assert fused[7] > fused[3]


def test_rrf_fuse_score_formula() -> None:
    fused = _rrf_fuse([[_hit(1)]])
    # Single list, rank 1 → 1 / (K + 1)
    assert abs(fused[1] - 1.0 / (RRF_K + 1)) < 1e-9


def test_parse_iso_valid_round_trip() -> None:
    parsed = _parse_iso("2026-05-19T10:00:00+00:00")
    assert parsed.year == 2026
    assert parsed.month == 5
    assert parsed.day == 19


def test_parse_iso_empty_string_returns_old_date() -> None:
    parsed = _parse_iso("")
    # Should be in the past, well past one year.
    import datetime as dt

    delta = dt.datetime.now(dt.UTC) - parsed
    assert delta.days >= 364


def test_parse_iso_malformed_returns_old_date() -> None:
    import datetime as dt

    parsed = _parse_iso("not a date at all")
    delta = dt.datetime.now(dt.UTC) - parsed
    assert delta.days >= 364
