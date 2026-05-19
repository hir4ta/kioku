"""Tests for ``lib.chunk``."""

from __future__ import annotations

import pytest
from kioku.chunk import chunk_body, estimate_tokens
from kioku.errors import ChunkError


def test_empty_body_raises() -> None:
    with pytest.raises(ChunkError):
        chunk_body("")


def test_whitespace_only_body_raises() -> None:
    with pytest.raises(ChunkError):
        chunk_body("   \n  \n  ")


def test_short_body_produces_toc_plus_one() -> None:
    chunks = chunk_body("# Title\n\nBody paragraph.")
    assert len(chunks) >= 2
    assert chunks[0].section == "toc"
    assert chunks[0].parent_index is None


def test_madr_sections_get_labels() -> None:
    body = "\n".join(
        [
            "# Decision: do X",
            "## Decision",
            "We will do X.",
            "## Why",
            "Because Y.",
            "## Consequences",
            "POSITIVE: Z.",
            "## Context",
            "Background.",
        ]
    )
    chunks = chunk_body(body)
    sections = {c.section for c in chunks}
    assert "decision" in sections
    assert "rationale" in sections
    assert "consequences" in sections
    assert "context" in sections


def test_parent_links_are_valid_indices() -> None:
    body = "# A\n\ntext one.\n\n## B\n\ntext two.\n\n### C\n\ntext three."
    chunks = chunk_body(body)
    for c in chunks:
        if c.parent_index is not None:
            assert 0 <= c.parent_index < len(chunks)


def test_long_body_is_split() -> None:
    para = ("This is a sentence. " * 200).strip()
    body = f"# Title\n\n{para}"
    chunks = chunk_body(body, target_tokens=128)
    # toc + ≥2 split pieces.
    assert len(chunks) >= 3
    # Each non-toc chunk should be roughly the target size (with slack for boundaries).
    for c in chunks[1:]:
        assert c.token_count <= 256


def test_estimate_tokens_grows_with_length() -> None:
    assert estimate_tokens("hi") < estimate_tokens("a" * 100)
    assert estimate_tokens("a" * 100) <= 100  # 1 token ≈ 4 chars, so 100 chars → ~25 tokens


def test_heading_path_accumulates() -> None:
    body = "# T\n\n## A\n\nfoo.\n\n### B\n\nbar."
    chunks = chunk_body(body)
    paths = {c.heading_path for c in chunks if c.heading_path}
    # We should see both ('T', 'A') and ('T', 'A', 'B') as heading paths.
    assert any(p == ("T", "A") for p in paths)
    assert any(p == ("T", "A", "B") for p in paths)
