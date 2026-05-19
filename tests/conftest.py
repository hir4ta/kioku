"""Shared pytest fixtures for kioku tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from kioku import config as config_mod


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Empty directory that callers can treat as a vault root."""
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_config_cache() -> Iterator[None]:
    """Clear ``load_settings``'s ``lru_cache`` around every test.

    Without this, a test that mutates env vars or supplies a config
    path would pollute every subsequent test.
    """
    config_mod.reset_cache()
    yield
    config_mod.reset_cache()


@pytest.fixture
def sample_memory_text() -> str:
    """A valid memory Markdown file used across vault + integration tests."""
    return """---
id: DEC-2026-05-19-test-decision
type: decision
status: active
created_at: '2026-05-19T10:00:00+00:00'
event_at: '2026-05-19T10:00:00+00:00'
source: user-notes
trust: high
tags:
  - test
  - voyage
importance: 0.7
pinned: false
---

# Decision: use voyage-4-large

## Decision

We will use voyage-4-large as the default general embedding model in kioku.

## Why

- Anthropic explicitly recommends Voyage as the embedding provider.
- voyage-4-large is the current latest generation (released 2026-01).
- 1024-dim Matryoshka allows truncation to 512 when latency matters.

## Consequences

- POSITIVE: highest retrieval quality on general English text.
- POSITIVE: API contract stays consistent if we later truncate dimensions.
- NEGATIVE: higher per-call latency than voyage-4-lite.

## Context

Earlier drafts considered voyage-3-large but it is now the previous generation.
Code-specific chunks still use voyage-code-3 as voyage-4 has no code variant yet.
"""
