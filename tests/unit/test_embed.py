"""Tests for ``lib.embed`` — stubbing out the Voyage SDK."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock

import pytest
import voyageai
from kioku.embed import Embedder, _truncate_and_normalize, content_hash
from kioku.errors import EmbedError

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_content_hash_collapses_whitespace() -> None:
    assert content_hash("foo  bar") == content_hash("foo bar")
    assert content_hash("foo\nbar") == content_hash("foo bar")
    assert content_hash("foo\tbar") == content_hash("foo bar")


def test_content_hash_distinguishes_distinct_text() -> None:
    assert content_hash("foo") != content_hash("bar")


def test_truncate_normalizes_to_unit() -> None:
    # ||(3,4)|| = 5 → (0.6, 0.8)
    out = _truncate_and_normalize([3.0, 4.0, 0.0, 0.0], dim=2)
    assert abs(out[0] - 0.6) < 1e-6
    assert abs(out[1] - 0.8) < 1e-6


def test_truncate_passes_through_when_dim_ge_vec() -> None:
    vec = [0.5, 0.5]
    assert _truncate_and_normalize(vec, dim=4) == vec


def test_truncate_rejects_zero_vector() -> None:
    with pytest.raises(EmbedError):
        _truncate_and_normalize([0.0, 0.0, 0.0], dim=2)


# ---------------------------------------------------------------------------
# Embedder + stub Voyage client
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_voyage(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace ``voyageai.Client`` with a stub that returns deterministic vectors."""

    class _Resp:
        def __init__(self, embeddings: list[list[float]]) -> None:
            self.embeddings = embeddings

    client = MagicMock()
    client.embed.side_effect = lambda texts, **_kwargs: _Resp(
        [[float(i + 1)] * 1024 for i, _ in enumerate(texts)]
    )
    monkeypatch.setattr(voyageai, "Client", lambda **_kwargs: client)
    return client


def test_embed_empty_returns_empty() -> None:
    e = Embedder(api_key="x")
    assert e.embed([]) == []


def test_embed_cache_hits_skip_voyage(stub_voyage: MagicMock) -> None:
    cache: dict[tuple[str, str], list[float]] = {}
    e = Embedder(
        api_key="x",
        cache_get=lambda h, m: cache.get((h, m)),
        cache_put=lambda h, m, v: cache.update({(h, m): v}),
    )
    results = e.embed(["alpha", "beta", "alpha"], kind="general", input_type="document")

    assert len(results) == 3
    # Two unique inputs → exactly one batched Voyage call.
    assert stub_voyage.embed.call_count == 1
    # First two are fresh; third is a cache hit.
    assert results[0].cached is False
    assert results[1].cached is False
    assert results[2].cached is True


def test_embed_uses_code_model_when_kind_is_code(stub_voyage: MagicMock) -> None:
    e = Embedder(
        api_key="x",
        model_general="voyage-4-large",
        model_code="voyage-code-3",
    )
    e.embed(["fn foo() {}"], kind="code", input_type="document")
    call_kwargs = stub_voyage.embed.call_args.kwargs
    assert call_kwargs["model"] == "voyage-code-3"


def test_embed_propagates_input_type(stub_voyage: MagicMock) -> None:
    e = Embedder(api_key="x")
    e.embed(["hi"], kind="general", input_type="query")
    assert stub_voyage.embed.call_args.kwargs["input_type"] == "query"


def test_embed_raises_when_voyage_returns_wrong_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Resp:
        embeddings: ClassVar[list[list[float]]] = [[0.1] * 1024]  # 1 result for 2 inputs

    client = MagicMock()
    client.embed.return_value = _Resp()
    monkeypatch.setattr(voyageai, "Client", lambda **_kwargs: client)

    e = Embedder(api_key="x")
    with pytest.raises(EmbedError):
        e.embed(["a", "b"])
