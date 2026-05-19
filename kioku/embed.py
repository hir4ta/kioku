"""Voyage AI embedding client.

Wraps :mod:`voyageai` with the conventions kioku needs everywhere:

* ``input_type="document"`` at index time, ``input_type="query"`` at
  retrieval time. Voyage's own docs make the distinction explicit and
  measurable: the wrong setting silently degrades retrieval quality.
* Batched calls (up to :data:`BATCH_SIZE` inputs per request) to amortize
  HTTP overhead.
* Content-hash cache keyed on ``(content_hash, model)``, persisted by
  the caller. Cache hits bypass the network entirely.
* Retry with exponential backoff via :mod:`tenacity` for transient 5xx
  and network errors.
* Matryoshka truncation: when the configured ``dim`` is smaller than
  Voyage's default (1024), the API returns the full vector and we slice
  + re-normalize client-side (Voyage docs recommend this exact pattern).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import voyageai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from kioku.errors import EmbedError

log = logging.getLogger("kioku.embed")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

InputType = Literal["document", "query"]
ModelKind = Literal["general", "code"]

CacheGet = Callable[[str, str], list[float] | None]
"""Callable: ``(content_hash, model) -> cached vector or None``."""

CachePut = Callable[[str, str, list[float]], None]
"""Callable: ``(content_hash, model, vector) -> None``."""


BATCH_SIZE = 128
VOYAGE_DEFAULT_DIM = 1024


@dataclass(slots=True, frozen=True)
class EmbedResult:
    """A single embedding plus provenance."""

    vector: list[float]
    model: str
    dim: int
    cached: bool


@dataclass(slots=True, frozen=True)
class RerankResult:
    """One rerank hit.

    ``index`` references the candidate's position in the input
    ``documents`` list so the caller can map back to chunk_ids etc.
    ``score`` is Voyage's relevance score in [0, 1].
    """

    index: int
    score: float


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def content_hash(text: str, *, normalize: bool = True) -> str:
    """Stable SHA-256 hex digest used as the cache key.

    With ``normalize=True`` (default), whitespace is collapsed before
    hashing so semantically identical inputs share a cache entry. This
    is intentionally lossless w.r.t. embedding quality — Voyage's
    tokenizer normalizes whitespace the same way.
    """
    if normalize:
        text = " ".join(text.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Truncate + renormalize (Matryoshka)
# ---------------------------------------------------------------------------


def _truncate_and_normalize(vec: list[float], dim: int) -> list[float]:
    """Slice ``vec`` to ``dim`` dimensions and re-normalize to unit length.

    Voyage's Matryoshka embeddings are arranged coarse-to-fine within a
    single vector, so naive slicing keeps the most informative axes.
    Re-normalization restores ``||v||₂ = 1`` so cosine similarity
    arithmetic remains numerically stable.
    """
    if dim >= len(vec):
        return vec
    head = vec[:dim]
    norm = math.sqrt(sum(x * x for x in head))
    if norm == 0:
        raise EmbedError("cannot normalize a zero-vector embedding")
    return [x / norm for x in head]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Voyage client with cache and retry, intended to be a long-lived singleton."""

    def __init__(
        self,
        *,
        api_key: str,
        model_general: str = "voyage-4-large",
        model_code: str = "voyage-code-3",
        model_rerank: str = "rerank-2.5",
        dim: int = 1024,
        cache_get: CacheGet | None = None,
        cache_put: CachePut | None = None,
    ) -> None:
        self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._model_general = model_general
        self._model_code = model_code
        self._model_rerank = model_rerank
        self._dim = dim
        self._cache_get: CacheGet = cache_get or (lambda _h, _m: None)
        self._cache_put: CachePut = cache_put or (lambda _h, _m, _v: None)

    def model_for(self, kind: ModelKind) -> str:
        return self._model_code if kind == "code" else self._model_general

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def rerank_model(self) -> str:
        return self._model_rerank

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Re-order ``documents`` against ``query`` via Voyage rerank-2.5.

        Returns :class:`RerankResult` entries ordered by ``score``
        descending. ``index`` references the original position in
        ``documents`` so callers can map results back to their own
        identifiers without round-tripping the body text.

        ``top_k`` defaults to ``len(documents)`` (return the full
        re-ordering). Pass a smaller value if the caller only wants
        the top hits and is OK discarding the tail.
        """
        if not documents:
            return []

        chosen_model = model or self._model_rerank
        k = top_k if top_k is not None else len(documents)

        try:
            response: Any = self._client.rerank(
                query=query,
                documents=documents,
                model=chosen_model,
                top_k=k,
            )
        except Exception as exc:
            log.warning("voyage rerank failed: %s", exc)
            raise EmbedError(str(exc)) from exc

        results = getattr(response, "results", None)
        if results is None:
            raise EmbedError(
                f"voyage rerank response missing 'results' (got {type(response).__name__})"
            )
        return [RerankResult(index=int(r.index), score=float(r.relevance_score)) for r in results]

    def embed(
        self,
        texts: list[str],
        *,
        kind: ModelKind = "general",
        input_type: InputType = "document",
    ) -> list[EmbedResult]:
        """Embed a batch, returning one result per input, order preserved.

        Duplicate inputs within a single batch are deduped on the hash
        key so Voyage sees each unique input exactly once. The first
        occurrence is reported with ``cached=False`` and later ones with
        ``cached=True``, matching the behavior the cache would produce
        across separate calls.
        """
        if not texts:
            return []

        model = self.model_for(kind)
        results: list[EmbedResult | None] = [None] * len(texts)

        # Group miss indices by content hash so each unique input is sent once.
        miss_indices: dict[str, list[int]] = {}
        miss_text: dict[str, str] = {}
        miss_order: list[str] = []

        for i, text in enumerate(texts):
            h = content_hash(text)
            cached = self._cache_get(h, model)
            if cached is not None:
                results[i] = EmbedResult(vector=cached, model=model, dim=len(cached), cached=True)
                continue
            if h not in miss_indices:
                miss_indices[h] = []
                miss_text[h] = text
                miss_order.append(h)
            miss_indices[h].append(i)

        for start in range(0, len(miss_order), BATCH_SIZE):
            batch_hashes = miss_order[start : start + BATCH_SIZE]
            batch_texts = [miss_text[h] for h in batch_hashes]
            vectors = self._call_voyage(batch_texts, model=model, input_type=input_type)
            if len(vectors) != len(batch_hashes):
                raise EmbedError(
                    f"voyage returned {len(vectors)} vectors for {len(batch_hashes)} inputs"
                )
            for h, vec in zip(batch_hashes, vectors, strict=True):
                self._cache_put(h, model, vec)
                for occurrence, idx in enumerate(miss_indices[h]):
                    results[idx] = EmbedResult(
                        vector=vec,
                        model=model,
                        dim=len(vec),
                        cached=occurrence > 0,
                    )

        # Every slot is filled by here. The ``filter not None`` narrows for mypy.
        return [r for r in results if r is not None]

    @retry(
        retry=retry_if_exception_type(EmbedError),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_voyage(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: InputType,
    ) -> list[list[float]]:
        try:
            response: Any = self._client.embed(texts, model=model, input_type=input_type)
        except Exception as exc:
            log.warning("voyage call failed: %s", exc)
            raise EmbedError(str(exc)) from exc

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise EmbedError(
                f"voyage response missing 'embeddings' attribute (got {type(response).__name__})"
            )
        vectors = [list(v) for v in embeddings]
        if self._dim < VOYAGE_DEFAULT_DIM:
            vectors = [_truncate_and_normalize(v, self._dim) for v in vectors]
        return vectors
