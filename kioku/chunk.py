"""Chunking — split Markdown bodies into retrieval-sized pieces.

Strategy: **recursive character splitting with parent-child links**.
Separators are tried in this order until each piece fits the budget:

1. Heading boundaries (``\\n# ``, ``\\n## ``, ``\\n### ``, ``\\n#### ``)
2. Paragraph boundary (``\\n\\n``)
3. Single newline (``\\n``)
4. Sentence end (``. ``, ``! ``, ``? ``)
5. Word boundary (`` ``)

This is the LangChain-style ``RecursiveCharacterTextSplitter`` approach,
re-implemented in-tree to avoid the dependency. Output is deterministic.

**Parent-child links** preserve the document hierarchy: the synthetic
"TOC" chunk (index 0) is the parent of each top-level heading chunk;
each heading chunk is the anchor for its descendants. The dashboard uses
parent links to expand a retrieval hit into its containing section
(Anthropic 2026-05 prompt engineering guide recommends quoting before
answering — parent chunks make that cheap).

**Token counts** are a character-based approximation (1 token ≈ 4 chars
for English; code-heavy content overestimates and therefore over-splits,
which is the safe direction). Voyage applies its own tokenizer
server-side, so this count is only used to size chunks for the dense
index. Phase 7 may swap in ``voyageai.Client().count_tokens()`` if
precision becomes load-bearing — for retrieval quality the current
estimate is sufficient.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from kioku.errors import ChunkError

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_TARGET_TOKENS = 512

# Higher index = finer split. Tried in order until each piece fits.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    " ",
)


# ---------------------------------------------------------------------------
# Chunk value type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    """A single chunk to be embedded.

    ``parent_index`` is an in-batch reference (0-based position in the
    returned list) rather than a database FK. The caller assigns DB IDs
    on insert and translates the in-batch reference into a real FK.
    """

    body: str
    section: str  # 'decision' | 'rationale' | 'context' | 'body' | 'toc' | ...
    token_count: int
    heading_path: tuple[str, ...]
    parent_index: int | None = None
    extras: dict[str, str] = field(default_factory=dict)


def estimate_tokens(s: str) -> int:
    """Cheap, deterministic token-count estimate."""
    return max(1, len(s) // CHARS_PER_TOKEN_ESTIMATE)


# ---------------------------------------------------------------------------
# Recursive splitter
# ---------------------------------------------------------------------------


def _split_by_separator(text: str, sep: str) -> list[str]:
    """Split keeping the separator on the right of each piece (except first)."""
    if sep == " ":
        return text.split(" ")
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    out = [parts[0]]
    for piece in parts[1:]:
        out.append(sep + piece)
    return out


def _recursive_split(
    text: str,
    *,
    target_tokens: int,
    separators: Sequence[str],
) -> list[str]:
    """Split ``text`` into pieces each ≤ ``target_tokens`` tokens."""
    if estimate_tokens(text) <= target_tokens:
        return [text]

    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = _split_by_separator(text, sep)
        if len(parts) == 1:
            continue

        # Greedily merge adjacent parts back together up to the budget.
        merged: list[str] = []
        buf = ""
        for p in parts:
            candidate = buf + p
            if estimate_tokens(candidate) <= target_tokens:
                buf = candidate
                continue
            if buf:
                merged.append(buf)
            if estimate_tokens(p) > target_tokens:
                merged.extend(
                    _recursive_split(
                        p,
                        target_tokens=target_tokens,
                        separators=separators[i + 1 :],
                    )
                )
                buf = ""
            else:
                buf = p
        if buf:
            merged.append(buf)
        return merged

    # No separator helped; force-split on character boundary.
    step = target_tokens * CHARS_PER_TOKEN_ESTIMATE
    return [text[i : i + step] for i in range(0, len(text), step)]


# ---------------------------------------------------------------------------
# Heading-aware parsing
# ---------------------------------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class _Section:
    level: int
    title: str
    body: str
    path: tuple[str, ...]


def _parse_sections(text: str) -> list[_Section]:
    """Flatten the document into ``_Section`` objects with heading paths.

    A *preamble* before any heading becomes a level-0 section with empty
    path. Heading paths accumulate ancestors of the same or higher level
    (e.g. an ``###`` under ``## A`` under ``# Title`` has path
    ``("Title", "A", <title>)``).
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [_Section(level=0, title="", body=text, path=())]

    sections: list[_Section] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(_Section(level=0, title="", body=preamble, path=()))

    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(t for _, t in stack)
        sections.append(_Section(level=level, title=title, body=body, path=path))

    return sections


_SECTION_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("decision", "tl;dr", "what was done"), "decision"),
    (("why", "rationale"), "rationale"),
    (("consequence", "trade-off", "tradeoff"), "consequences"),
    (("context", "background"), "context"),
    (("verification", "verify"), "verification"),
    (("next", "todo"), "next"),
    (("open question", "unresolved"), "open"),
)


def _classify_section(heading: str) -> str:
    """Map a heading to a coarse section label used for retrieval scoring."""
    h = heading.lower()
    if h.startswith("ac"):
        return "verification"
    for keywords, label in _SECTION_KEYWORDS:
        if any(kw in h for kw in keywords):
            return label
    return "body"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_body(
    body: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Chunk a Markdown body into retrieval-sized pieces.

    Index 0 is always a synthetic *TOC* chunk holding the heading
    outline (or the preamble if no headings exist). Every other chunk
    references its nearest heading ancestor via ``parent_index``.

    Raises
    ------
    ChunkError
        ``body`` is empty after stripping.
    """
    text = body.strip()
    if not text:
        raise ChunkError("body is empty")

    sections = _parse_sections(text)

    # Synthetic parent chunk: outline if the doc has headings, else the
    # first ~200 chars as a stand-in.
    outline_lines: list[str] = []
    for sec in sections:
        if sec.title:
            outline_lines.append(f"{'#' * sec.level} {sec.title}")
    toc_body = "\n".join(outline_lines) if outline_lines else text[:200]
    chunks: list[Chunk] = [
        Chunk(
            body=toc_body,
            section="toc",
            token_count=estimate_tokens(toc_body),
            heading_path=(),
            parent_index=None,
        )
    ]

    path_to_index: dict[tuple[str, ...], int] = {(): 0}

    for sec in sections:
        if not sec.body:
            continue
        pieces = _recursive_split(
            sec.body,
            target_tokens=target_tokens,
            separators=separators,
        )
        section_label = _classify_section(sec.title) if sec.title else "body"
        parent = path_to_index.get(sec.path[:-1], 0) if sec.path else 0

        for piece in pieces:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    body=piece,
                    section=section_label,
                    token_count=estimate_tokens(piece),
                    heading_path=sec.path,
                    parent_index=parent,
                )
            )
            path_to_index.setdefault(sec.path, idx)

    return chunks
