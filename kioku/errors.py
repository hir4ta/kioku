"""Exception hierarchy for kioku.

All non-system exceptions raised by kioku derive from :class:`KiokuError`.
System-boundary callers (hook entry points, CLI, cron) catch
:class:`KiokuError` and translate it into a structured exit code / JSON
response. Internal call sites do **not** catch broadly — they let
specific subclasses propagate.

Why a custom hierarchy at all (vs. raising ``ValueError`` / ``RuntimeError``):

* Provenance: ``except VaultError`` is a one-look diagnosis of which layer
  failed, even when the chain is several frames deep.
* Stability: stdlib exception classes are reused by libraries we depend
  on (jsonschema, sqlite3); catching them ambiguously catches their
  errors too.
* User-facing messages: the CLI prints ``error.__class__.__name__`` to
  stderr; a kioku-namespaced class produces a more useful first line
  than ``RuntimeError``.
"""

from __future__ import annotations


class KiokuError(Exception):
    """Base class for all kioku-specific exceptions."""


class ConfigError(KiokuError):
    """Settings file is missing, malformed, or fails schema validation.

    Raised by :mod:`lib.config` when ``~/.config/kioku/config.toml`` cannot
    be loaded, fails to satisfy ``schemas/settings.schema.json``, or
    references a vault path that does not exist.
    """


class SchemaError(KiokuError):
    """A record does not conform to its declared JSON Schema.

    Raised by :mod:`lib.vault` (memory frontmatter), :mod:`lib.classify`
    (conflict records), and :mod:`lib.benchmark` (result records). The
    message includes the schema ``$id`` and the failing JSON Pointer so
    the caller can locate the offending field.
    """


class VaultError(KiokuError):
    """Vault filesystem operation failed (read, write, walk, scaffold).

    Wraps ``OSError`` / ``PermissionError`` with vault-relative context so
    the user sees ``"semantic/decisions/active/foo.md: permission denied"``
    instead of an absolute path.
    """


class ChunkError(KiokuError):
    """Chunking failed (e.g. body could not be tokenized).

    Raised by :mod:`lib.chunk` when tokenizer initialization fails or a
    body is malformed (e.g. unbalanced fences that confuse the recursive
    splitter).
    """


class EmbedError(KiokuError):
    """Voyage API call failed after retries, or returned an unexpected shape.

    Raised by :mod:`lib.embed` when ``tenacity`` exhausts retries on
    network / 5xx errors, or when the response is missing the expected
    ``embeddings`` field. The wrapped ``__cause__`` carries the underlying
    ``httpx.HTTPError``.
    """


class StoreError(KiokuError):
    """SQLite or DuckDB operation failed.

    Raised by :mod:`lib.store_sqlite` / :mod:`lib.store_duckdb`. Wraps
    ``sqlite3.OperationalError`` etc. with the offending SQL truncated to
    a single line so secrets in bound parameters are not leaked.
    """


class TrustError(KiokuError):
    """A memory was rejected by the trust gate (provenance or classifier).

    Raised by :mod:`lib.trust` when:

    * A memory's ``source`` is not in the allow-list for the current
      injection path.
    * The classifier flagged the body as a likely prompt-injection
      attempt (``Ignore previous instructions`` and similar).
    * Shannon entropy of the body exceeds a threshold (suggests an
      embedded secret).

    See ``docs/threat-model.md`` for the full rule set.
    """


class BudgetError(KiokuError):
    """Monthly Voyage budget exceeded.

    Only raised if ``[voyage].budget_mode = "fail-closed"`` in the user
    settings. The default mode (``"warn"``) emits a stderr warning and
    continues, so this exception is rare in practice.
    """
