"""Configuration loader for kioku.

Settings live in three places, looked up in this order on read:

1. **Environment variables** (e.g. ``VOYAGE_API_KEY``, ``KIOKU_VAULT_PATH``)
   — always win, intended for CI / docker overrides.
2. **OS keyring** (macOS Keychain, Linux SecretService, Windows Credential
   Locker) via the :mod:`keyring` library — for secrets only.
3. **``~/.config/kioku/config.toml``** — on-disk source of truth for
   non-secret settings.

The TOML file is validated against :class:`KiokuSettings` (Pydantic v2).
A missing file is **not** an error: defaults populate every field, and a
stub is created on first run by ``kioku scaffold``.

Why Pydantic in addition to ``schemas/settings.schema.json``:

* Pydantic is the in-process source of truth (IDE autocomplete,
  ``mypy --strict`` clean, immutable copies via ``model_copy``).
* ``schemas/settings.schema.json`` is the *external* contract for the
  dashboard's TypeScript code-gen. A CI check (Phase 7+) will assert the
  two are equivalent.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kioku.errors import ConfigError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VAULT_PATH = "~/Documents/Obsidian/Kioku"
DEFAULT_CONFIG_PATH = "~/.config/kioku/config.toml"

ENV_VAULT_PATH = "KIOKU_VAULT_PATH"
ENV_CONFIG_PATH = "KIOKU_CONFIG_PATH"
ENV_VOYAGE_API_KEY = "VOYAGE_API_KEY"

KEYRING_SERVICE_DEFAULT = "kioku"
KEYRING_ACCOUNT_VOYAGE = "voyage_api_key"


# ---------------------------------------------------------------------------
# Pydantic models — one per top-level TOML table.
# extra="forbid" everywhere catches typos at load time instead of letting
# them silently become no-ops.
# ---------------------------------------------------------------------------


class VaultConfig(BaseModel):
    """Vault location and per-walk ignore globs."""

    model_config = ConfigDict(extra="forbid")

    path: str = DEFAULT_VAULT_PATH
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [".obsidian/**", "_meta/**", "**/*.tmp.md"]
    )

    @field_validator("path")
    @classmethod
    def _expand_home(cls, raw: str) -> str:
        return str(Path(raw).expanduser())


class VoyageConfig(BaseModel):
    """Voyage AI model selection and budget."""

    model_config = ConfigDict(extra="forbid")

    model_general: str = "voyage-4-large"
    model_code: str = "voyage-code-3"
    model_rerank: str = "rerank-2.5"
    dim: Literal[256, 512, 1024, 2048] = 1024
    monthly_budget_usd: float = 20.0
    budget_mode: Literal["warn", "fail-closed"] = "warn"


class InjectConfig(BaseModel):
    """Injection-time policy: token cap, top-K, cache hints."""

    model_config = ConfigDict(extra="forbid")

    active_recall_token_cap: int = 32_000
    default_top_k: int = 5
    fetch_pool_size: int = 20
    cache_breakpoints: bool = True


class CronConfig(BaseModel):
    """Scheduled job timing. Strings are crontab-friendly (HH:MM or 'sun HH:MM')."""

    model_config = ConfigDict(extra="forbid")

    nightly_consolidate_time: str = "02:00"
    decay_pass_time: str = "03:00"
    conflict_scan_time: str = "04:00"
    weekly_review_time: str = "sun 09:00"
    benchmark_time: str = "sat 02:00"


class SecretsConfig(BaseModel):
    """Where API keys live."""

    model_config = ConfigDict(extra="forbid")

    storage: Literal["keyring", "env", "file"] = "keyring"
    keyring_service: str = KEYRING_SERVICE_DEFAULT


class ScoringConfig(BaseModel):
    """Generative-Agents-style scoring weights (arxiv 2304.03442).

    Weights are not normalized automatically; ``lib.rank`` applies them
    directly so the user can deliberately bias toward one axis.
    """

    model_config = ConfigDict(extra="forbid")

    relevance: float = 0.40
    recency: float = 0.25
    importance: float = 0.20
    access_freq: float = 0.10
    project_match: float = 0.05
    recency_decay_days: float = 30.0


class KiokuSettings(BaseModel):
    """Root settings model. Loaded by :func:`load_settings`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault: VaultConfig = Field(default_factory=VaultConfig)
    voyage: VoyageConfig = Field(default_factory=VoyageConfig)
    inject: InjectConfig = Field(default_factory=InjectConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)

    @property
    def vault_path(self) -> Path:
        """Resolved absolute vault path. Does not check existence.

        ``expanduser`` is applied here as defense-in-depth: the
        :class:`VaultConfig` validator runs on values read from TOML,
        but Pydantic skips validators on default values unless
        ``validate_default`` is set. This property is the only public
        accessor for the path, so handling expansion here covers both.
        """
        return Path(self.vault.path).expanduser()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    """Return the settings file location, respecting ``$KIOKU_CONFIG_PATH``."""
    override = os.environ.get(ENV_CONFIG_PATH)
    if override:
        return Path(override).expanduser()
    return Path(DEFAULT_CONFIG_PATH).expanduser()


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> KiokuSettings:
    """Load, validate, and env-override the kioku settings.

    Lookup order:

    1. ``path`` argument (used by tests).
    2. ``$KIOKU_CONFIG_PATH`` if set.
    3. ``~/.config/kioku/config.toml`` if it exists.
    4. Pydantic defaults otherwise (no file required).

    After file load, ``$KIOKU_VAULT_PATH`` overrides ``vault.path`` if
    set. This is the single env override we expose for non-secret
    settings — CI / docker workflows rely on it.

    Raises
    ------
    ConfigError
        File exists but cannot be parsed, or validation failed.
    """
    resolved = path or default_config_path()
    if resolved.is_file():
        try:
            with resolved.open("rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"failed to load {resolved}: {exc}") from exc
        try:
            settings = KiokuSettings.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"invalid settings in {resolved}: {exc}") from exc
    else:
        settings = KiokuSettings()

    env_vault = os.environ.get(ENV_VAULT_PATH)
    if env_vault:
        settings = settings.model_copy(
            update={
                "vault": settings.vault.model_copy(
                    update={"path": str(Path(env_vault).expanduser())}
                )
            }
        )

    return settings


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def get_voyage_api_key(settings: KiokuSettings | None = None) -> str:
    """Resolve the Voyage AI API key.

    Lookup order:

    1. ``$VOYAGE_API_KEY`` (always wins; Voyage's own SDK uses this env
       name natively, so kioku stays consistent).
    2. OS keyring (``service=settings.secrets.keyring_service``,
       ``account="voyage_api_key"``).

    There is no on-disk fallback for the API key by design — TOML files
    are easy to leak. Users who really want a file-based store can set
    ``$VOYAGE_API_KEY`` via a ``.env`` loader of their choice.

    Raises
    ------
    ConfigError
        No key found in any tier.
    """
    if env_key := os.environ.get(ENV_VOYAGE_API_KEY):
        return env_key

    resolved = settings or load_settings()
    try:
        import keyring  # noqa: PLC0415 — lazy import to keep startup cheap
    except ImportError as exc:  # pragma: no cover — keyring is a direct dep
        raise ConfigError("keyring is not installed and $VOYAGE_API_KEY is not set") from exc

    keyring_value: str | None = keyring.get_password(
        resolved.secrets.keyring_service, KEYRING_ACCOUNT_VOYAGE
    )
    if keyring_value:
        return keyring_value

    raise ConfigError(
        "Voyage API key not found. "
        f"Set ${ENV_VOYAGE_API_KEY}, or store it in keyring as "
        f"service={resolved.secrets.keyring_service!r}, "
        f"account={KEYRING_ACCOUNT_VOYAGE!r}."
    )


def reset_cache() -> None:
    """Clear the ``load_settings`` cache. Used by tests."""
    load_settings.cache_clear()
