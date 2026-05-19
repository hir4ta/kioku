"""Tests for ``lib.config``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from kioku.config import (
    ENV_VAULT_PATH,
    ENV_VOYAGE_API_KEY,
    KiokuSettings,
    get_voyage_api_key,
    load_settings,
)
from kioku.errors import ConfigError


def test_load_settings_defaults_when_no_file(tmp_path: Path) -> None:
    settings = load_settings(path=tmp_path / "nonexistent.toml")
    assert isinstance(settings, KiokuSettings)
    assert settings.voyage.model_general == "voyage-4-large"
    assert settings.voyage.dim == 1024


def test_load_settings_parses_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        textwrap.dedent(
            """
            [vault]
            path = "~/some/vault"

            [voyage]
            model_general = "voyage-4"
            dim = 512
            """
        ),
        encoding="utf-8",
    )
    settings = load_settings(path=path)
    assert settings.vault.path.endswith("/some/vault")
    assert settings.voyage.model_general == "voyage-4"
    assert settings.voyage.dim == 512


def test_load_settings_rejects_unknown_top_level(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('bogus_section = true\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path=path)


def test_load_settings_rejects_unknown_subkey(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[voyage]\nbogus_key = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path=path)


def test_env_overrides_vault_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "alt-vault"
    monkeypatch.setenv(ENV_VAULT_PATH, str(override))
    settings = load_settings(path=tmp_path / "nonexistent.toml")
    assert settings.vault.path == str(override)


def test_get_voyage_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VOYAGE_API_KEY, "pa-test")
    assert get_voyage_api_key() == "pa-test"


def test_get_voyage_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VOYAGE_API_KEY, raising=False)
    import keyring

    monkeypatch.setattr(keyring, "get_password", lambda _s, _u: None)
    with pytest.raises(ConfigError, match="Voyage API key not found"):
        get_voyage_api_key()
