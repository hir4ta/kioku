"""Tests for ``kioku.redact``."""

from __future__ import annotations

from kioku.redact import _shannon_entropy, redact


def test_plain_prose_passes_through_unchanged() -> None:
    text = "Hello world. This is plain prose with no secrets."
    result = redact(text)
    assert result.text == text
    assert result.total_redactions == 0


def test_redacts_anthropic_api_key() -> None:
    text = "set ANTHROPIC_API_KEY=sk-ant-api03-AbCdEf0123456789AbCdEf0123456789"
    result = redact(text)
    assert "sk-ant-" not in result.text
    assert "[REDACTED:anthropic-key]" in result.text
    assert result.n_regex_redactions >= 1


def test_redacts_voyage_api_key() -> None:
    text = "export VOYAGE_API_KEY=pa-AbCdEfGhIj0123456789KlMnOpQrSt"
    result = redact(text)
    assert "pa-AbCdEf" not in result.text
    assert "[REDACTED:voyage-key]" in result.text


def test_redacts_openai_api_key() -> None:
    text = "sk-proj-AbCdEfGhIj0123456789KlMnOpQrStUvWxYz_-"
    result = redact(text)
    assert "[REDACTED:openai-key]" in result.text


def test_redacts_aws_access_key() -> None:
    text = "AWS_KEY=AKIAIOSFODNN7EXAMPLE somewhere in the log"
    result = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result.text
    assert "[REDACTED:aws-access-key]" in result.text


def test_redacts_github_pat() -> None:
    text = "GITHUB_TOKEN=ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    result = redact(text)
    assert "[REDACTED:github-pat]" in result.text


def test_redacts_jwt() -> None:
    text = "Bearer eyJabc.eyJdef.ghijkl_-mn0123"
    result = redact(text)
    assert "[REDACTED:jwt]" in result.text


def test_redacts_private_key_header() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
    result = redact(text)
    assert "[REDACTED:private-key]" in result.text


def test_entropy_pass_catches_random_token() -> None:
    # 40-char random-looking base64-ish blob; not a known regex shape.
    blob = "kJh78f0LAqW3RpZmYgcAJpcyByYW5kb20gZW51diA="
    result = redact(f"opaque session token: {blob} signed")
    assert "[REDACTED:high-entropy]" in result.text
    assert result.n_entropy_redactions >= 1


def test_entropy_does_not_redact_short_words() -> None:
    # 'Anthropic' is short and low entropy; must survive.
    text = "Anthropic and Voyage cooperate on retrieval."
    result = redact(text)
    assert result.text == text


def test_redact_placeholder_not_re_redacted() -> None:
    # Make sure a placeholder isn't itself flagged by the entropy pass.
    text = "[REDACTED:openai-key]"
    result = redact(text)
    assert result.text == text
    assert result.total_redactions == 0


def test_shannon_entropy_of_uniform_string_is_zero() -> None:
    assert _shannon_entropy("aaaaaaaaaa") == 0.0


def test_shannon_entropy_grows_with_diversity() -> None:
    e_low = _shannon_entropy("aabbccdd")
    e_high = _shannon_entropy("ab" * 4 + "cdefghij")
    assert e_high > e_low
