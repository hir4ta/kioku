"""Tests for ``kioku.inject``."""

from __future__ import annotations

from kioku.inject import (
    InjectedMemory,
    InjectionPayload,
    estimate_tokens,
    format_memory,
    format_payload,
    render,
    xml_escape,
)

# ---------------------------------------------------------------------------
# xml_escape
# ---------------------------------------------------------------------------


def test_xml_escape_escapes_all_five_entities() -> None:
    s = "<a href=\"x\" b='y'> & </a>"
    out = xml_escape(s)
    assert "&lt;" in out
    assert "&gt;" in out
    assert "&amp;" in out
    assert "&quot;" in out
    assert "&apos;" in out
    assert "<" not in out
    assert ">" not in out


def test_xml_escape_passes_through_clean_text() -> None:
    assert xml_escape("hello world") == "hello world"


# ---------------------------------------------------------------------------
# format_memory
# ---------------------------------------------------------------------------


def _make(*, body: str | None = None, mid: str = "DEC-2026-05-19-x") -> InjectedMemory:
    return InjectedMemory(
        id=mid,
        source="user-notes",
        trust="high",
        event_at="2026-05-19T10:00:00+00:00",
        vault_path="/vault/x.md",
        title="x",
        body=body,
    )


def test_format_memory_identifier_only_omits_content() -> None:
    rendered = format_memory(_make(body=None))
    assert "<content>" not in rendered
    assert "<title>x</title>" in rendered
    assert "<vault_path>/vault/x.md</vault_path>" in rendered


def test_format_memory_full_body_includes_content() -> None:
    rendered = format_memory(_make(body="# Header\nBody."))
    assert "<content># Header" in rendered
    assert "Body." in rendered


def test_format_memory_provenance_attributes() -> None:
    rendered = format_memory(_make(mid="DEC-x"))
    assert 'id="DEC-x"' in rendered
    assert 'source="user-notes"' in rendered
    assert 'trust="high"' in rendered
    assert 'event_at="2026-05-19T10:00:00+00:00"' in rendered


def test_format_memory_escapes_xml_in_content() -> None:
    rendered = format_memory(_make(body="<bad>'\""))
    assert "&lt;bad&gt;" in rendered
    assert "&apos;" in rendered
    assert "&quot;" in rendered


def test_format_memory_empty_string_body_treated_as_identifier() -> None:
    rendered = format_memory(_make(body=""))
    assert "<content>" not in rendered


# ---------------------------------------------------------------------------
# render / layer construction
# ---------------------------------------------------------------------------


def test_render_always_includes_constraint() -> None:
    out = render(InjectionPayload())
    assert "<system_constraint>" in out
    assert "UNTRUSTED" in out


def test_render_omits_empty_layers() -> None:
    out = render(InjectionPayload())
    assert "<system_memory_layer" not in out
    assert "<session_memory_layer" not in out
    assert "<query_relevant_memory" not in out


def test_render_layer_order_constraint_system_session_query() -> None:
    payload = InjectionPayload(
        system_memories=[_make(mid="SYS-1")],
        session_memories=[_make(mid="SESS-1")],
        query_relevant=[_make(mid="QRY-1")],
    )
    out = render(payload)
    p_constraint = out.index("<system_constraint")
    p_system = out.index("<system_memory_layer")
    p_session = out.index("<session_memory_layer")
    p_query = out.index("<query_relevant_memory")
    assert p_constraint < p_system < p_session < p_query


def test_render_layer_trust_attributes() -> None:
    payload = InjectionPayload(
        system_memories=[_make(mid="SYS-1")],
        session_memories=[_make(mid="SESS-1")],
        query_relevant=[_make(mid="QRY-1")],
    )
    out = render(payload)
    assert 'system_memory_layer trust="system"' in out
    assert 'session_memory_layer trust="harness"' in out
    assert 'query_relevant_memory trust="dynamic"' in out


# ---------------------------------------------------------------------------
# format_payload (with budget enforcement)
# ---------------------------------------------------------------------------


def test_format_payload_under_budget_emits_full_content() -> None:
    payload = InjectionPayload(session_memories=[_make(body="some body")])
    xml, notes = format_payload(payload, token_budget=10_000)
    assert "<content>some body</content>" in xml
    assert notes == []


def test_format_payload_degrades_query_full_body_first() -> None:
    # Each ~20k chars ≈ 5k tokens. With a 6k budget, the session full
    # body fits but a full-body query does not — degradation should hit
    # query first and leave session content intact.
    big = "x" * 20_000
    payload = InjectionPayload(
        session_memories=[_make(body=big, mid="DEC-session-1")],
        query_relevant=[_make(body=big, mid="DEC-query-1")],
    )
    xml, notes = format_payload(payload, token_budget=6_000)
    assert any("query" in n for n in notes)
    # Session full body still present.
    assert "<content>" in xml


def test_format_payload_drops_query_tail_when_degrade_insufficient() -> None:
    # Budget so tight that even identifier-only entries cannot all fit.
    big = "x" * 20_000
    payload = InjectionPayload(
        query_relevant=[
            _make(body=big, mid="DEC-q1"),
            _make(body=big, mid="DEC-q2"),
            _make(body=big, mid="DEC-q3"),
        ],
    )
    xml, notes = format_payload(payload, token_budget=100)
    drop_count = sum(1 for n in notes if "dropped query memory" in n)
    assert drop_count >= 1
    # Constraint always survives.
    assert "<system_constraint>" in xml


def test_format_payload_does_not_mutate_caller_object() -> None:
    payload = InjectionPayload(query_relevant=[_make(body="x" * 10_000)])
    before = len(payload.query_relevant)
    format_payload(payload, token_budget=100)
    assert len(payload.query_relevant) == before


def test_estimate_tokens_grows_with_length() -> None:
    assert estimate_tokens("hi") < estimate_tokens("a" * 100)
