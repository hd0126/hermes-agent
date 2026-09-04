"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

from gateway.runtime_footer import (
    _home_relative_cwd,
    _model_short,
    build_footer_line,
    format_runtime_footer,
    resolve_footer_config,
)


# ---------------------------------------------------------------------------
# _model_short + _home_relative_cwd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_short_drops_vendor_prefix(model, expected):
    assert _model_short(model) == expected


def test_home_relative_cwd_collapses_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "projects" / "hermes"
    sub.mkdir(parents=True)
    result = _home_relative_cwd(str(sub))
    assert result == "~/projects/hermes"


# ---------------------------------------------------------------------------
# format_runtime_footer
# ---------------------------------------------------------------------------

def test_format_footer_all_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "projects" / "hermes"))
    (tmp_path / "projects" / "hermes").mkdir(parents=True)
    out = format_runtime_footer(
        model="openrouter/openai/gpt-5.4",
        context_tokens=68000,
        context_length=100000,
        cwd=None,  # falls back to TERMINAL_CWD env var
        fields=("model", "context_pct", "cwd"),
    )
    assert out == "gpt-5.4 · 68% · ~/projects/hermes"


def test_format_footer_skips_missing_context_length():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=500,
        context_length=None,
        cwd="/tmp/wd",
        fields=("model", "context_pct", "cwd"),
    )
    # context_pct dropped silently; no "?%" artifact
    assert "%" not in out
    assert "gpt-5.4" in out
    assert "/tmp/wd" in out


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------


def test_resolve_platform_override_wins():
    user = {
        "display": {
            "runtime_footer": {"enabled": True, "fields": ["model"]},
            "platforms": {
                "slack": {"runtime_footer": {"enabled": False}},
            },
        },
    }
    # Telegram picks up the global enable
    assert resolve_footer_config(user, "telegram")["enabled"] is True
    # Slack overrides to off
    assert resolve_footer_config(user, "slack")["enabled"] is False


def test_resolve_platform_can_add_fields_only():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {
                "discord": {"runtime_footer": {"fields": ["context_pct"]}},
            },
        },
    }
    tg = resolve_footer_config(user, "telegram")
    assert tg["enabled"] is True
    assert tg["fields"] == ["model", "context_pct", "cwd"]
    dc = resolve_footer_config(user, "discord")
    assert dc["enabled"] is True
    assert dc["fields"] == ["context_pct"]


# ---------------------------------------------------------------------------
# build_footer_line — top-level entry point used by gateway/run.py
# ---------------------------------------------------------------------------


def test_build_footer_per_platform_off_suppresses():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {"slack": {"runtime_footer": {"enabled": False}}},
        },
    }
    out = build_footer_line(
        user_config=user,
        platform_key="slack",
        model="openai/gpt-5.4",
        context_tokens=10, context_length=100,
        cwd="/tmp",
    )
    assert out == ""



# ---------------------------------------------------------------------------
# latency — opt-in wall-clock turn duration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "<1s"),
        (0.4, "<1s"),
        (0.999, "<1s"),
        (1.0, "1s"),
        (22.0, "22s"),
        (22.4, "22s"),
        (59.4, "59s"),
        (59.6, "1m00s"),
        (60.0, "1m00s"),
        (65.0, "1m05s"),
        (125.0, "2m05s"),
        (3600.0, "60m00s"),
    ],
)
def test_format_latency(seconds, expected):
    from gateway.runtime_footer import _format_latency

    assert _format_latency(seconds) == expected


def test_format_footer_latency_renders():
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=22.0,
        fields=("latency",),
    )
    assert out == "22s"


def test_format_footer_latency_skipped_when_unmeasured():
    """A call site that doesn't measure timing leaves the field out entirely."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=None,
        fields=("latency",),
    )
    assert out == ""


def test_format_footer_latency_skipped_when_negative():
    """A nonsensical (negative) duration is dropped rather than rendered."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=-1.0,
        fields=("latency",),
    )
    assert out == ""


def test_format_footer_latency_zero_renders_sub_second():
    """Zero is a real measurement (a very fast turn), not missing data."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=0.0,
        fields=("latency",),
    )
    assert out == "<1s"


def test_format_footer_latency_in_field_order(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=68_000,
        context_length=100_000,
        cwd=str(tmp_path),
        turn_seconds=65.0,
        fields=("model", "context_pct", "latency", "cwd"),
    )
    assert out == "gpt-5.4 · 68% · 1m05s · ~"


def test_build_footer_line_threads_turn_seconds(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "latency"],
                }
            }
        },
        platform_key="discord",
        model="gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=22.0,
    )
    assert out == "gpt-5.4 · 22s"


def test_format_footer_codex_week_renders_used_and_remaining():
    out = format_runtime_footer(
        model="gpt-5.6-sol",
        context_tokens=68_000,
        context_length=100_000,
        cwd="",
        turn_seconds=22.0,
        codex_week_used_percent=61.0,
        fields=("context_pct", "latency", "codex_week"),
    )
    assert out == "68% · 22s · Codex week 61% used / 39% left"


def test_format_footer_codex_week_skips_missing_usage():
    out = format_runtime_footer(
        model="gpt-5.6-sol",
        context_tokens=0,
        context_length=None,
        cwd="",
        codex_week_used_percent=None,
        fields=("codex_week",),
    )
    assert out == ""


def test_cached_codex_week_usage_refreshes_off_hot_path(monkeypatch):
    from gateway import runtime_footer

    calls = []
    targets = []
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="Weekly", used_percent=61.0),),
    )

    class DeferredThread:
        def __init__(self, *, target, daemon, name):
            assert daemon is True
            targets.append(target)

        def start(self):
            return None

    monkeypatch.setattr(runtime_footer.threading, "Thread", DeferredThread)
    monkeypatch.setattr(
        runtime_footer,
        "fetch_account_usage",
        lambda provider: calls.append(provider) or snapshot,
    )
    runtime_footer.clear_account_usage_cache()

    # The first footer render never performs the network fetch inline.
    assert runtime_footer.get_cached_codex_week_used_percent(ttl_seconds=300) is None
    assert calls == []
    assert len(targets) == 1

    # Simulate completion of the background refresh; subsequent renders reuse it.
    targets[0]()
    assert calls == ["openai-codex"]
    assert runtime_footer.get_cached_codex_week_used_percent(ttl_seconds=300) == 61.0
    assert calls == ["openai-codex"]


def test_cached_codex_week_usage_is_profile_scoped_and_inherits_context(monkeypatch):
    from agent.secret_scope import (
        current_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from gateway import runtime_footer
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    targets = []
    observations = []

    class DeferredThread:
        def __init__(self, *, target, daemon, name):
            assert daemon is True
            targets.append(target)

        def start(self):
            return None

    def fake_fetch(_provider):
        scope = current_secret_scope()
        observations.append((str(get_hermes_home()), dict(scope or {})))
        used = 11.0 if scope and scope.get("TEST_PROFILE") == "a" else 22.0
        return AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime.now(timezone.utc),
            windows=(AccountUsageWindow(label="Weekly", used_percent=used),),
        )

    monkeypatch.setattr(runtime_footer.threading, "Thread", DeferredThread)
    monkeypatch.setattr(runtime_footer, "fetch_account_usage", fake_fetch)
    runtime_footer.clear_account_usage_cache()

    def schedule(profile, marker):
        home_token = set_hermes_home_override(profile)
        secret_token = set_secret_scope({"TEST_PROFILE": marker})
        try:
            return runtime_footer.get_cached_codex_week_used_percent()
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    assert schedule("/tmp/profile-a", "a") is None
    targets.pop(0)()
    assert schedule("/tmp/profile-b", "b") is None
    targets.pop(0)()

    assert observations == [
        ("/tmp/profile-a", {"TEST_PROFILE": "a"}),
        ("/tmp/profile-b", {"TEST_PROFILE": "b"}),
    ]
    assert schedule("/tmp/profile-a", "a") == 11.0
    assert schedule("/tmp/profile-b", "b") == 22.0


def test_cached_codex_week_usage_fails_silently(monkeypatch):
    from gateway import runtime_footer

    def fail(_provider):
        raise RuntimeError("usage backend unavailable")

    monkeypatch.setattr(runtime_footer, "fetch_account_usage", fail)
    runtime_footer.clear_account_usage_cache()
    assert runtime_footer.get_cached_codex_week_used_percent() is None

    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["context_pct", "latency", "codex_week"],
                }
            }
        },
        platform_key="discord",
        model="gpt-5.6-sol",
        context_tokens=68_000,
        context_length=100_000,
        turn_seconds=22.0,
    )
    assert out == "68% · 22s"


def test_build_footer_line_fetches_weekly_only_when_field_enabled(monkeypatch):
    from gateway import runtime_footer

    calls = []
    monkeypatch.setattr(
        runtime_footer,
        "get_cached_codex_week_used_percent",
        lambda: calls.append(True) or 61.0,
    )
    config = {
        "display": {
            "runtime_footer": {
                "enabled": True,
                "fields": ["context_pct", "latency", "codex_week"],
            }
        }
    }
    out = build_footer_line(
        user_config=config,
        platform_key="discord",
        model="gpt-5.6-sol",
        context_tokens=68_000,
        context_length=100_000,
        turn_seconds=22.0,
    )
    assert out == "68% · 22s · Codex week 61% used / 39% left"
    assert calls == [True]


def test_build_footer_line_avoids_usage_fetch_when_field_disabled(monkeypatch):
    from gateway import runtime_footer

    monkeypatch.setattr(
        runtime_footer,
        "get_cached_codex_week_used_percent",
        lambda: pytest.fail("usage API should not be consulted"),
    )
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["context_pct", "latency"],
                }
            }
        },
        platform_key="discord",
        model="gpt-5.6-sol",
        context_tokens=68_000,
        context_length=100_000,
        turn_seconds=22.0,
    )
    assert out == "68% · 22s"


# ---------------------------------------------------------------------------
# Byte-stability: `latency` is opt-in, so the DEFAULT footer is unchanged.
#
# Upstream doctrine: a system prompt / rendered surface must be byte-stable for
# the life of a conversation.  Adding a field to _DEFAULT_FIELDS would silently
# change the footer text of every user who already enabled it.  These tests pin
# the default set and the exact default-config output strings.
# ---------------------------------------------------------------------------

_LEGACY_DEFAULT_FIELDS = ["model", "context_pct", "cwd"]


def test_latency_not_in_default_fields():
    from gateway.runtime_footer import _DEFAULT_FIELDS

    assert "latency" not in _DEFAULT_FIELDS
    assert list(_DEFAULT_FIELDS) == _LEGACY_DEFAULT_FIELDS


def test_resolve_footer_config_default_fields_exclude_latency():
    assert resolve_footer_config({}, "telegram")["fields"] == _LEGACY_DEFAULT_FIELDS
    assert resolve_footer_config(
        {"display": {"runtime_footer": {"enabled": True}}}, "discord"
    )["fields"] == _LEGACY_DEFAULT_FIELDS


@pytest.mark.parametrize(
    "model,tokens,window,cwd,expected",
    [
        ("openai/gpt-5.4", 50_247, 1_000_000, "/var/data", "gpt-5.4 · 5% · /var/data"),
        ("claude-opus-4-8", 68_000, 100_000, "/var/data", "claude-opus-4-8 · 68% · /var/data"),
        ("m", 0, None, "/var/data", "m · /var/data"),
        ("", 10, 100, "/var/data", "10% · /var/data"),
        ("m", 10, 100, "", "m · 10%"),
    ],
)
def test_default_footer_renders_byte_identically(
    monkeypatch, model, tokens, window, cwd, expected
):
    """Default-config output is byte-for-byte what it was before `latency`.

    Note `turn_seconds` IS supplied — proving that even when the caller
    measures timing, a default-configured footer does not show it.
    """
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    out = format_runtime_footer(
        model=model,
        context_tokens=tokens,
        context_length=window,
        cwd=cwd,
        turn_seconds=22.0,
        # fields deliberately NOT passed — exercises the default.
    )
    assert out == expected


def test_default_build_footer_line_ignores_turn_seconds(monkeypatch):
    """build_footer_line with default fields is unaffected by turn_seconds."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    common = dict(
        user_config={"display": {"runtime_footer": {"enabled": True}}},
        platform_key="discord",
        model="openai/gpt-5.4",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd="/var/data",
    )
    baseline = build_footer_line(**common)
    with_timing = build_footer_line(**common, turn_seconds=125.0)
    assert baseline == "gpt-5.4 · 5% · /var/data"
    assert with_timing == baseline
