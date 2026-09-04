"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide

Available fields:
    model        — bare model id, vendor prefix dropped (``gpt-5.4``)
    context_pct  — last-call context occupancy as a percent (``5%``)
    latency      — wall-clock duration of the turn (``22s``, ``1m05s``)
    codex_week   — cached Codex weekly usage, when the account API provides it
    cwd          — home-relative working dir (``~``)

``latency`` and ``codex_week`` are opt-in: they are NOT in the default field
set, so a footer whose ``fields`` are unset renders exactly as before. Codex
account usage refreshes in a daemon thread and is cached for five minutes; the
reply hot path never waits for that network request.

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

from __future__ import annotations

import os
import threading
import time
from contextvars import copy_context
from functools import partial
from typing import Any, Iterable, Optional

from agent.account_usage import fetch_account_usage
from hermes_constants import get_hermes_home

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "
_ACCOUNT_USAGE_CACHE_LOCK = threading.Lock()
_ACCOUNT_USAGE_CACHE: dict[str, tuple[float, Optional[float]]] = {}
_ACCOUNT_USAGE_REFRESHING: set[str] = set()


def _account_usage_cache_key() -> str:
    """Return the stable profile identity for the active runtime scope."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(get_hermes_home()))))


def clear_account_usage_cache() -> None:
    """Clear all process-local Codex usage caches (primarily for tests)."""
    with _ACCOUNT_USAGE_CACHE_LOCK:
        _ACCOUNT_USAGE_CACHE.clear()
        _ACCOUNT_USAGE_REFRESHING.clear()


def _refresh_codex_week_usage(cache_key: str) -> None:
    value: Optional[float] = None
    try:
        snapshot = fetch_account_usage("openai-codex")
        if snapshot and snapshot.available:
            for window in snapshot.windows:
                if window.label.strip().lower() == "weekly" and window.used_percent is not None:
                    value = float(window.used_percent)
                    break
    except Exception:
        value = None
    finally:
        with _ACCOUNT_USAGE_CACHE_LOCK:
            _ACCOUNT_USAGE_CACHE[cache_key] = (time.monotonic(), value)
            _ACCOUNT_USAGE_REFRESHING.discard(cache_key)


def get_cached_codex_week_used_percent(*, ttl_seconds: float = 300.0) -> Optional[float]:
    """Return profile-scoped Codex weekly usage without blocking the reply path.

    The refresh worker inherits the caller's contextvars so multiplexed profiles
    retain their Hermes-home and secret scopes. Failed and incomplete lookups are
    cached as ``None`` too, avoiding repeated requests for an unavailable endpoint.
    """
    cache_key = _account_usage_cache_key()
    now = time.monotonic()
    with _ACCOUNT_USAGE_CACHE_LOCK:
        cached = _ACCOUNT_USAGE_CACHE.get(cache_key)
        if cached and now - cached[0] < max(0.0, ttl_seconds):
            return cached[1]
        stale_value = cached[1] if cached else None
        if cache_key not in _ACCOUNT_USAGE_REFRESHING:
            _ACCOUNT_USAGE_REFRESHING.add(cache_key)
            target = partial(
                copy_context().run,
                _refresh_codex_week_usage,
                cache_key,
            )
            try:
                threading.Thread(
                    target=target,
                    daemon=True,
                    name="hermes-codex-usage-refresh",
                ).start()
            except Exception:
                _ACCOUNT_USAGE_REFRESHING.discard(cache_key)
        return stale_value


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]

    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    codex_week_used_percent: Optional[float] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "codex_week":
            if codex_week_used_percent is not None:
                used = max(0, min(100, round(float(codex_week_used_percent))))
                parts.append(f"Codex week {used}% used / {100 - used}% left")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = cfg.get("fields") or _DEFAULT_FIELDS
    codex_week_used_percent = (
        get_cached_codex_week_used_percent() if "codex_week" in fields else None
    )
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_seconds=turn_seconds,
        codex_week_used_percent=codex_week_used_percent,
        fields=fields,
    )
