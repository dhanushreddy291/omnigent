"""Server-side intelligent model routing.

Infers available model tiers from the session's harness type and makes
a lightweight LLM judge call (cheapest tier) to pick the best model for
each turn. The verdict is applied as ``model_override`` on the runner
body before the turn is forwarded — the runner sees a concrete model,
not a routing config.

No per-agent YAML is needed: tiers are inferred from the harness family,
and the feature is gated by the session's ``cost_control_mode_override``
toggle ("on" = route, anything else = skip).

Server-side LLM credentials come from the same ``llm:`` block in the
server config that policies use::

    # config.yaml
    llm:
      model: databricks-claude-haiku-4-5
      profile: <databricks-profile>

The routing judge reuses the :class:`~omnigent.policies.types.PolicyLLMClient`
built at startup from that config.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)

# ── Tier templates ──────────────────────────────────────────────────────────

# Default model tiers per harness model family.  The server infers the
# family from the session's harness type and uses these defaults.
# Dynamic catalog enumeration (model_catalog) is a future enhancement.
TIER_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "claude": {
        "cheap": ["databricks-claude-haiku-4-5"],
        "medium": ["databricks-claude-sonnet-4-6"],
        "expensive": ["databricks-claude-opus-4-8"],
    },
}

# Harness type → model family.  Only families with a tier template
# above are routable; others silently skip (no routing, no error).
_HARNESS_FAMILY: dict[str, str] = {
    "claude-sdk": "claude",
    "claude_sdk": "claude",
    "claude-native": "claude",
}

# Valid tier names the judge may return.
_VALID_TIERS = frozenset({"cheap", "medium", "expensive"})


def infer_tiers(harness: str | None) -> dict[str, list[str]] | None:
    """Return model tiers for *harness*, or ``None`` if unroutable."""
    if harness is None:
        return None
    family = _HARNESS_FAMILY.get(harness)
    if family is None:
        return None
    return TIER_TEMPLATES.get(family)


# ── Judge rubric ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_TEMPLATE = """\
You are an intelligent model router for a coding assistant.  Given the
user's message, classify its difficulty and pick the best model.

Available tiers (cheapest first):
{tier_menu}

Classification guide:
- **cheap**: trivial questions, greetings, one-line lookups, clarifications,
  conversational follow-ups ("yes", "thanks", "go ahead").
- **medium**: focused single-file changes, writing tasks, moderate analysis,
  explaining code, standard debugging.
- **expensive**: multi-file refactors, architecture design, security audits,
  deep reasoning chains, performance optimization across modules.

Return **strict JSON only** — no markdown, no explanation outside the object:
{{"tier": "<name>", "model": "<id>", "rationale": "<one sentence>"}}
"""


def _build_rubric(tiers: dict[str, list[str]]) -> str:
    """Format the judge system prompt with the tier menu."""
    tier_order = ["cheap", "medium", "expensive"]
    lines = []
    for name in tier_order:
        models = tiers.get(name, [])
        if models:
            lines.append(f"  {name}: {', '.join(models)}")
    return _JUDGE_SYSTEM_TEMPLATE.format(tier_menu="\n".join(lines))


# ── LLM client resolution ──────────────────────────────────────────────────

# The routing judge reuses the server-level PolicyLLMClient built from
# the ``llm:`` config block.  It's resolved lazily on first use and
# cached for the process lifetime (same lifecycle as PolicyLLMClient).


def _get_llm_client() -> Any | None:  # type: ignore[explicit-any]  # PolicyLLMClient
    """Return the server-level PolicyLLMClient, or ``None``.

    Resolved from :attr:`RuntimeCaps.llm` via the same builder the
    policy engine uses.  Returns ``None`` when the server config has
    no ``llm:`` block — routing is silently disabled.
    """
    try:
        from omnigent.runtime import get_runtime_caps
    except ImportError:
        return None
    caps = get_runtime_caps()
    if caps is None:
        return None
    server_llm = caps.llm
    if server_llm is None:
        _logger.debug("smart_routing: no server llm config; routing disabled")
        return None
    # Build a PolicyLLMClient the same way the policy engine does.
    from omnigent.runtime.policies.builder import (
        _build_policy_llm_client,
        _resolve_server_llm_connection,
    )

    conn = _resolve_server_llm_connection(server_llm)
    return _build_policy_llm_client(server_llm, conn)


# Module-level cache for the resolved client.
_llm_client_cache: dict[str, Any] = {}  # type: ignore[explicit-any]


def _resolve_llm_client() -> Any | None:  # type: ignore[explicit-any]
    """Cached version of :func:`_get_llm_client`."""
    if "client" not in _llm_client_cache:
        _llm_client_cache["client"] = _get_llm_client()
    return _llm_client_cache["client"]


def reset_llm_client_cache() -> None:
    """Reset the cached LLM client (for testing)."""
    _llm_client_cache.clear()


# ── Judge LLM call ──────────────────────────────────────────────────────────


async def _call_judge(
    message: str,
    tiers: dict[str, list[str]],
) -> dict[str, Any] | None:
    """Call the server-level LLM as a routing judge.

    Uses the :class:`~omnigent.policies.types.PolicyLLMClient` from
    the server's ``llm:`` config.  Returns the parsed verdict dict or
    ``None`` on any failure (the turn proceeds without routing —
    fail-open).
    """
    llm = _resolve_llm_client()
    if llm is None:
        return None

    rubric = _build_rubric(tiers)
    try:
        response = await llm.create(
            instructions=rubric,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message[:4000],
                        }
                    ],
                }
            ],
            max_output_tokens=256,
        )
        # Extract text from the Responses API result.
        text = response.output_text
        return json.loads(text)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        _logger.warning("smart_routing: judge call failed", exc_info=True)
        return None


# ── Public API ──────────────────────────────────────────────────────────────


async def route_turn(
    harness: str | None,
    user_message: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Pick the best model for a turn.

    :param harness: Canonical harness name, e.g. ``"claude-sdk"``.
    :param user_message: The user's message text (first 4 000 chars used).
    :returns: ``(model_id, verdict_dict)`` when routing applies, or
        ``(None, None)`` when the harness is unroutable, credentials are
        missing, or the judge call fails.
    """
    tiers = infer_tiers(harness)
    if tiers is None:
        return None, None

    verdict = await _call_judge(user_message, tiers)
    if verdict is None:
        return None, None

    model = verdict.get("model")
    tier = verdict.get("tier")
    if not model or not isinstance(model, str):
        _logger.warning("smart_routing: judge returned no model; skipping")
        return None, None
    if tier not in _VALID_TIERS:
        _logger.warning(
            "smart_routing: judge returned unknown tier %r; skipping",
            tier,
        )
        return None, None

    # Clamp: if the model isn't in the declared tier, fall back to the
    # first model of that tier (hallucination guard).
    tier_models = tiers.get(str(tier), [])
    if model not in tier_models and tier_models:
        _logger.info(
            "smart_routing: judge hallucinated model %r for tier %s; clamping to %s",
            model,
            tier,
            tier_models[0],
        )
        model = tier_models[0]

    _logger.info(
        "smart_routing: verdict tier=%s model=%s rationale=%s",
        tier,
        model,
        verdict.get("rationale", ""),
    )
    return model, verdict
