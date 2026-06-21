"""Oqlo Code — central configuration layer.

Loads BYOK credentials, declares the model catalogue as strongly-typed
``ModelProfile`` records, and assembles the ordered ``PRIORITY_CHAIN`` that the
router walks when a provider rate-limits or errors.

The model *slugs* below (e.g. ``claude-4.6-opus``) are intentionally treated as
configuration, not constants baked into logic. Swap them for whatever model IDs
your accounts actually expose; the router never hard-codes a slug.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

try:  # python-dotenv is optional at runtime; degrade quietly if missing.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a hard dep.
    pass


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class Provider(str, enum.Enum):
    """Supported upstream API families.

    The router speaks two wire dialects: the OpenAI "chat completions" schema
    (shared by OpenAI, OpenRouter and — via translation — Gemini) and the
    Anthropic "messages" schema. ``ANTHROPIC`` uses the latter; everything else
    uses the former.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    CURSOR = "cursor"  # local IDE codex fallback — no network call.

    @property
    def wire_dialect(self) -> str:
        """Return the tool/message schema family this provider expects."""
        return "anthropic" if self is Provider.ANTHROPIC else "openai"


class RoutingPolicy(str, enum.Enum):
    """How the priority chain is ordered before execution begins."""

    MAX_INTELLIGENCE = "max_intelligence"
    COST_OPTIMIZATION = "cost_optimization"


# --------------------------------------------------------------------------- #
# Credentials / endpoints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """Resolved connection details for a single provider."""

    provider: Provider
    api_key: str | None
    base_url: str
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_configured(self) -> bool:
        """A network provider is usable only if it has a key. Cursor is local."""
        if self.provider is Provider.CURSOR:
            return True
        return bool(self.api_key)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


ENDPOINTS: Final[dict[Provider, ProviderEndpoint]] = {
    Provider.OPENROUTER: ProviderEndpoint(
        provider=Provider.OPENROUTER,
        api_key=_env("OPENROUTER_API_KEY") or None,
        base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        extra_headers={
            # OpenRouter asks for these for attribution; harmless if generic.
            "HTTP-Referer": "https://github.com/karub58/oqlocode",
            "X-Title": "Oqlo Code",
        },
    ),
    Provider.ANTHROPIC: ProviderEndpoint(
        provider=Provider.ANTHROPIC,
        api_key=_env("ANTHROPIC_API_KEY") or None,
        base_url=_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        extra_headers={
            "anthropic-version": _env("ANTHROPIC_VERSION", "2023-06-01"),
        },
    ),
    Provider.OPENAI: ProviderEndpoint(
        provider=Provider.OPENAI,
        api_key=_env("OPENAI_API_KEY") or None,
        base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ),
    Provider.GEMINI: ProviderEndpoint(
        provider=Provider.GEMINI,
        api_key=_env("GEMINI_API_KEY") or None,
        base_url=_env(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    ),
    Provider.CURSOR: ProviderEndpoint(
        provider=Provider.CURSOR,
        api_key=None,
        base_url="local://cursor",
    ),
}


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A single routable model.

    Attributes
    ----------
    name:
        Human-friendly display name.
    slug:
        The model identifier sent on the wire. Replace with a real model ID.
    provider:
        Which :class:`Provider` serves this model.
    input_cost_per_1m:
        USD per 1,000,000 input tokens. Used by the cost balancer and the
        live consumption table.
    output_cost_per_1m:
        USD per 1,000,000 output tokens.
    intelligence_tier:
        Higher == more capable. Used to order the MAX_INTELLIGENCE policy.
    fallback_tier:
        Position in the default priority chain (0 == primary).
    max_output_tokens:
        Cap requested from the provider per turn.
    """

    name: str
    slug: str
    provider: Provider
    input_cost_per_1m: float
    output_cost_per_1m: float
    intelligence_tier: int
    fallback_tier: int
    max_output_tokens: int = 4096

    @property
    def endpoint(self) -> ProviderEndpoint:
        return ENDPOINTS[self.provider]

    @property
    def is_available(self) -> bool:
        """True when the backing provider has the credentials it needs."""
        return self.endpoint.is_configured

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the USD cost of a single exchange of the given token sizes."""
        return (
            input_tokens / 1_000_000 * self.input_cost_per_1m
            + output_tokens / 1_000_000 * self.output_cost_per_1m
        )


# NOTE: slugs are placeholders for not-yet-released models. Edit freely.
MODEL_CATALOGUE: Final[tuple[ModelProfile, ...]] = (
    ModelProfile(
        name="Claude 4.6 Opus",
        slug="claude-4.6-opus",
        provider=Provider.ANTHROPIC,
        input_cost_per_1m=15.0,
        output_cost_per_1m=75.0,
        intelligence_tier=100,
        fallback_tier=0,
        max_output_tokens=8192,
    ),
    ModelProfile(
        name="GPT-5.5 Pro",
        slug="gpt-5.5-pro",
        provider=Provider.OPENAI,
        input_cost_per_1m=10.0,
        output_cost_per_1m=30.0,
        intelligence_tier=95,
        fallback_tier=1,
        max_output_tokens=8192,
    ),
    ModelProfile(
        name="Gemini 3.5 Flash",
        slug="gemini-3.5-flash",
        provider=Provider.GEMINI,
        input_cost_per_1m=0.35,
        output_cost_per_1m=1.05,
        intelligence_tier=80,
        fallback_tier=2,
        max_output_tokens=8192,
    ),
    ModelProfile(
        name="Cursor Codex (local)",
        slug="cursor-codex",
        provider=Provider.CURSOR,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        intelligence_tier=60,
        fallback_tier=3,
        max_output_tokens=4096,
    ),
)

MODELS_BY_SLUG: Final[dict[str, ModelProfile]] = {
    m.slug: m for m in MODEL_CATALOGUE
}


# --------------------------------------------------------------------------- #
# Priority chain assembly
# --------------------------------------------------------------------------- #
def get_routing_policy() -> RoutingPolicy:
    raw = _env("OQLO_ROUTING_POLICY", RoutingPolicy.MAX_INTELLIGENCE.value)
    try:
        return RoutingPolicy(raw)
    except ValueError:
        return RoutingPolicy.MAX_INTELLIGENCE


def build_priority_chain(
    policy: RoutingPolicy | None = None,
    *,
    only_available: bool = True,
) -> list[ModelProfile]:
    """Return the ordered list of models the router will try, best first.

    ``MAX_INTELLIGENCE`` orders by capability; ``COST_OPTIMIZATION`` orders by
    blended price (cheapest first). When ``only_available`` is set, models whose
    provider lacks credentials are dropped so the chain reflects reality.
    """
    policy = policy or get_routing_policy()
    models = list(MODEL_CATALOGUE)

    if policy is RoutingPolicy.COST_OPTIMIZATION:
        models.sort(
            key=lambda m: (m.input_cost_per_1m + m.output_cost_per_1m,
                           -m.intelligence_tier)
        )
    else:  # MAX_INTELLIGENCE
        models.sort(key=lambda m: (-m.intelligence_tier, m.fallback_tier))

    if only_available:
        models = [m for m in models if m.is_available]
    return models


# Eagerly-built default chain for callers that just want the standard order.
PRIORITY_CHAIN: Final[list[ModelProfile]] = build_priority_chain(
    only_available=False
)


# --------------------------------------------------------------------------- #
# Bridge endpoints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BridgeConfig:
    blender_ws_url: str
    unity_rest_url: str
    cursor_workspace: Path | None
    fbx_export_dir: Path


BRIDGES: Final[BridgeConfig] = BridgeConfig(
    blender_ws_url=_env("OQLO_BLENDER_WS_URL", "ws://127.0.0.1:9876"),
    unity_rest_url=_env("OQLO_UNITY_REST_URL", "http://127.0.0.1:8088"),
    cursor_workspace=(
        Path(_env("OQLO_CURSOR_WORKSPACE")).expanduser()
        if _env("OQLO_CURSOR_WORKSPACE")
        else None
    ),
    fbx_export_dir=Path(
        _env("OQLO_FBX_DIR", str(Path.home() / ".oqlo" / "exports"))
    ).expanduser(),
)


SYSTEM_PROMPT: Final[str] = (
    "You are Oqlo Code, an asynchronous OS orchestrator. You do not write whole "
    "programs from scratch; you drive pre-built automation Skills exposed as "
    "tools across Blender, Unity, and Cursor. Plan compound requests as an "
    "ordered sequence of tool calls, pipe artifacts (FBX paths, asset GUIDs, "
    "log dumps) between steps, and when a tool returns an error wrapped in "
    "[SYSTEM ERROR: ...], diagnose the traceback and re-issue a corrected tool "
    "call instead of giving up. Be terse in prose; let tool calls do the work."
)
