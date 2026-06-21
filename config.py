"""Oqlo Code — central configuration layer.

Loads BYOK credentials, declares the model catalogue as strongly-typed
``ModelProfile`` records, and assembles the ordered ``PRIORITY_CHAIN`` the router
walks when a provider rate-limits or errors.

Provider priority (token-saving by design — free local executives first, paid
cloud APIs only as a last resort):

    1. Antigravity   (local agent — no token cost)
    2. Cursor        (local agent — no token cost)
    3. Codex         (local executive / offline safety net)
    4. API           (OpenRouter / Anthropic / OpenAI / Gemini — costs tokens)

The model *slugs* (e.g. ``claude-4.6-opus``) are configuration, not constants.
Swap them for whatever model IDs your accounts expose; nothing hard-codes a slug.
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
    """Every routable backend, local or cloud.

    The router speaks three families of wire format:

    * ``local``      — Antigravity / Cursor / Codex. No network, no token spend.
    * ``anthropic``  — Anthropic Messages API.
    * ``openai``     — OpenAI, OpenRouter, and (via adapter) Gemini.
    """

    # --- local executives (priority 1-3) ---
    ANTIGRAVITY = "antigravity"
    CURSOR = "cursor"
    CODEX = "codex"
    # --- cloud APIs (priority 4) ---
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"

    @property
    def is_local(self) -> bool:
        return self in (Provider.ANTIGRAVITY, Provider.CURSOR, Provider.CODEX)

    @property
    def wire_dialect(self) -> str:
        """Return the tool/message schema family this provider expects."""
        if self.is_local:
            return "local"
        if self is Provider.ANTHROPIC:
            return "anthropic"
        return "openai"


class RoutingPolicy(str, enum.Enum):
    """How the priority chain is ordered before execution begins."""

    # Default: honor the explicit 1-4 priority (local first, API last).
    LOCAL_FIRST = "local_first"
    MAX_INTELLIGENCE = "max_intelligence"
    COST_OPTIMIZATION = "cost_optimization"


# --------------------------------------------------------------------------- #
# Runtime credential store (set live via the /api command)
# --------------------------------------------------------------------------- #
# Keys bound at runtime take precedence over environment variables. This is what
# lets `/api openrouter sk-...` work without restarting the process.
_RUNTIME_KEYS: dict[Provider, str] = {}
# Local agents are inactive until enabled (they defer to the next tier).
_LOCAL_ACTIVE: dict[Provider, bool] = {
    Provider.ANTIGRAVITY: os.environ.get("OQLO_ANTIGRAVITY_ACTIVE", "") == "1",
    Provider.CURSOR: os.environ.get("OQLO_CURSOR_ACTIVE", "") == "1",
    Provider.CODEX: os.environ.get("OQLO_CODEX_ACTIVE", "") == "1",
}


def set_runtime_key(provider: Provider, key: str | None) -> None:
    if key:
        _RUNTIME_KEYS[provider] = key.strip()
    else:
        _RUNTIME_KEYS.pop(provider, None)


def set_local_active(provider: Provider, active: bool) -> None:
    if provider.is_local:
        _LOCAL_ACTIVE[provider] = active


def is_local_active(provider: Provider) -> bool:
    return _LOCAL_ACTIVE.get(provider, False)


def any_api_key_configured() -> bool:
    """True if at least one cloud provider has a usable key."""
    return any(
        ENDPOINTS[p].is_configured
        for p in (Provider.OPENROUTER, Provider.ANTHROPIC,
                  Provider.OPENAI, Provider.GEMINI)
    )


# --------------------------------------------------------------------------- #
# Credentials / endpoints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """Resolved connection details for a single provider.

    ``api_key`` is resolved dynamically (runtime store first, then env var) so
    keys bound via the ``/api`` command take effect immediately.
    """

    provider: Provider
    base_url: str
    env_var: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        if self.provider in _RUNTIME_KEYS:
            return _RUNTIME_KEYS[self.provider]
        val = os.environ.get(self.env_var, "").strip() if self.env_var else ""
        return val or None

    @property
    def is_configured(self) -> bool:
        if self.provider.is_local:
            return True  # always selectable; activity is gated separately.
        return bool(self.api_key)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


ENDPOINTS: Final[dict[Provider, ProviderEndpoint]] = {
    Provider.ANTIGRAVITY: ProviderEndpoint(
        provider=Provider.ANTIGRAVITY, base_url="local://antigravity"),
    Provider.CURSOR: ProviderEndpoint(
        provider=Provider.CURSOR, base_url="local://cursor"),
    Provider.CODEX: ProviderEndpoint(
        provider=Provider.CODEX, base_url="local://codex"),
    Provider.OPENROUTER: ProviderEndpoint(
        provider=Provider.OPENROUTER,
        base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        env_var="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://github.com/karub58/oqlocode",
            "X-Title": "Oqlo Code",
        },
    ),
    Provider.ANTHROPIC: ProviderEndpoint(
        provider=Provider.ANTHROPIC,
        base_url=_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        env_var="ANTHROPIC_API_KEY",
        extra_headers={"anthropic-version": _env("ANTHROPIC_VERSION", "2023-06-01")},
    ),
    Provider.OPENAI: ProviderEndpoint(
        provider=Provider.OPENAI,
        base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        env_var="OPENAI_API_KEY",
    ),
    Provider.GEMINI: ProviderEndpoint(
        provider=Provider.GEMINI,
        base_url=_env(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        env_var="GEMINI_API_KEY",
    ),
}

# Map the friendly names accepted by `/api <name> <key>` to providers.
API_PROVIDER_ALIASES: Final[dict[str, Provider]] = {
    "openrouter": Provider.OPENROUTER,
    "anthropic": Provider.ANTHROPIC,
    "antrophic": Provider.ANTHROPIC,  # common misspelling.
    "openai": Provider.OPENAI,
    "gemini": Provider.GEMINI,
    "google": Provider.GEMINI,
}


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A single routable model.

    Attributes
    ----------
    priority_rank:
        The explicit 1-4 tier (1 == tried first). Used by LOCAL_FIRST policy.
    intelligence_tier / fallback_tier:
        Used by the MAX_INTELLIGENCE / COST_OPTIMIZATION policies.
    """

    name: str
    slug: str
    provider: Provider
    input_cost_per_1m: float
    output_cost_per_1m: float
    intelligence_tier: int
    fallback_tier: int
    priority_rank: int
    max_output_tokens: int = 4096

    @property
    def endpoint(self) -> ProviderEndpoint:
        return ENDPOINTS[self.provider]

    @property
    def is_local(self) -> bool:
        return self.provider.is_local

    @property
    def is_available(self) -> bool:
        """True when this model can be placed in the active chain."""
        return self.endpoint.is_configured

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_cost_per_1m
            + output_tokens / 1_000_000 * self.output_cost_per_1m
        )


# NOTE: slugs are placeholders for not-yet-released models. Edit freely.
MODEL_CATALOGUE: Final[tuple[ModelProfile, ...]] = (
    # --- priority 1-3: local executives, zero token cost ---
    ModelProfile(
        name="Antigravity (local)", slug="antigravity-agent",
        provider=Provider.ANTIGRAVITY,
        input_cost_per_1m=0.0, output_cost_per_1m=0.0,
        intelligence_tier=70, fallback_tier=0, priority_rank=1),
    ModelProfile(
        name="Cursor (local)", slug="cursor-agent",
        provider=Provider.CURSOR,
        input_cost_per_1m=0.0, output_cost_per_1m=0.0,
        intelligence_tier=65, fallback_tier=1, priority_rank=2),
    ModelProfile(
        name="Codex (local)", slug="codex-local",
        provider=Provider.CODEX,
        input_cost_per_1m=0.0, output_cost_per_1m=0.0,
        intelligence_tier=60, fallback_tier=2, priority_rank=3),
    # --- priority 4: cloud APIs ---
    ModelProfile(
        name="Claude 4.6 Opus", slug="claude-4.6-opus",
        provider=Provider.ANTHROPIC,
        input_cost_per_1m=15.0, output_cost_per_1m=75.0,
        intelligence_tier=100, fallback_tier=3, priority_rank=4,
        max_output_tokens=8192),
    ModelProfile(
        name="GPT-5.5 Pro", slug="gpt-5.5-pro",
        provider=Provider.OPENAI,
        input_cost_per_1m=10.0, output_cost_per_1m=30.0,
        intelligence_tier=95, fallback_tier=4, priority_rank=4,
        max_output_tokens=8192),
    ModelProfile(
        name="Gemini 3.5 Flash", slug="gemini-3.5-flash",
        provider=Provider.GEMINI,
        input_cost_per_1m=0.35, output_cost_per_1m=1.05,
        intelligence_tier=80, fallback_tier=5, priority_rank=4,
        max_output_tokens=8192),
    ModelProfile(
        name="OpenRouter (auto)", slug="openrouter/auto",
        provider=Provider.OPENROUTER,
        input_cost_per_1m=5.0, output_cost_per_1m=15.0,
        intelligence_tier=85, fallback_tier=6, priority_rank=4,
        max_output_tokens=8192),
)

MODELS_BY_SLUG: Final[dict[str, ModelProfile]] = {
    m.slug: m for m in MODEL_CATALOGUE
}


# --------------------------------------------------------------------------- #
# Priority chain assembly
# --------------------------------------------------------------------------- #
def get_routing_policy() -> RoutingPolicy:
    raw = _env("OQLO_ROUTING_POLICY", RoutingPolicy.LOCAL_FIRST.value)
    try:
        return RoutingPolicy(raw)
    except ValueError:
        return RoutingPolicy.LOCAL_FIRST


def build_priority_chain(
    policy: RoutingPolicy | None = None,
    *,
    only_available: bool = True,
) -> list[ModelProfile]:
    """Return the ordered list of models the router will try, best first.

    * ``LOCAL_FIRST`` (default): honor the explicit 1-4 priority, then
      intelligence as a tie-breaker — free local executives before paid APIs.
    * ``MAX_INTELLIGENCE``: most capable first.
    * ``COST_OPTIMIZATION``: cheapest blended price first.
    """
    policy = policy or get_routing_policy()
    models = list(MODEL_CATALOGUE)

    if policy is RoutingPolicy.COST_OPTIMIZATION:
        models.sort(key=lambda m: (m.input_cost_per_1m + m.output_cost_per_1m,
                                   -m.intelligence_tier))
    elif policy is RoutingPolicy.MAX_INTELLIGENCE:
        models.sort(key=lambda m: (-m.intelligence_tier, m.fallback_tier))
    else:  # LOCAL_FIRST
        models.sort(key=lambda m: (m.priority_rank, -m.intelligence_tier))

    if only_available:
        models = [m for m in models if m.is_available]
    return models


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


# --------------------------------------------------------------------------- #
# Token-saving defaults (everything stays in the terminal — never launch apps)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TokenBudget:
    """Knobs that trade a little quality for fewer tokens. Tunable at runtime."""

    plan_first: bool = True            # cheap planning pass before execution.
    plan_max_tokens: int = 350         # hard cap on the planning response.
    max_history_messages: int = 16     # trim the buffer sent upstream.
    max_tool_chars: int = 1500         # truncate each tool result fed back.
    saver_mode: bool = True            # aggressive trimming + lower caps.
    saver_output_tokens: int = 2048    # output cap when saver_mode is on.

    def effective_output_tokens(self, model_default: int) -> int:
        return min(model_default, self.saver_output_tokens) if self.saver_mode \
            else model_default


TOKEN_BUDGET: Final[TokenBudget] = TokenBudget()

TERMINAL_ONLY: Final[bool] = _env("OQLO_TERMINAL_ONLY", "1") != "0"


PLAN_PROMPT: Final[str] = (
    "Before doing anything, output a SHORT numbered plan (max 6 steps) naming "
    "exactly which tools you will call and in what order. No prose, no code — "
    "just the plan. This keeps the run cheap."
)

SYSTEM_PROMPT: Final[str] = (
    "You are Oqlo Code, an asynchronous OS orchestrator running entirely in a "
    "terminal. You NEVER launch GUI applications; you only talk to already-"
    "running tools over their local bridges and report everything as text. You "
    "do not write whole programs from scratch — you drive pre-built automation "
    "Skills exposed as tools across Blender, Unity, and Cursor. Plan compound "
    "requests as an ordered sequence of tool calls, pipe artifacts (FBX paths, "
    "asset GUIDs, log dumps) between steps, and keep prose minimal to save "
    "tokens. When a tool returns an error wrapped in [SYSTEM ERROR: ...], "
    "diagnose the traceback and re-issue a corrected tool call instead of "
    "giving up."
)
