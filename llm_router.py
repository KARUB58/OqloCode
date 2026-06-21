"""Oqlo Code — Universal Multi-Provider LLM Router.

This is the state-preserving gateway. It keeps a single *canonical* conversation
buffer and, at send time, translates it (and the tool manifest) into whichever
provider dialect is currently active. When a provider rate-limits (HTTP 429),
is unavailable (503), or times out, the router captures the exact buffer and
re-issues the request against the next model in the priority chain — no message
state is lost in the hand-off.

Two wire dialects are implemented natively:

* **openai**  — OpenAI, OpenRouter, and (via a dedicated adapter) Gemini.
* **anthropic** — Anthropic Messages API.

A local **cursor** provider acts as the terminal fallback so the loop always
resolves even with zero cloud connectivity.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import config
from config import ModelProfile, Provider, TOKEN_BUDGET
from telemetry import FinancialVelocityGuard
from tools_manifest import tools_for_dialect

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Canonical conversation model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role == "tool":
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass(slots=True)
class LLMResponse:
    model: ModelProfile
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    cost_usd: float


class Conversation:
    """An append-only canonical message buffer shared across all providers."""

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.messages: list[Message] = []

    def user(self, text: str) -> None:
        self.messages.append(Message(role="user", text=text))

    def assistant(self, text: str, tool_calls: list[ToolCall]) -> None:
        self.messages.append(
            Message(role="assistant", text=text, tool_calls=tool_calls)
        )

    def tool_result(self, call: ToolCall, content: str) -> None:
        self.messages.append(
            Message(
                role="tool",
                text=content,
                tool_call_id=call.id,
                tool_name=call.name,
            )
        )

    def system_error(self, traceback: str) -> None:
        """Inject the self-heal signal as a user-visible system message."""
        self.messages.append(
            Message(role="user", text=f"[SYSTEM ERROR: {traceback}]")
        )


def trim_for_send(conv: Conversation) -> Conversation:
    """Return a token-budgeted shallow copy of ``conv`` for upstream sending.

    Token-saving measure #1: cap the number of trailing messages and truncate
    bulky tool results. The original buffer is never mutated, so full state is
    preserved for fallback and persistence; only the *wire payload* shrinks.

    Care is taken not to orphan a ``tool`` message from the ``assistant`` turn
    that requested it, which providers reject.
    """
    budget = TOKEN_BUDGET
    msgs = conv.messages
    keep_n = max(2, budget.max_history_messages)
    window = msgs[-keep_n:] if len(msgs) > keep_n else list(msgs)

    # Don't lead the window with an orphan tool result.
    while window and window[0].role == "tool":
        window = window[1:]

    trimmed = Conversation(conv.system_prompt)
    cap = budget.max_tool_chars
    for m in window:
        if m.role == "tool" and len(m.text) > cap:
            clipped = m.text[:cap] + f"\n…[truncated {len(m.text) - cap} chars]"
            trimmed.messages.append(
                Message(role="tool", text=clipped,
                        tool_call_id=m.tool_call_id, tool_name=m.tool_name)
            )
        else:
            trimmed.messages.append(m)
    return trimmed


# --------------------------------------------------------------------------- #
# Dialect translation: canonical -> provider payloads
# --------------------------------------------------------------------------- #
def to_openai_messages(conv: Conversation) -> list[dict[str, Any]]:
    """Render the canonical buffer into OpenAI chat-completions messages."""
    out: list[dict[str, Any]] = [
        {"role": "system", "content": conv.system_prompt}
    ]
    for m in conv.messages:
        if m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.text,
                }
            )
        else:  # user / system-as-user
            out.append({"role": m.role, "content": m.text})
    return out


def to_anthropic_messages(conv: Conversation) -> list[dict[str, Any]]:
    """Render the canonical buffer into Anthropic Messages API content blocks."""
    out: list[dict[str, Any]] = []
    for m in conv.messages:
        if m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.text:
                blocks.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.text,
                        }
                    ],
                }
            )
        else:
            out.append({"role": "user", "content": m.text})
    return out


def to_gemini_contents(conv: Conversation) -> list[dict[str, Any]]:
    """Render the canonical buffer into Gemini generateContent `contents`."""
    out: list[dict[str, Any]] = []
    for m in conv.messages:
        if m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.text:
                parts.append({"text": m.text})
            for tc in m.tool_calls:
                parts.append(
                    {"functionCall": {"name": tc.name, "args": tc.arguments}}
                )
            out.append({"role": "model", "parts": parts})
        elif m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": m.tool_name,
                                "response": {"result": m.text},
                            }
                        }
                    ],
                }
            )
        else:
            out.append({"role": "user", "parts": [{"text": m.text}]})
    return out


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
EventCallback = Callable[[str, dict[str, Any]], None]


class ProviderError(Exception):
    """Raised internally to trigger fallback. Carries an HTTP-ish status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class AllProvidersExhausted(Exception):
    """Every model in the chain failed; nothing left to fall back to."""


class LLMRouter:
    """Async multi-provider execution manager with stateful fallback."""

    # Status codes that should trigger a jump to the next provider.
    FALLBACK_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

    def __init__(
        self,
        chain: list[ModelProfile] | None = None,
        *,
        on_event: EventCallback | None = None,
        max_retries_per_model: int = 1,
    ) -> None:
        self.chain = chain or config.build_priority_chain(only_available=True)
        if not self.chain:
            # Local executives are always selectable; Codex is the safety net.
            self.chain = [config.MODELS_BY_SLUG["codex-local"]]
        self.on_event = on_event or (lambda *_: None)
        self.max_retries_per_model = max_retries_per_model
        self.session_usage = Usage()
        self.session_cost_usd: float = 0.0
        # Engine 9: track $/min spend velocity to halt runaway loops.
        self.cost_guard = FinancialVelocityGuard()
        self.active_model: ModelProfile = self.chain[0]
        # When set, caps output tokens for the next call(s) — used by plan().
        self._output_override: int | None = None
        # Module 1: live /model override — when set, the chain is bypassed.
        self.override_model: ModelProfile | None = None
        # When True, tools are omitted from the request (direct-chat mode).
        self._suppress_tools: bool = False
        self._client: Any = (
            httpx.AsyncClient(timeout=httpx.Timeout(60.0)) if httpx else None
        )

    @property
    def effective_chain(self) -> list[ModelProfile]:
        """The chain actually used: a live override bypasses the priority chain."""
        return [self.override_model] if self.override_model else self.chain

    def set_override(self, model: ModelProfile) -> None:
        self.override_model = model
        self.active_model = model

    def clear_override(self) -> None:
        self.override_model = None
        if self.chain:
            self.active_model = self.chain[0]

    def _effective_output(self, model: ModelProfile) -> int:
        if self._output_override is not None:
            return min(model.max_output_tokens, self._output_override)
        return TOKEN_BUDGET.effective_output_tokens(model.max_output_tokens)

    async def plan(self, user_input: str) -> str:
        """Token-saving measure #2: a cheap, capped planning pass.

        Produces a short numbered tool plan on a throwaway buffer with output
        hard-capped at ``plan_max_tokens``. A focused plan reduces wandering in
        the (more expensive) execution loop, so the overall run costs less.
        Best-effort: any failure returns an empty string and execution proceeds.
        """
        scratch = Conversation(config.SYSTEM_PROMPT)
        scratch.user(f"{user_input}\n\n{config.PLAN_PROMPT}")
        prev = self._output_override
        self._output_override = TOKEN_BUDGET.plan_max_tokens
        try:
            resp = await self.complete(scratch)
            return resp.text.strip()
        except Exception:  # noqa: BLE001 - planning is optional.
            return ""
        finally:
            self._output_override = prev

    async def chat(self, conv: Conversation) -> LLMResponse:
        """Direct conversational turn — no tools, straight model response.

        Used by the INTENT_CHAT fast path so casual messages never trigger tool
        calls or the DAG pipeline (and cost fewer tokens).
        """
        prev = self._suppress_tools
        self._suppress_tools = True
        try:
            return await self.complete(conv)
        finally:
            self._suppress_tools = prev

    # --- public API ------------------------------------------------------- #
    async def complete(self, conv: Conversation) -> LLMResponse:
        """Run one turn, walking the fallback chain until one provider answers."""
        last_error: Exception | None = None
        for model in self.effective_chain:
            if not model.is_available:
                self._emit("skip", model=model.name, reason="no credentials")
                continue
            for attempt in range(1, self.max_retries_per_model + 1):
                self.active_model = model
                self._emit("attempt", model=model.name, attempt=attempt)
                started = time.perf_counter()
                try:
                    resp = await self._dispatch(model, conv)
                except ProviderError as exc:
                    last_error = exc
                    self._emit(
                        "fallback",
                        model=model.name,
                        status=exc.status,
                        message=str(exc)[:200],
                    )
                    if exc.status in self.FALLBACK_STATUSES:
                        break  # jump to next model immediately.
                    # Non-fallback error (e.g. 400): also move on, but note it.
                    break
                except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                    last_error = exc
                    self._emit(
                        "fallback", model=model.name, status=0,
                        message=f"{type(exc).__name__}: {exc}"[:200],
                    )
                    break
                else:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    self.session_usage += resp.usage
                    self.session_cost_usd += resp.cost_usd
                    self.cost_guard.record(resp.cost_usd)
                    self._emit(
                        "success",
                        model=model.name,
                        latency_ms=round(elapsed_ms, 1),
                        in_tok=resp.usage.input_tokens,
                        out_tok=resp.usage.output_tokens,
                        cost=round(resp.cost_usd, 6),
                        tool_calls=len(resp.tool_calls),
                    )
                    return resp
        raise AllProvidersExhausted(
            f"All {len(self.effective_chain)} providers failed. "
            f"Last error: {last_error}"
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # --- dispatch --------------------------------------------------------- #
    async def _dispatch(self, model: ModelProfile, conv: Conversation) -> LLMResponse:
        if model.provider.is_local:
            return self._call_local(model, conv)
        if model.provider is Provider.ANTHROPIC:
            return await self._call_anthropic(model, conv)
        if model.provider is Provider.GEMINI:
            return await self._call_gemini(model, conv)
        # OpenAI + OpenRouter share the chat-completions dialect.
        return await self._call_openai_compatible(model, conv)

    def _require_client(self) -> Any:
        if self._client is None:
            raise ProviderError(0, "httpx is not installed; cannot reach network.")
        return self._client

    # --- OpenAI / OpenRouter --------------------------------------------- #
    async def _call_openai_compatible(
        self, model: ModelProfile, conv: Conversation
    ) -> LLMResponse:
        client = self._require_client()
        ep = model.endpoint
        headers = {"Authorization": f"Bearer {ep.api_key}", **ep.extra_headers}
        body: dict[str, Any] = {
            "model": model.slug,
            "messages": to_openai_messages(trim_for_send(conv)),
            "max_tokens": self._effective_output(model),
        }
        if not self._suppress_tools:
            body["tools"] = tools_for_dialect("openai")
            body["tool_choice"] = "auto"
        data = await self._post_json(client, f"{ep.base_url}/chat/completions",
                                     headers, body)
        choice = data["choices"][0]["message"]
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                name=tc["function"]["name"],
                arguments=_safe_json(tc["function"].get("arguments", "{}")),
            )
            for tc in choice.get("tool_calls") or []
        ]
        usage = _usage_from_openai(data.get("usage", {}))
        return self._finish(model, choice.get("content") or "", tool_calls, usage)

    # --- Anthropic -------------------------------------------------------- #
    async def _call_anthropic(
        self, model: ModelProfile, conv: Conversation
    ) -> LLMResponse:
        client = self._require_client()
        ep = model.endpoint
        headers = {"x-api-key": ep.api_key or "", **ep.extra_headers}
        body: dict[str, Any] = {
            "model": model.slug,
            "system": [
                {"type": "text", "text": conv.system_prompt,
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": to_anthropic_messages(trim_for_send(conv)),
            "max_tokens": self._effective_output(model),
        }
        if not self._suppress_tools:
            # Engine 8: mark the stable tool manifest as cacheable so prompt
            # caching reuses those tokens across turns.
            tools = tools_for_dialect("anthropic")
            if tools:
                tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
            body["tools"] = tools
        data = await self._post_json(client, f"{ep.base_url}/messages",
                                     headers, body)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )
        u = data.get("usage", {})
        usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
        )
        return self._finish(model, "".join(text_parts), tool_calls, usage)

    # --- Gemini ----------------------------------------------------------- #
    async def _call_gemini(
        self, model: ModelProfile, conv: Conversation
    ) -> LLMResponse:
        client = self._require_client()
        ep = model.endpoint
        url = (
            f"{ep.base_url}/models/{model.slug}:generateContent"
            f"?key={ep.api_key}"
        )
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": conv.system_prompt}]},
            "contents": to_gemini_contents(trim_for_send(conv)),
            "generationConfig": {
                "maxOutputTokens": self._effective_output(model)
            },
        }
        if not self._suppress_tools:
            decls = [t["function"] for t in tools_for_dialect("openai")]
            body["tools"] = [{"function_declarations": decls}]
        data = await self._post_json(client, url, {}, body)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = data.get("candidates", [])
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []):
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=fc["name"],
                            arguments=fc.get("args", {}),
                        )
                    )
        um = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=um.get("promptTokenCount", 0),
            output_tokens=um.get("candidatesTokenCount", 0),
        )
        return self._finish(model, "".join(text_parts), tool_calls, usage)

    # --- Local executives (Antigravity / Cursor / Codex) ----------------- #
    def _call_local(self, model: ModelProfile, conv: Conversation) -> LLMResponse:
        """Handle a priority 1-3 local provider — zero token cost, no network.

        * Antigravity & Cursor: only respond when explicitly activated by the
          operator (they assume a real local IDE agent is wired in). Otherwise
          they *defer* — raising a fallback so the next tier is tried.
        * Codex: the offline safety net. It defers whenever any cloud API key
          is configured (so the API tier does the real tool work), but answers
          locally when there is no API at all, guaranteeing the loop resolves
          even fully offline instead of crashing.
        """
        provider = model.provider
        if provider in (Provider.ANTIGRAVITY, Provider.CURSOR):
            if not config.is_local_active(provider):
                raise ProviderError(
                    0, f"{model.name} inactive — deferring "
                       f"(enable with /local {provider.value} on)")
            return self._local_plan_response(model, conv)

        # Codex
        if config.any_api_key_configured() and not config.is_local_active(provider):
            raise ProviderError(0, "Codex deferring to configured API tier")
        return self._local_plan_response(model, conv)

    def _local_plan_response(
        self, model: ModelProfile, conv: Conversation
    ) -> LLMResponse:
        last_user = next(
            (m.text for m in reversed(conv.messages) if m.role == "user"), "")
        text = (
            f"[{model.name}] Operating offline with no cloud model, so I cannot "
            "emit tool calls — but no state is lost. Captured request:\n"
            f"{last_user[:400]}\n\n"
            "Add an API key to execute the pipeline, e.g. "
            "/api openrouter <key>."
        )
        return self._finish(model, text, [], Usage())

    # --- shared helpers --------------------------------------------------- #
    async def _post_json(
        self,
        client: Any,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except Exception as exc:  # noqa: BLE001 - network/timeout -> fallback.
            raise ProviderError(0, f"network error: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(resp.status_code, resp.text[:500])
        return resp.json()

    def _finish(
        self,
        model: ModelProfile,
        text: str,
        tool_calls: list[ToolCall],
        usage: Usage,
    ) -> LLMResponse:
        cost = model.estimate_cost(usage.input_tokens, usage.output_tokens)
        return LLMResponse(
            model=model,
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost,
        )

    def _emit(self, event: str, **fields: Any) -> None:
        self.on_event(event, fields)


def _safe_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}


def _usage_from_openai(u: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
    )
