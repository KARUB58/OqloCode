"""Oqlo Code — Module 1: lightweight Intent Classification Layer.

Every raw (non-slash) terminal line passes through here *before* any LLM call,
so casual conversation never spins up the expensive multi-agent DAG. The
classifier is pure heuristics — zero tokens, sub-millisecond — which is exactly
what an always-on "first hop" router should be.

Result:
* ``INTENT_AGENTIC`` — actionable system operations (build/model/script/debug…).
* ``INTENT_CHAT``    — questions, explanations, chit-chat.
"""

from __future__ import annotations

import enum
import re


class Intent(str, enum.Enum):
    CHAT = "INTENT_CHAT"
    AGENTIC = "INTENT_AGENTIC"


# Verbs that imply the operator wants something *done* on the system.
_ACTION_VERBS = {
    "build", "create", "generate", "make", "model", "render", "export",
    "import", "compile", "debug", "fix", "patch", "write", "code", "script",
    "refactor", "implement", "deploy", "run", "execute", "spawn", "add",
    "instantiate", "bake", "texture", "animate", "rig", "setup", "set up",
    "configure", "install", "automate", "orchestrate", "pipeline", "swarm",
}

# Domain nouns that anchor an action to Oqlo's toolchain.
_DOMAIN_NOUNS = {
    "blender", "unity", "cursor", "mesh", "scene", "prefab", "fbx", "bpy",
    "shader", "material", "asset", "level", "component", "monobehaviour",
    "c#", "csharp", "script", "camera", "lighting", "terrain", "model",
    "texture", "animation", "project", "workspace", "editor", "vertex",
    "polygon", "rigging", "uv", "skybox", "collider",
}

# Strong chat signals (questions / conversational openers).
_CHAT_OPENERS = re.compile(
    r"^\s*(what|who|when|where|why|how|which|whats|what's|is|are|do|does|did|"
    r"can|could|would|should|tell me|explain|define|describe|hi|hey|hello|"
    r"thanks|thank you|good morning|good evening|lol|haha|ok|okay)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z#+]+", text.lower())


def classify_intent(user_input: str) -> Intent:
    """Classify a raw terminal line as chat vs. agentic work."""
    text = user_input.strip()
    if not text:
        return Intent.CHAT

    toks = set(_tokens(text))
    has_action = bool(toks & _ACTION_VERBS) or "c#" in text.lower()
    has_domain = bool(toks & _DOMAIN_NOUNS) or "c#" in text.lower()
    ends_question = text.endswith("?")
    chat_opener = bool(_CHAT_OPENERS.match(text))

    # A clear, domain-anchored instruction is agentic even if phrased politely.
    if has_action and has_domain:
        return Intent.AGENTIC

    # Pure question or conversational opener with no toolchain anchor -> chat.
    if (ends_question or chat_opener) and not has_domain:
        return Intent.CHAT

    # An action verb on its own (e.g. "build a cyberpunk city") leans agentic;
    # imperative phrasing without a question mark reinforces it.
    if has_action and not ends_question:
        return Intent.AGENTIC

    # Domain noun mentioned but no clear action and it's a question -> chat
    # (e.g. "what is a prefab in unity?").
    if has_domain and ends_question:
        return Intent.CHAT

    return Intent.CHAT
