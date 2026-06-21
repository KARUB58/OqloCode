"""Oqlo Code — the closed-loop orchestration engine.

Ties the :class:`LLMRouter` (brain) to the :class:`BridgeRouter` (hands). Runs
the classic agent loop:

    model -> tool calls -> bridge execution -> results -> model -> ...

until the model stops requesting tools or a step budget is hit. Tool failures
are *not* terminal: their tracebacks are wrapped as ``[SYSTEM ERROR: ...]`` and
fed back so the active model can self-heal and re-issue a corrected call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import config
from bridges import BridgeRouter, ToolResult
from llm_router import Conversation, LLMResponse, LLMRouter, ToolCall
from memory_rag import MemoryRAG


@dataclass(slots=True)
class StepRecord:
    """One observable event in an orchestration run, for UI rendering."""

    kind: str  # "assistant" | "tool" | "heal" | "final"
    text: str = ""
    tool: str = ""
    ok: bool = True
    simulated: bool = False
    model: str = ""
    data: dict[str, Any] = field(default_factory=dict)


StepCallback = Callable[[StepRecord], None]


class Orchestrator:
    """Drives a single user request to completion across model + bridges."""

    def __init__(
        self,
        router: LLMRouter,
        bridges: BridgeRouter,
        conversation: Conversation,
        *,
        max_steps: int = 12,
        max_heals_per_tool: int = 2,
        on_step: StepCallback | None = None,
        memory: MemoryRAG | None = None,
    ) -> None:
        self.router = router
        self.bridges = bridges
        self.conv = conversation
        self.max_steps = max_steps
        self.max_heals_per_tool = max_heals_per_tool
        self.on_step = on_step or (lambda _: None)
        self.memory = memory
        self._heal_counts: dict[str, int] = {}

    async def run(self, user_input: str) -> str:
        """Process one compound user request; return the final assistant text."""
        self._heal_counts.clear()
        final_text = ""

        # Token-saving: draft a cheap, capped plan first so the execution loop
        # stays focused (fewer iterations -> fewer tokens).
        if config.TOKEN_BUDGET.plan_first:
            plan = await self.router.plan(user_input)
            if plan and not plan.startswith("["):  # skip local "offline" stubs.
                self._emit(StepRecord(kind="plan", text=plan,
                                      model=self.router.active_model.name))
                user_input = (
                    f"{user_input}\n\n[Approved plan — follow it concisely]\n{plan}"
                )

        # Engine 6: inject proven patterns from long-term memory so the model
        # reuses historically successful structures instead of re-deriving them.
        if self.memory is not None:
            ctx = self.memory.context_block(user_input, top_k=3)
            if ctx:
                self._emit(StepRecord(kind="memory", text=ctx))
                user_input = f"{user_input}\n\n{ctx}"

        self.conv.user(user_input)

        for _ in range(self.max_steps):
            # Engine 9: refuse to continue a runaway, money-burning loop.
            frozen, msg = self.router.cost_guard.check()
            if frozen:
                final_text = (
                    f"⛔ {msg}\nExecution frozen for Human-In-The-Loop review. "
                    "Resume with /budget resume after inspecting the run."
                )
                self._emit(StepRecord(kind="freeze", ok=False, text=final_text))
                break

            response: LLMResponse = await self.router.complete(self.conv)
            self.conv.assistant(response.text, response.tool_calls)

            if response.text:
                self._emit(
                    StepRecord(kind="assistant", text=response.text,
                               model=response.model.name)
                )

            if not response.tool_calls:
                final_text = response.text
                self._emit(StepRecord(kind="final", text=final_text,
                                      model=response.model.name))
                break

            # Execute every requested tool call, feeding results back.
            for call in response.tool_calls:
                result = await self._execute_with_heal(call)
                self.conv.tool_result(call, result.to_model_text())
                if result.ok and not result.simulated:
                    self._maybe_remember(call)
        else:
            final_text = (
                "Reached the maximum step budget before the task fully resolved. "
                "Inspect the step log above for where it stalled."
            )
            self._emit(StepRecord(kind="final", text=final_text))

        return final_text

    def _maybe_remember(self, call: ToolCall) -> None:
        """Engine 6: commit a proven code snippet to long-term memory."""
        if self.memory is None:
            return
        if call.name == "execute_bpy_command" and call.arguments.get("code"):
            self.memory.remember(call.arguments["code"], kind="code",
                                 tags=["blender", "proven"])
        elif call.name == "create_csharp_script" and call.arguments.get("source"):
            self.memory.remember(call.arguments["source"], kind="csharp",
                                 tags=["unity", "proven"])

    async def _execute_with_heal(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, tracking self-heal budget per tool name."""
        result = await self.bridges.execute(call.name, call.arguments)
        self._emit(
            StepRecord(
                kind="tool",
                tool=call.name,
                ok=result.ok,
                simulated=result.simulated,
                text=result.summary,
                data=result.data,
            )
        )
        if not result.ok:
            count = self._heal_counts.get(call.name, 0)
            if count < self.max_heals_per_tool:
                self._heal_counts[call.name] = count + 1
                self._emit(
                    StepRecord(
                        kind="heal",
                        tool=call.name,
                        ok=False,
                        text=f"Self-heal attempt {count + 1} queued for "
                             f"{call.name}: {result.error}",
                    )
                )
            # Either way the error text (wrapped) is returned so the model sees it.
        return result

    def _emit(self, record: StepRecord) -> None:
        self.on_step(record)
