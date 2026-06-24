"""Shared result type for all bridge tool executions.

Kept in its own module so individual bridge modules can import it without a
circular dependency on the :mod:`bridges` package ``__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    """Uniform outcome of a single tool execution across all bridges."""

    ok: bool
    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    simulated: bool = False

    def to_model_text(self) -> str:
        """Render the result as the text fed back to the model as a tool result.

        Errors are wrapped in the ``[SYSTEM ERROR: ...]`` envelope the system
        prompt instructs the model to self-heal from.
        """
        if not self.ok and self.error:
            return f"[SYSTEM ERROR: {self.error}]"
        prefix = "[SIMULATED] " if self.simulated else ""
        lines = [f"{prefix}{self.summary}"]
        for key, value in self.data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
