"""Oqlo Code bridges — async clients that drive local creative applications.

Every bridge call returns a :class:`ToolResult` rather than raising, so a failed
tool execution flows back into the LLM loop as a self-heal opportunity instead of
crashing the orchestrator.

Each bridge degrades gracefully: if the target application is not reachable on
its local port, the bridge enters ``simulated`` mode, performs whatever portion
of the work it safely can on disk, and clearly flags the result as simulated.
"""

from __future__ import annotations

from typing import Any

from .result import ToolResult
from .blender_bridge import BlenderBridge
from .unity_bridge import UnityBridge
from .cursor_bridge import CursorBridge

__all__ = [
    "ToolResult",
    "BlenderBridge",
    "UnityBridge",
    "CursorBridge",
    "BridgeRouter",
]


class BridgeRouter:
    """Maps tool names to the bridge instance that fulfills them."""

    def __init__(self) -> None:
        self.blender = BlenderBridge()
        self.unity = UnityBridge()
        self.cursor = CursorBridge()
        self._dispatch = {
            "execute_bpy_command": self.blender.execute_bpy_command,
            "export_active_scene_to_fbx": self.blender.export_active_scene_to_fbx,
            "import_asset_to_project": self.unity.import_asset_to_project,
            "create_csharp_script": self.unity.create_csharp_script,
            "get_editor_logs": self.unity.get_editor_logs,
            "patch_workspace_file": self.cursor.patch_workspace_file,
            "set_antigravity_model_preset": self.cursor.set_antigravity_model_preset,
        }

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._dispatch.get(tool_name)
        if handler is None:
            return ToolResult(
                ok=False,
                tool=tool_name,
                summary="Unknown tool",
                error=f"No bridge handler registered for tool '{tool_name}'.",
            )
        try:
            return await handler(**arguments)
        except TypeError as exc:
            # Bad/missing arguments from the model — surface as self-heal signal.
            return ToolResult(
                ok=False,
                tool=tool_name,
                summary="Invalid tool arguments",
                error=f"TypeError calling {tool_name}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop.
            return ToolResult(
                ok=False,
                tool=tool_name,
                summary="Unhandled bridge exception",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def aclose(self) -> None:
        await self.blender.aclose()
        await self.unity.aclose()
