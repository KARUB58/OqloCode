"""Blender bridge — drives a live Blender instance over a local WebSocket.

A small addon (see :data:`BLENDER_ADDON_TEMPLATE`) runs inside Blender and
listens on ``ws://127.0.0.1:9876``. This module connects as a client, ships
``bpy`` snippets for execution, and triggers FBX exports. When Blender is not
listening, every method falls back to a simulated result so the orchestration
loop keeps flowing.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Any

import config
from .result import ToolResult

try:
    import websockets
    from websockets.exceptions import WebSocketException
except Exception:  # pragma: no cover - websockets is an optional runtime dep.
    websockets = None  # type: ignore[assignment]

    class WebSocketException(Exception):  # type: ignore[no-redef]
        ...


# The addon a user drops into Blender (Edit > Preferences > Add-ons > Install).
# It exposes a JSON command server so this bridge can drive a *running* Blender.
BLENDER_ADDON_TEMPLATE: str = textwrap.dedent(
    '''
    bl_info = {"name": "Oqlo Bridge", "blender": (3, 0, 0), "category": "System"}
    import asyncio, json, threading, traceback, bpy

    async def _handle(ws):
        async for raw in ws:
            msg = json.loads(raw)
            op, payload = msg.get("op"), msg.get("payload", {})
            try:
                if op == "exec":
                    scope = {"bpy": bpy}
                    exec(payload["code"], scope)
                    await ws.send(json.dumps({"ok": True, "stdout": "executed"}))
                elif op == "export_fbx":
                    bpy.ops.export_scene.fbx(
                        filepath=payload["filepath"],
                        use_selection=payload.get("selected_only", False),
                        use_mesh_modifiers=payload.get("apply_modifiers", True),
                    )
                    await ws.send(json.dumps({"ok": True, "path": payload["filepath"]}))
                else:
                    await ws.send(json.dumps({"ok": False, "error": "unknown op"}))
            except Exception:
                await ws.send(json.dumps({"ok": False, "error": traceback.format_exc()}))

    def _serve():
        import websockets
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websockets.serve(_handle, "127.0.0.1", 9876))
        loop.run_forever()

    def register():
        threading.Thread(target=_serve, daemon=True).start()

    def unregister():
        pass
    '''
).strip()


class BlenderBridge:
    """Async client for a live Blender WebSocket addon."""

    def __init__(self, ws_url: str | None = None) -> None:
        self.ws_url = ws_url or config.BRIDGES.blender_ws_url
        self.export_dir = config.BRIDGES.fbx_export_dir
        self._conn: Any = None

    # --- connection management -------------------------------------------- #
    async def _send(self, op: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Send one command, returning the parsed reply or ``None`` if offline."""
        if websockets is None:
            return None
        try:
            async with websockets.connect(self.ws_url, open_timeout=2) as ws:
                await ws.send(json.dumps({"op": op, "payload": payload}))
                reply = await asyncio.wait_for(ws.recv(), timeout=30)
                return json.loads(reply)
        except (OSError, WebSocketException, asyncio.TimeoutError):
            return None

    # --- tools ------------------------------------------------------------ #
    async def execute_bpy_command(
        self, code: str, description: str | None = None
    ) -> ToolResult:
        reply = await self._send("exec", {"code": code})
        if reply is None:
            # Blender offline: validate syntax locally so we can still self-heal
            # on obvious mistakes, then report a simulated success.
            try:
                compile(code, "<bpy-snippet>", "exec")
            except SyntaxError as exc:
                return ToolResult(
                    ok=False,
                    tool="execute_bpy_command",
                    summary="bpy snippet failed to compile",
                    error=f"SyntaxError: {exc}",
                )
            return ToolResult(
                ok=True,
                tool="execute_bpy_command",
                summary=description or "Executed bpy snippet (Blender not running).",
                data={"lines": len(code.splitlines())},
                simulated=True,
            )
        if not reply.get("ok"):
            return ToolResult(
                ok=False,
                tool="execute_bpy_command",
                summary="bpy execution raised inside Blender",
                error=reply.get("error", "unknown Blender error"),
            )
        return ToolResult(
            ok=True,
            tool="execute_bpy_command",
            summary=description or "Executed bpy snippet in live Blender.",
            data={"stdout": reply.get("stdout", "")},
        )

    async def export_active_scene_to_fbx(
        self,
        filename: str,
        selected_only: bool = False,
        apply_modifiers: bool = True,
    ) -> ToolResult:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        target = self.export_dir / f"{filename}.fbx"
        reply = await self._send(
            "export_fbx",
            {
                "filepath": str(target),
                "selected_only": selected_only,
                "apply_modifiers": apply_modifiers,
            },
        )
        if reply is None:
            # Write a placeholder so downstream Unity import has a real path.
            target.write_bytes(b"; Oqlo simulated FBX placeholder\n")
            return ToolResult(
                ok=True,
                tool="export_active_scene_to_fbx",
                summary="Exported scene to FBX (simulated placeholder).",
                data={"fbx_path": str(target)},
                simulated=True,
            )
        if not reply.get("ok"):
            return ToolResult(
                ok=False,
                tool="export_active_scene_to_fbx",
                summary="FBX export failed inside Blender",
                error=reply.get("error", "unknown export error"),
            )
        return ToolResult(
            ok=True,
            tool="export_active_scene_to_fbx",
            summary="Exported active scene to FBX.",
            data={"fbx_path": reply.get("path", str(target))},
        )

    async def aclose(self) -> None:
        # Connections are short-lived (per-command); nothing persistent to close.
        return None
